import os
os.environ["TORCH_CUDNN_V8_API_ENABLED"] = "0" 
import cv2
import numpy as np
import pandas as pd
import torch

# Disabilitiamo CUDNN per evitare il mismatch di librerie
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path

from facenet_pytorch import InceptionResnetV1
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# Utilizziamo PYTHONPATH=src per gli import
from util.plot.utils_plot_specific import (
    plot_targeted_success_curve,
    plot_target_confidence_growth,
    plot_source_target_heatmap,
    plot_attack_outcome_distribution,
    plot_vulnerability_vs_epsilon_heatmap
)
from util.plot.utils_plot_shared import (
    plot_adversarial_showcase,
    plot_frequency_spectrum,
    plot_gradcam_shift,
    plot_latent_trajectory,
    plot_round_robin_plotly_grouped
)

def main():
    print("======================================================")
    print(" METRICHE & PLOT: PGD TARGETED (Error-Specific)       ")
    print("======================================================\n")

    # =========================================================
    # BLOCCO 0: SETUP E CARICAMENTO CSV DISTRIBUITI
    # =========================================================
    base_dir = Path(os.getcwd())
    
    # MODIFICA 1: Puntiamo alle cartelle PGD invece che FGSM
    base_attacks_dir = base_dir / "dataset" / "attacks" / "error_specific" / "pgd"
    base_plots_dir = base_dir / "plots" / "3_Adversarial_Examples" / "error_specific" / "pgd"
    
    # MODIFICA 2: Le strategie che hai effettivamente generato con PGD
    # (La logica per rr_* è stata lasciata nel codice nel caso ti servisse in futuro)
    strategies = ["next_best", "least-likely", "random"]

    # Inizializziamo il modello una sola volta fuori dal ciclo per risparmiare tempo e VRAM
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"-> Inizializzazione NN1 globale su {device}...")
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    resnet.classify = True 

    for strategy in strategies:
        print(f"\n======================================================")
        print(f" AVVIO VALUTAZIONE STRATEGIA: {strategy.upper()}")
        print(f"======================================================")
        
        # =========================================================
        # BLOCCO 0: SETUP E CARICAMENTO CSV DISTRIBUITI
        # =========================================================
        attacks_dir = base_attacks_dir / strategy
        output_eval_dir = base_plots_dir / strategy
        
        progression_dir = output_eval_dir / "visual_progression"
        explain_dir = output_eval_dir / "explainability"
        
        for d in [output_eval_dir, progression_dir, explain_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # 0a. Ricerca di tutti i CSV di tracciamento nelle cartelle eps_X_XXX per la strategia corrente
        tracker_files = list(attacks_dir.glob("eps_*/tracker_eps_*.csv"))
        
        if not tracker_files:
            print(f"[WARNING] Nessun file tracker trovato in {attacks_dir}. Salto strategia.")
            continue

        print(f"-> Trovati {len(tracker_files)} file tracker locali. Unione in corso...")
        
        # 0b. Lettura e concatenazione in un unico DataFrame temporaneo
        df_list = [pd.read_csv(f) for f in tracker_files]
        df = pd.concat(df_list, ignore_index=True)
        
        # Assicuriamoci che gli epsilon siano ordinati dal più piccolo al più grande
        epsilons = sorted(df['eps'].unique())
        print(f"-> Epsilon rilevati: {epsilons}")
        
        # Pre-inizializziamo le colonne se non esistono
        if 'adv_pred_class' not in df.columns:
            df['clean_pred_class'] = -1 
            df['adv_pred_class'] = -1
            df['target_confidence'] = 0.0

        # =========================================================
        # BLOCCO 1: INFERENZA IN BATCH E AGGIORNAMENTO DATI
        # =========================================================
        batch_size = 64 
        print(f"\n[BLOCCO 1 - {strategy.upper()}] Inferenza delle immagini avversarie e originali...")

        with torch.no_grad():
            for eps in epsilons:
                df_eps = df[df['eps'] == eps]
                
                for i in tqdm(range(0, len(df_eps), batch_size), desc=f"Inferenza eps={eps:.3f}"):
                    batch_df = df_eps.iloc[i : i + batch_size]
                    
                    x_adv_batch, x_clean_batch = [], []
                    for _, row in batch_df.iterrows():
                        # Carichiamo sia la Clean che la Adv
                        c_rgb = cv2.cvtColor(cv2.resize(cv2.imread(row['source_image_path']), (160, 160)), cv2.COLOR_BGR2RGB)
                        a_rgb = cv2.cvtColor(cv2.imread(row['adversarial_image_path']), cv2.COLOR_BGR2RGB)
                        x_clean_batch.append(np.transpose(c_rgb, (2, 0, 1)).astype(np.float32) / 255.0)
                        x_adv_batch.append(np.transpose(a_rgb, (2, 0, 1)).astype(np.float32) / 255.0)
                        
                    x_clean_tensor = torch.tensor(np.array(x_clean_batch)).to(device)
                    x_adv_tensor = torch.tensor(np.array(x_adv_batch)).to(device)
                    
                    # Inferenza su entrambe
                    clean_preds = torch.argmax(resnet(x_clean_tensor * 2 - 1), dim=1).cpu().numpy()
                    
                    adv_logits = resnet(x_adv_tensor * 2 - 1)
                    adv_preds = torch.argmax(adv_logits, dim=1).cpu().numpy()
                    adv_probs = F.softmax(adv_logits, dim=1).cpu().numpy()
                    
                    targets = batch_df['target_class'].values
                    
                    for j in range(len(adv_preds)):
                        c_pred = clean_preds[j]
                        a_pred = adv_preds[j]
                        tgt_class = targets[j]
                        
                        original_idx = batch_df.index[j]
                        df.loc[original_idx, 'clean_pred_class'] = c_pred
                        df.loc[original_idx, 'adv_pred_class'] = a_pred
                        df.loc[original_idx, 'target_confidence'] = adv_probs[j, tgt_class]

        # MODIFICA 3: Rinominato il file di output per PGD
        evaluated_csv_path = output_eval_dir / f"pgd_targeted_evaluated_{strategy}.csv"
        df.to_csv(evaluated_csv_path, index=False)
        print(f"-> Master Data per {strategy} salvato in {evaluated_csv_path}")

        # =========================================================
        # BLOCCO 2: GENERAZIONE GRAFICI GLOBALI E DISTRIBUZIONE ESITI
        # =========================================================
        print(f"\n[BLOCCO 2 - {strategy.upper()}] Generazione Grafici Globali (t-ASR, Confidence & Outcome)...")
        asr_dict = {"PGD Targeted": []} # <--- AGGIORNATO
        confidence_data = []
        
        # Dizionario per i 3 stati
        outcome_data = {"Resisted": [], "Untargeted": [], "Targeted": []}

        for eps in epsilons:
            df_eps = df[df['eps'] == eps]
            total = len(df_eps)
            
            successes_df = df_eps[df_eps['adv_pred_class'] == df_eps['target_class']]
            resisted_df = df_eps[df_eps['adv_pred_class'] == df_eps['clean_pred_class']]
            
            successes = len(successes_df)
            resisted = len(resisted_df)
            untargeted = total - successes - resisted 
            
            asr_dict["PGD Targeted"].append(successes / total) # <--- AGGIORNATO
            confidence_data.append(df_eps['target_confidence'].values)
            
            outcome_data["Targeted"].append((successes / total) * 100)
            outcome_data["Resisted"].append((resisted / total) * 100)
            outcome_data["Untargeted"].append((untargeted / total) * 100)
            
            # --- ESPORTAZIONE CSV RESISTENTI ---
            if resisted > 0:
                eps_str_fmt = f"{eps:.3f}".replace('.', '_')
                resisted_csv_path = output_eval_dir / f"resisted_attacks_eps_{eps_str_fmt}.csv"
                
                # Salviamo solo le colonne utili per l'analisi investigativa
                resisted_export = resisted_df[[
                    'dataset_label', 'identity_name', 'target_class', 
                    'clean_pred_class', 'target_confidence', 
                    'source_image_path', 'adversarial_image_path'
                ]]
                resisted_export.to_csv(resisted_csv_path, index=False)

        # Chiamata ai grafici (Titolati PGD)
        plot_targeted_success_curve(epsilons, asr_dict, "NN1", True, str(output_eval_dir / "tasr_curve_global.png"))
        plot_target_confidence_growth(epsilons, confidence_data, f"PGD Targeted ({strategy})", True, str(output_eval_dir / "target_confidence_global.png"))
        plot_attack_outcome_distribution(epsilons, outcome_data, f"PGD Targeted ({strategy})", True, str(output_eval_dir / "outcome_distribution_stacked.png"))

        # =========================================================
        # BLOCCO 3: PROGRESSION SHOWCASE (Impatto visivo per Epsilon)
        # =========================================================
        print(f"\n[BLOCCO 3 - {strategy.upper()}] Generazione Progression Showcase...")
        # Scegliamo un ID immagine fisso (es. il primo del CSV) per vedere come cambia al variare di eps
        sample_source_path = df['source_image_path'].iloc[0]
        
        for eps in epsilons:
            sample = df[(df['eps'] == eps) & (df['source_image_path'] == sample_source_path)].iloc[0]
            
            c_bgr = cv2.resize(cv2.imread(sample['source_image_path']), (160, 160))
            a_bgr = cv2.imread(sample['adversarial_image_path'])
            
            c_rgb = cv2.cvtColor(c_bgr, cv2.COLOR_BGR2RGB)
            a_rgb = cv2.cvtColor(a_bgr, cv2.COLOR_BGR2RGB)

            eps_str_fmt = f"{eps:.3f}".replace('.', '_')
            
            plot_adversarial_showcase(
                c_rgb, a_rgb, 
                f"ID {int(sample['clean_pred_class'])}", 
                f"ID {int(sample['adv_pred_class'])}", 
                True, str(progression_dir / f"showcase_eps_{eps_str_fmt}.png")
            )
            
            plot_frequency_spectrum(c_rgb, a_rgb, True, str(progression_dir / f"spectrum_eps_{eps_str_fmt}.png"))

        # =========================================================
        # BLOCCO 4: MATRICI 
        # =========================================================
        PIVOT_EPS = 0.10
        df['eps_rounded'] = df['eps'].round(5) 
        
        print(f"\n[BLOCCO 4 - {strategy.upper()}] Generazione Matrici (Pivot: eps={PIVOT_EPS})...")
        
        if round(PIVOT_EPS, 5) in df['eps_rounded'].values:
            df_pivot = df[df['eps_rounded'] == round(PIVOT_EPS, 5)].copy()
            df_pivot['success'] = (df_pivot['adv_pred_class'] == df_pivot['target_class']).astype(int)
            
            # --- LOGICA 1: ROUND ROBIN (Matrice 10x10 esatta) ---
            if strategy.startswith("rr_"):
                # Le 10 identità che formano questo specifico subset
                rr_identities = sorted(df_pivot['identity_name'].unique())
                matrix = np.zeros((len(rr_identities), len(rr_identities)))
                
                for i, src_name in enumerate(rr_identities):
                    src_data = df_pivot[df_pivot['identity_name'] == src_name]
                    for j, tgt_name in enumerate(rr_identities):
                        if src_name == tgt_name: continue
                        
                        # Troviamo l'ID Facenet corrispondente al Target
                        tgt_class_series = df[df['identity_name'] == tgt_name]['clean_pred_class']
                        if not tgt_class_series.empty:
                            tgt_class = tgt_class_series.iloc[0]
                            attempts = src_data[src_data['target_class'] == tgt_class]
                            if not attempts.empty:
                                matrix[i, j] = attempts['success'].mean() * 100
                
                plot_source_target_heatmap(matrix, rr_identities, rr_identities, True, str(output_eval_dir / f"{strategy}_confusion_matrix.png"))
                
                # Per la XAI prenderemo il primo e l'ultimo di questa lista per fare i casi studio
                weakest_10 = rr_identities
                strongest_10 = rr_identities[::-1] # Ordine inverso

            # --- LOGICA 2: NEXT_BEST / LEAST_LIKELY / RANDOM (Matrici Data-Driven) ---
            else:
                source_asr = df_pivot.groupby('identity_name')['success'].mean() * 100
                weakest_10 = source_asr.sort_values(ascending=False).head(10).index.tolist()
                strongest_10 = source_asr.sort_values(ascending=True).head(10).index.tolist()

                def build_data_driven_source_target_matrix(df_pivot, top_k=15, filename="st_heatmap_datadriven.png"):
                    successful = df_pivot[df_pivot['success'] == 1]
                    if successful.empty: return

                    pair_counts = successful.groupby(['identity_name', 'target_class']).size().reset_index(name='count')
                    top_pairs = pair_counts.sort_values(by='count', ascending=False).head(top_k)
                    
                    top_srcs = top_pairs['identity_name'].unique().tolist()
                    top_tgts = top_pairs['target_class'].unique().tolist()
                    
                    matrix = np.zeros((len(top_srcs), len(top_tgts)))
                    for i, src in enumerate(top_srcs):
                        for j, tgt in enumerate(top_tgts):
                            attempts = df_pivot[(df_pivot['identity_name'] == src) & (df_pivot['target_class'] == tgt)]
                            if not attempts.empty:
                                matrix[i, j] = attempts['success'].mean() * 100
                                
                    target_labels = [f"Class {t}" for t in top_tgts]
                    plot_source_target_heatmap(matrix, top_srcs, target_labels, True, str(output_eval_dir / filename))

                build_data_driven_source_target_matrix(df_pivot, 15, f"st_heatmap_datadriven_{strategy}.png")

                def build_vuln_eps_matrix(identity_subset, filename, title):
                    subset_df = df[df['identity_name'].isin(identity_subset)].copy()
                    subset_df['success'] = (subset_df['adv_pred_class'] == subset_df['target_class']).astype(int)
                    pivot_table = subset_df.pivot_table(index='identity_name', columns='eps_rounded', values='success', aggfunc='mean') * 100
                    plot_vulnerability_vs_epsilon_heatmap(pivot_table.values, [f"{e:.3f}" for e in pivot_table.columns], pivot_table.index.tolist(), title, True, str(output_eval_dir / filename))

                build_vuln_eps_matrix(weakest_10, "vuln_vs_eps_weakest.png", "Vulnerability vs Epsilon (Weakest)")
                build_vuln_eps_matrix(strongest_10, "vuln_vs_eps_strongest.png", "Vulnerability vs Epsilon (Strongest)")

            # =========================================================
            # BLOCCO 5: EXPLAINABLE AI (XAI) SUI CASI STUDIO
            # =========================================================
            print(f"\n[BLOCCO 5 - {strategy.upper()}] Generazione Casi Studio XAI (Grad-CAM & UMAP)...")
            from util.plot.utils_plot_shared import plot_latent_trajectory
            cam = GradCAM(model=resnet, target_layers=[resnet.block8])

            def run_xai_pipeline(identity_name, case_folder_name):
                # Usiamo le immagini perturbate a eps=0.10
                df_010 = df[df['eps_rounded'] == round(0.10, 5)]
                sample_df = df_010[df_010['identity_name'] == identity_name]
                
                if sample_df.empty: return
                
                case_dir = explain_dir / case_folder_name
                case_dir.mkdir(exist_ok=True)
                
                # --- GRAD-CAM (Singolo Showcase) ---
                sample = sample_df.iloc[0]
                c_rgb = cv2.cvtColor(cv2.resize(cv2.imread(sample['source_image_path']), (160, 160)), cv2.COLOR_BGR2RGB)
                a_rgb = cv2.cvtColor(cv2.imread(sample['adversarial_image_path']), cv2.COLOR_BGR2RGB)
                
                c_chw = np.transpose(c_rgb, (2, 0, 1)).astype(np.float32) / 255.0
                a_chw = np.transpose(a_rgb, (2, 0, 1)).astype(np.float32) / 255.0
                
                t_clean = torch.tensor(np.expand_dims(c_chw, 0) * 2 - 1).to(device)
                t_adv = torch.tensor(np.expand_dims(a_chw, 0) * 2 - 1).to(device)
                
                clean_cam = cam(input_tensor=t_clean, targets=[ClassifierOutputTarget(sample['clean_pred_class'])])[0, :]
                adv_cam = cam(input_tensor=t_adv, targets=[ClassifierOutputTarget(sample['adv_pred_class'])])[0, :]
                
                plot_gradcam_shift(c_rgb, a_rgb, clean_cam, adv_cam, True, str(case_dir / "1_attention_shift.png"))
                
                # --- UMAP (Intero Cluster) ---
                resnet.classify = False
                
                # Sfondo: prendiamo 4 identità neutre
                df_unique_clean = df[df['eps'] == epsilons[0]]
                bg_identities = np.random.choice([i for i in df_unique_clean['identity_name'].unique() if i != identity_name], 4, replace=False)
                bg_df = df_unique_clean[df_unique_clean['identity_name'].isin(bg_identities)]
                
                bg_emb, bg_labels = [], []
                src_clean_emb, src_adv_emb = [], []
                
                with torch.no_grad():
                    # 1. Sfondo
                    for _, row in bg_df.iterrows():
                        img = np.transpose(cv2.cvtColor(cv2.resize(cv2.imread(row['source_image_path']), (160, 160)), cv2.COLOR_BGR2RGB), (2, 0, 1)).astype(np.float32) / 255.0
                        bg_emb.append(resnet(torch.tensor(np.expand_dims(img, 0) * 2 - 1).to(device)).cpu().numpy()[0])
                        bg_labels.append(row['identity_name'])
                        
                    # 2. Protagonisti (Clean & Adv)
                    for _, row in sample_df.iterrows():
                        c_img = np.transpose(cv2.cvtColor(cv2.resize(cv2.imread(row['source_image_path']), (160, 160)), cv2.COLOR_BGR2RGB), (2, 0, 1)).astype(np.float32) / 255.0
                        a_img = np.transpose(cv2.cvtColor(cv2.imread(row['adversarial_image_path']), cv2.COLOR_BGR2RGB), (2, 0, 1)).astype(np.float32) / 255.0
                        src_clean_emb.append(resnet(torch.tensor(np.expand_dims(c_img, 0) * 2 - 1).to(device)).cpu().numpy()[0])
                        src_adv_emb.append(resnet(torch.tensor(np.expand_dims(a_img, 0) * 2 - 1).to(device)).cpu().numpy()[0])
                
                resnet.classify = True
                
                # Vettori di flag e nomi
                adv_success_flags = (sample_df['adv_pred_class'] == sample_df['target_class']).values
                adv_target_names = []
                adv_actual_pred_names = []
                
                for _, row in sample_df.iterrows():
                    tgt_id = row['target_class']
                    pred_id = row['adv_pred_class']
                    
                    t_name_df = df_unique_clean[df_unique_clean['clean_pred_class'] == tgt_id]
                    p_name_df = df_unique_clean[df_unique_clean['clean_pred_class'] == pred_id]
                    
                    t_str = t_name_df['identity_name'].iloc[0] if not t_name_df.empty else f"Class {tgt_id}"
                    p_str = p_name_df['identity_name'].iloc[0] if not p_name_df.empty else f"Class {pred_id}"
                    
                    adv_target_names.append(t_str)
                    adv_actual_pred_names.append(p_str)
                
                # Logica per l'Area Rossa Target 
                tgt_clean_emb = None
                unique_targets = sample_df['target_class'].unique()
                if len(unique_targets) == 1:
                    tgt_id = unique_targets[0]
                    tgt_df = df_unique_clean[df_unique_clean['clean_pred_class'] == tgt_id]
                    if not tgt_df.empty:
                        tgt_clean_emb = []
                        resnet.classify = False
                        with torch.no_grad():
                            for _, row in tgt_df.iterrows():
                                img = np.transpose(cv2.cvtColor(cv2.resize(cv2.imread(row['source_image_path']), (160, 160)), cv2.COLOR_BGR2RGB), (2, 0, 1)).astype(np.float32) / 255.0
                                tgt_clean_emb.append(resnet(torch.tensor(np.expand_dims(img, 0) * 2 - 1).to(device)).cpu().numpy()[0])
                        resnet.classify = True
                        tgt_clean_emb = np.array(tgt_clean_emb)
                
                custom_title = f'"{identity_name}" attacked with \u03B5={PIVOT_EPS:.3f}'
                
                plot_latent_trajectory(
                    np.array(bg_emb), bg_labels,
                    np.array(src_clean_emb), np.array(src_adv_emb),
                    src_label_name=custom_title, 
                    adv_success_flags=adv_success_flags,
                    adv_target_names=adv_target_names,
                    adv_actual_pred_names=adv_actual_pred_names,
                    tgt_clean_emb=tgt_clean_emb,
                    save_flag=True, save_path=str(case_dir / "2_umap_trajectory.png")
                )

            # --- IL BIVIO DECISIONALE ---
            if strategy.startswith("rr_"):
                print(f" -> Generazione UMAP Corale 10x10 per {strategy}...")
                df_010 = df[df['eps_rounded'] == round(0.10, 5)]
                
                if not df_010.empty:
                    rr_identities = df_010['identity_name'].unique()
                    df_clean_rr = df[(df['eps'] == epsilons[0]) & (df['identity_name'].isin(rr_identities))]
                    
                    c_embs, c_lbls = [], []
                    a_embs, a_flags = [], []
                    a_src_lbls = [] 
                    a_tgt_lbls = [] 

                    resnet.classify = False
                    with torch.no_grad():
                        for _, row in df_clean_rr.iterrows():
                            img = np.transpose(cv2.cvtColor(cv2.resize(cv2.imread(str(base_dir / row['source_image_path'])), (160, 160)), cv2.COLOR_BGR2RGB), (2, 0, 1)).astype(np.float32) / 255.0
                            c_embs.append(resnet(torch.tensor(np.expand_dims(img, 0) * 2 - 1).to(device)).cpu().numpy()[0])
                            c_lbls.append(row['identity_name'])
                            
                        for _, row in df_010.iterrows():
                            a_rgb = cv2.cvtColor(cv2.imread(str(base_dir / row['adversarial_image_path'])), cv2.COLOR_BGR2RGB)
                            img = np.transpose(a_rgb, (2, 0, 1)).astype(np.float32) / 255.0
                            
                            a_embs.append(resnet(torch.tensor(np.expand_dims(img, 0) * 2 - 1).to(device)).cpu().numpy()[0])
                            a_flags.append(row['adv_pred_class'] == row['target_class'])
                            a_src_lbls.append(row['identity_name']) 
                            a_tgt_lbls.append(str(row['target_class'])) 
                            
                    resnet.classify = True
                    
                    explain_dir.mkdir(exist_ok=True)
                    plot_round_robin_plotly_grouped( 
                        np.array(c_embs), np.array(c_lbls), 
                        np.array(a_embs), np.array(a_flags), 
                        np.array(a_src_lbls), np.array(a_tgt_lbls),
                        str(explain_dir / f"round_robin_umap_{strategy}.html") 
                    )
                else:
                    print(" -> [SKIP] Dati insufficienti a eps 0.10")

            else:
                if len(weakest_10) > 0:
                    print(f" -> Elaborazione Caso 1 per {strategy}: Identità Debole")
                    run_xai_pipeline(weakest_10[0], "Case_1_Weakest")
                    
                if len(strongest_10) > 0:
                    print(f" -> Elaborazione Caso 2 per {strategy}: Identità Forte")
                    run_xai_pipeline(strongest_10[-1], "Case_2_Strongest")

        else:
            print(f"\n[WARNING] Pivot epsilon {PIVOT_EPS} non presente nei dati per {strategy}. Matrici e XAI saltate.")

    print("\n[OK] Pipeline di Evaluation conclusa con successo per tutte le strategie!")

if __name__ == "__main__":
    main()