import os
# --- FIX PER I CRASH SILENZIOSI (UMAP + PYTORCH SU WINDOWS) ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["NUMBA_NUM_THREADS"] = "1"
os.environ["NUMBA_DISABLE_JIT"] = "0"
os.environ["NUMBA_CACHE_DIR"] = os.path.join(os.path.expanduser("~"), ".numba_cache")
os.environ["TORCH_CUDNN_V8_API_ENABLED"] = "0"
os.environ["MKL_NUM_THREADS"] = "1"       
os.environ["OPENBLAS_NUM_THREADS"] = "1"  

import sys
import numpy as np
import pandas as pd
import torch

# Disabilitiamo CUDNN per evitare il mismatch di librerie (RISOLVE IL SEGFAULT)
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path
from PIL import Image, ImageOps, UnidentifiedImageError

# =========================================================================
# RISOLUZIONE ROBUSTA DEI PATH E FUNZIONI IMMAGINE (PIL)
# =========================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[5]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from facenet_pytorch import InceptionResnetV1
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# IMPORTIAMO IL GOOGLE LOGGER
from util.google_logger import GoogleSheetLogger

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

IMAGE_SIZE = 160

def resolve_project_path(base_dir: Path, path_value) -> Path:
    path = Path(str(path_value))
    if path.is_absolute():
        return path
    return base_dir / path

def load_rgb_image(path: Path, image_size: int = IMAGE_SIZE) -> np.ndarray:
    try:
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            if image.size != (image_size, image_size):
                image = image.resize((image_size, image_size), Image.Resampling.BILINEAR)
            return np.asarray(image, dtype=np.uint8)
    except (OSError, UnidentifiedImageError) as exc:
        raise FileNotFoundError(f"Immagine non leggibile: {path}") from exc

def rgb_to_chw_01(image_rgb: np.ndarray) -> np.ndarray:
    return np.transpose(image_rgb, (2, 0, 1)).astype(np.float32) / 255.0

def main():
    print("======================================================")
    print(" METRICHE & PLOT: BIM TARGETED (Error-Specific)       ")
    print("======================================================\n")

    base_dir = PROJECT_ROOT
    print(f"-> Project Root impostata a: {base_dir}")
    
    # PERCORSI IMPOSTATI PER BIM
    base_attacks_dir = base_dir / "dataset" / "attacks" / "error_specific" / "bim"
    base_plots_dir = base_dir / "plots" / "3_Adversarial_Examples" / "error_specific" / "bim"
    
    strategies = ["next_best", "least-likely", "random"]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"-> Inizializzazione NN1 globale su {device}...")
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    resnet.classify = True 

    # INIZIALIZZAZIONE DEL LOGGER
    logger = GoogleSheetLogger()

    for strategy in strategies:
        print(f"\n======================================================")
        print(f" AVVIO VALUTAZIONE STRATEGIA: {strategy.upper()}")
        print(f"======================================================")
        
        attacks_dir = base_attacks_dir / strategy
        output_eval_dir = base_plots_dir / strategy
        
        progression_dir = output_eval_dir / "visual_progression"
        explain_dir = output_eval_dir / "explainability"
        
        for d in [output_eval_dir, progression_dir, explain_dir]:
            d.mkdir(parents=True, exist_ok=True)

        tracker_files = list(attacks_dir.glob("eps_*/tracker_eps_*.csv"))
        
        if not tracker_files:
            print(f"[WARNING] Nessun file tracker trovato in {attacks_dir}. Salto strategia.")
            continue
            
        df_list = [pd.read_csv(f) for f in tracker_files]
        df = pd.concat(df_list, ignore_index=True)
        df['eps'] = pd.to_numeric(df['eps'], errors='raise').astype(float)
        df['target_class'] = pd.to_numeric(df['target_class'], errors='raise').astype(int)
        
        epsilons = sorted(df['eps'].unique())
        print(f"-> Epsilon rilevati: {epsilons}")
        
        if 'adv_pred_class' not in df.columns:
            df['clean_pred_class'] = -1 
            df['adv_pred_class'] = -1
            df['target_confidence'] = 0.0

        batch_size = 64 
        print(f"\n[BLOCCO 1 - {strategy.upper()}] Inferenza delle immagini avversarie...")

        with torch.no_grad():
            for eps in epsilons:
                df_eps = df[df['eps'] == eps]
                
                for i in tqdm(range(0, len(df_eps), batch_size), desc=f"Inferenza eps={eps:.3f}"):
                    batch_df = df_eps.iloc[i : i + batch_size]
                    
                    x_adv_batch, x_clean_batch = [], []
                    for _, row in batch_df.iterrows():
                        c_path = resolve_project_path(base_dir, row['source_image_path'])
                        a_path = resolve_project_path(base_dir, row['adversarial_image_path'])
                        
                        c_rgb = load_rgb_image(c_path)
                        a_rgb = load_rgb_image(a_path)
                        
                        x_clean_batch.append(rgb_to_chw_01(c_rgb))
                        x_adv_batch.append(rgb_to_chw_01(a_rgb))
                        
                    x_clean_tensor = torch.tensor(np.array(x_clean_batch)).to(device)
                    x_adv_tensor = torch.tensor(np.array(x_adv_batch)).to(device)
                    
                    clean_preds = torch.argmax(resnet(x_clean_tensor * 2 - 1), dim=1).cpu().numpy()
                    
                    adv_logits = resnet(x_adv_tensor * 2 - 1)
                    adv_preds = torch.argmax(adv_logits, dim=1).cpu().numpy()
                    adv_probs = F.softmax(adv_logits, dim=1).cpu().numpy()
                    
                    targets = batch_df['target_class'].values
                    
                    for j in range(len(adv_preds)):
                        c_pred = int(clean_preds[j])
                        a_pred = int(adv_preds[j])
                        tgt_class = int(targets[j])
                        
                        original_idx = batch_df.index[j]
                        df.loc[original_idx, 'clean_pred_class'] = c_pred
                        df.loc[original_idx, 'adv_pred_class'] = a_pred
                        df.loc[original_idx, 'target_confidence'] = adv_probs[j, tgt_class]

        evaluated_csv_path = output_eval_dir / f"bim_targeted_evaluated_{strategy}.csv"
        df.to_csv(evaluated_csv_path, index=False)
        print(f"-> Master Data per {strategy} salvato in {evaluated_csv_path}")

        print(f"\n[BLOCCO 2 - {strategy.upper()}] Generazione Grafici Globali (t-ASR, Confidence & Outcome)...")
        asr_dict = {"BIM Targeted": []}
        confidence_data = []
        outcome_data = {"Resisted": [], "Untargeted": [], "Targeted": []}

        for eps in epsilons:
            df_eps = df[df['eps'] == eps]
            total = len(df_eps)
            
            # --- FIX MATEMATICO: Stati Mutuamente Esclusivi ---
            # 1. Resisted
            resisted_mask = df_eps['adv_pred_class'] == df_eps['clean_pred_class']
            # 2. Targeted Success
            targeted_mask = (df_eps['adv_pred_class'] == df_eps['target_class']) & (~resisted_mask)
            # 3. Untargeted Success
            untargeted_mask = (~resisted_mask) & (~targeted_mask)
            
            resisted = resisted_mask.sum()
            successes = targeted_mask.sum()
            untargeted = untargeted_mask.sum()
            
            robust_accuracy = resisted / total
            targeted_asr = successes / total
            untargeted_asr = untargeted / total

            # Aggiorniamo i dizionari per i plot
            asr_dict["BIM Targeted"].append(targeted_asr)
            confidence_data.append(df_eps['target_confidence'].values)
            
            outcome_data["Targeted"].append(targeted_asr * 100)
            outcome_data["Resisted"].append(robust_accuracy * 100)
            outcome_data["Untargeted"].append(untargeted_asr * 100)
            
            # --- ESPORTAZIONE CSV RESISTENTI ---
            resisted_df = df_eps[resisted_mask]
            if resisted > 0:
                eps_str_fmt = f"{eps:.3f}".replace('.', '_')
                resisted_csv_path = output_eval_dir / f"resisted_attacks_eps_{eps_str_fmt}.csv"
                
                resisted_export = resisted_df[[
                    'dataset_label', 'identity_name', 'target_class', 
                    'clean_pred_class', 'target_confidence', 
                    'source_image_path', 'adversarial_image_path'
                ]]
                resisted_export.to_csv(resisted_csv_path, index=False)

            # --- LOGGING SU GOOGLE SHEETS ---
            # Fallback intelligente a seconda di come hai strutturato il logger
            if hasattr(logger, 'log_attack_metrics'):
                logger.log_attack_metrics(
                    tester="Andrea",
                    attack_type="BIM Error-Specific",
                    strategy=strategy,
                    epsilon=eps,
                    defense_type="None",
                    robust_accuracy=robust_accuracy,
                    targeted_asr=targeted_asr,
                    untargeted_asr=untargeted_asr,
                    notes="Valutazione Clean -> Adv"
                )
            else:
                logger.log_biometric_metrics(
                    tester="Andrea", 
                    phase=f"Targeted Attack ({strategy})",
                    attack_type="BIM Error-Specific",
                    epsilon=eps,
                    defense_type="None",
                    accuracy=targeted_asr, # Salviamo la Targeted ASR nella colonna principale
                    eer=0.0,
                    far=0.0,
                    frr=0.0,
                    threshold=0.0,
                    notes=f"Untargeted: {untargeted_asr:.1%}, Resisted: {robust_accuracy:.1%}"
                )

        plot_targeted_success_curve(epsilons, asr_dict, "NN1", True, str(output_eval_dir / "tasr_curve_global.png"))
        plot_target_confidence_growth(epsilons, confidence_data, f"BIM Targeted ({strategy})", True, str(output_eval_dir / "target_confidence_global.png"))
        plot_attack_outcome_distribution(epsilons, outcome_data, f"BIM Targeted ({strategy})", True, str(output_eval_dir / "outcome_distribution_stacked.png"))

        print(f"\n[BLOCCO 3 - {strategy.upper()}] Generazione Progression Showcase...")
        sample_source_path = df['source_image_path'].iloc[0]
        
        for eps in epsilons:
            sample_candidates = df[(df['eps'] == eps) & (df['source_image_path'] == sample_source_path)]
            if sample_candidates.empty: continue
            sample = sample_candidates.iloc[0]
            
            c_rgb = load_rgb_image(resolve_project_path(base_dir, sample['source_image_path']))
            a_rgb = load_rgb_image(resolve_project_path(base_dir, sample['adversarial_image_path']))

            eps_str_fmt = f"{eps:.3f}".replace('.', '_')
            plot_adversarial_showcase(
                c_rgb, a_rgb, 
                f"ID {int(sample['clean_pred_class'])}", 
                f"ID {int(sample['adv_pred_class'])}", 
                True, str(progression_dir / f"showcase_eps_{eps_str_fmt}.png")
            )
            plot_frequency_spectrum(c_rgb, a_rgb, True, str(progression_dir / f"spectrum_eps_{eps_str_fmt}.png"))

        PIVOT_EPS = 0.10
        df['eps_rounded'] = df['eps'].round(5) 
        
        print(f"\n[BLOCCO 4 - {strategy.upper()}] Generazione Matrici (Pivot: eps={PIVOT_EPS})...")
        
        if round(PIVOT_EPS, 5) in df['eps_rounded'].values:
            df_pivot = df[df['eps_rounded'] == round(PIVOT_EPS, 5)].copy()
            df_pivot['success'] = (df_pivot['adv_pred_class'] == df_pivot['target_class']).astype(int)
            
            if strategy.startswith("rr_"):
                rr_identities = sorted(df_pivot['identity_name'].unique())
                matrix = np.zeros((len(rr_identities), len(rr_identities)))
                
                for i, src_name in enumerate(rr_identities):
                    src_data = df_pivot[df_pivot['identity_name'] == src_name]
                    for j, tgt_name in enumerate(rr_identities):
                        if src_name == tgt_name: continue
                        
                        tgt_class_series = df[df['identity_name'] == tgt_name]['clean_pred_class']
                        if not tgt_class_series.empty:
                            tgt_class = tgt_class_series.iloc[0]
                            attempts = src_data[src_data['target_class'] == tgt_class]
                            if not attempts.empty:
                                matrix[i, j] = attempts['success'].mean() * 100
                
                plot_source_target_heatmap(matrix, rr_identities, rr_identities, True, str(output_eval_dir / f"{strategy}_confusion_matrix.png"))
                weakest_10 = rr_identities
                strongest_10 = rr_identities[::-1] 
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

                if len(weakest_10) > 0:
                    build_vuln_eps_matrix(weakest_10, "vuln_vs_eps_weakest.png", "Vulnerability vs Epsilon (Weakest)")
                if len(strongest_10) > 0:
                    build_vuln_eps_matrix(strongest_10, "vuln_vs_eps_strongest.png", "Vulnerability vs Epsilon (Strongest)")

            # =====================================================================
            # BLOCCO 5: EXPLAINABLE AI (XAI) SUI CASI STUDIO
            # =====================================================================
            print(f"\n[BLOCCO 5 - {strategy.upper()}] Generazione Casi Studio XAI (Grad-CAM & UMAP)...")
            cam = GradCAM(model=resnet, target_layers=[resnet.block8])

            def run_xai_pipeline(identity_name, case_folder_name):
                df_010 = df[df['eps_rounded'] == round(0.10, 5)]
                sample_df = df_010[df_010['identity_name'] == identity_name]
                
                if sample_df.empty: return
                
                case_dir = explain_dir / case_folder_name
                case_dir.mkdir(exist_ok=True)
                
                sample = sample_df.iloc[0]
                
                c_rgb = load_rgb_image(resolve_project_path(base_dir, sample['source_image_path']))
                a_rgb = load_rgb_image(resolve_project_path(base_dir, sample['adversarial_image_path']))
                
                c_chw = rgb_to_chw_01(c_rgb)
                a_chw = rgb_to_chw_01(a_rgb)
                
                t_clean = torch.tensor(np.expand_dims(c_chw, 0) * 2 - 1).to(device)
                t_adv = torch.tensor(np.expand_dims(a_chw, 0) * 2 - 1).to(device)
                
                clean_cam = cam(input_tensor=t_clean, targets=[ClassifierOutputTarget(int(sample['clean_pred_class']))])[0, :]
                adv_cam = cam(input_tensor=t_adv, targets=[ClassifierOutputTarget(int(sample['adv_pred_class']))])[0, :]
                
                plot_gradcam_shift(c_rgb, a_rgb, clean_cam, adv_cam, True, str(case_dir / "1_attention_shift.png"))
                
                resnet.classify = False
                
                df_unique_clean = df[df['eps'] == epsilons[0]]
                bg_identities = np.random.choice([i for i in df_unique_clean['identity_name'].unique() if i != identity_name], 4, replace=False)
                bg_df = df_unique_clean[df_unique_clean['identity_name'].isin(bg_identities)]
                
                bg_emb, bg_labels = [], []
                src_clean_emb, src_adv_emb = [], []
                
                with torch.no_grad():
                    for _, row in bg_df.iterrows():
                        img_path = resolve_project_path(base_dir, row['source_image_path'])
                        img = rgb_to_chw_01(load_rgb_image(img_path))
                        bg_emb.append(resnet(torch.tensor(np.expand_dims(img, 0) * 2 - 1).to(device)).cpu().numpy()[0])
                        bg_labels.append(row['identity_name'])
                        
                    for _, row in sample_df.iterrows():
                        c_img_proc = rgb_to_chw_01(load_rgb_image(resolve_project_path(base_dir, row['source_image_path'])))
                        a_img_proc = rgb_to_chw_01(load_rgb_image(resolve_project_path(base_dir, row['adversarial_image_path'])))
                        src_clean_emb.append(resnet(torch.tensor(np.expand_dims(c_img_proc, 0) * 2 - 1).to(device)).cpu().numpy()[0])
                        src_adv_emb.append(resnet(torch.tensor(np.expand_dims(a_img_proc, 0) * 2 - 1).to(device)).cpu().numpy()[0])
                
                resnet.classify = True
                
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
                                img_proc = rgb_to_chw_01(load_rgb_image(resolve_project_path(base_dir, row['source_image_path'])))
                                tgt_clean_emb.append(resnet(torch.tensor(np.expand_dims(img_proc, 0) * 2 - 1).to(device)).cpu().numpy()[0])
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
                            img_proc = rgb_to_chw_01(load_rgb_image(resolve_project_path(base_dir, row['source_image_path'])))
                            c_embs.append(resnet(torch.tensor(np.expand_dims(img_proc, 0) * 2 - 1).to(device)).cpu().numpy()[0])
                            c_lbls.append(row['identity_name'])
                            
                        for _, row in df_010.iterrows():
                            img_proc = rgb_to_chw_01(load_rgb_image(resolve_project_path(base_dir, row['adversarial_image_path'])))
                            a_embs.append(resnet(torch.tensor(np.expand_dims(img_proc, 0) * 2 - 1).to(device)).cpu().numpy()[0])
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