import os
os.environ["TORCH_CUDNN_V8_API_ENABLED"] = "0" 
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path

# Disabilitiamo CUDNN per evitare il mismatch di librerie
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

from facenet_pytorch import InceptionResnetV1
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from util.google_logger import GoogleSheetLogger

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

# ==========================================
# WRAPPER (Per Grad-CAM in 64-bit)
# ==========================================
class ARTFloat64Wrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x):
        return self.model(x.to(torch.float64))

def main():
    print("======================================================")
    print(" METRICHE & PLOT: C&W TARGETED (Error-Specific)       ")
    print("======================================================\n")

    base_dir = Path.cwd()
    base_attacks_dir = base_dir / "dataset" / "attacks" / "NN1" / "error_specific" / "cw"
    base_plots_dir = base_dir / "plots" / "3_Adversarial_Examples" / "error_specific" / "cw"
    
    strategies = ["next_best"]#, "least-likely", "random"]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"-> Inizializzazione NN1 globale su {device}...")
    
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    resnet.classify = True 
    resnet.double() # Mantieni 64-bit per XAI ad alta precisione
    
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

        tracker_path = attacks_dir / f"tracker_{strategy}.csv"
        
        if not tracker_path.exists():
            print(f"[WARNING] File tracker {tracker_path} non trovato. Salto strategia.")
            continue

        print(f"-> Lettura tracker per {strategy}...")
        df = pd.read_csv(tracker_path)
        total_images = len(df)
        
        # Le predizioni sono già state calcolate nel generatore! Nessun Blocco 1 necessario.

        # =========================================================
        # BLOCCO 2: GENERAZIONE CURVE DI VALUTAZIONE (Soglie Multiple)
        # =========================================================
        print(f"\n[BLOCCO 2 - {strategy.upper()}] Generazione Curve (Epsilon Dinamici)...")
        
        # Calcolo dinamico degli Epsilon basato sulla distribuzione dei successi mirati
        successful_attacks = df[df['adv_pred_class'] == df['target_class']]
        
        if not successful_attacks.empty:
            percentiles = np.linspace(0, 100, 6)
            epsilons_raw = np.percentile(successful_attacks['linf'], percentiles).tolist()
        else:
            print("Attenzione: Nessun attacco mirato ha avuto successo. Uso default.")
            epsilons_raw = [0.0, 0.025, 0.05, 0.075, 0.1, 0.15, 0.20]
            
        epsilons = [0.0] + [round(e, 8) for e in epsilons_raw] + [round(epsilons_raw[-1] + 0.001, 8)]
        epsilons = sorted(list(set(epsilons))) 
        print(f"-> Epsilon calcolati dinamicamente: {epsilons}")
        
        asr_dict = {"C&W Targeted": []}
        confidence_data = []
        outcome_data = {"Resisted": [], "Untargeted": [], "Targeted": []}

        for eps in epsilons:
            within_budget = df['linf'] <= eps
            
            # Stati mutualmente esclusivi
            targeted_mask = within_budget & (df['adv_pred_class'] == df['target_class'])
            resisted_attack = within_budget & (df['adv_pred_class'] == df['clean_pred_class'])
            untargeted_mask = within_budget & (~targeted_mask) & (~resisted_attack)
            
            # Le immagini out-of-budget contano come resistite
            resisted_budget = df['linf'] > eps
            
            successes = targeted_mask.sum()
            untargeted = untargeted_mask.sum()
            resisted = resisted_budget.sum() + resisted_attack.sum()
            
            robust_accuracy = resisted / total_images
            targeted_asr = successes / total_images
            untargeted_asr = untargeted / total_images

            asr_dict["C&W Targeted"].append(targeted_asr) 
            
            outcome_data["Targeted"].append(targeted_asr * 100)
            outcome_data["Resisted"].append(robust_accuracy * 100)
            outcome_data["Untargeted"].append(untargeted_asr * 100)
            
            confs = np.where(resisted_budget, df['clean_target_confidence'], df['adv_target_confidence'])
            confidence_data.append(confs)

            if hasattr(logger, 'log_attack_metrics'):
                logger.log_attack_metrics(
                    tester="Leonardo", 
                    attack_type="C&W Error-Specific",
                    strategy=strategy,
                    epsilon=eps,
                    defense_type="None",
                    robust_accuracy=robust_accuracy,
                    targeted_asr=targeted_asr,
                    untargeted_asr=untargeted_asr,
                    notes="Valutazione Retroattiva C&W (TIFF 32-bit, Eps Dinamico)"
                )

        plot_targeted_success_curve(epsilons, asr_dict, "NN1", True, str(output_eval_dir / "tasr_curve_global.png"))
        plot_target_confidence_growth(epsilons, confidence_data, f"C&W Targeted ({strategy})", True, str(output_eval_dir / "target_confidence_global.png"))
        plot_attack_outcome_distribution(epsilons, outcome_data, f"C&W Targeted ({strategy})", True, str(output_eval_dir / "outcome_distribution_stacked.png"))

        # =========================================================
        # BLOCCO 3: VISUAL SHOWCASE
        # =========================================================
        print(f"\n[BLOCCO 3 - {strategy.upper()}] Generazione Progression Showcase...")
        
        for i, eps in enumerate(epsilons):
            if eps == 0.0: continue 
            
            lower_bound = epsilons[i-1] 
            suitable = df[(df['adv_pred_class'] == df['target_class']) & (df['linf'] <= eps) & (df['linf'] > lower_bound)]
            
            if not suitable.empty:
                sample = suitable.sort_values(by='linf', ascending=False).iloc[0]
                
                # Entrambi sono TIFF a 32-bit float [0, 1]
                c_bgr_float32 = cv2.imread(str(base_dir / sample['source_image_path']), cv2.IMREAD_UNCHANGED)
                a_bgr_float32 = cv2.imread(str(base_dir / sample['adversarial_image_path']), cv2.IMREAD_UNCHANGED)
                
                if c_bgr_float32 is None or a_bgr_float32 is None: continue
                
                c_rgb = cv2.cvtColor(c_bgr_float32, cv2.COLOR_BGR2RGB)
                a_rgb = cv2.cvtColor(a_bgr_float32, cv2.COLOR_BGR2RGB)

                eps_str_fmt = f"{eps:.4f}".replace('.', '_')
                
                plot_adversarial_showcase(
                    c_rgb, a_rgb, 
                    f"Orig: ID {int(sample['clean_pred_class'])}", 
                    f"Target: ID {int(sample['adv_pred_class'])} (HIT)", 
                    True, str(progression_dir / f"showcase_eps_limit_{eps_str_fmt}.png")
                )
                plot_frequency_spectrum(c_rgb, a_rgb, True, str(progression_dir / f"spectrum_eps_limit_{eps_str_fmt}.png"))
            else:
                print(f" -> [SKIP] Nessun campione rappresentativo trovato tra {lower_bound:.4f} e {eps:.4f}")

        # =========================================================
        # BLOCCO 4: MATRICI (Pivot: EPS=0.10)
        # =========================================================
        PIVOT_EPS = 0.10
        print(f"\n[BLOCCO 4 - {strategy.upper()}] Generazione Matrici (Pivot max: eps={PIVOT_EPS})...")
        
        df_pivot = df.copy()
        df_pivot['success'] = ((df_pivot['adv_pred_class'] == df_pivot['target_class']) & (df_pivot['linf'] <= PIVOT_EPS)).astype(int)
        
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
            for src in top_srcs:
                for tgt in top_tgts:
                    attempts = df_pivot[(df_pivot['identity_name'] == src) & (df_pivot['target_class'] == tgt)]
                    if not attempts.empty:
                        i = top_srcs.index(src)
                        j = top_tgts.index(tgt)
                        matrix[i, j] = attempts['success'].mean() * 100
                        
            target_labels = [f"Class {t}" for t in top_tgts]
            plot_source_target_heatmap(matrix, top_srcs, target_labels, True, str(output_eval_dir / filename))

        build_data_driven_source_target_matrix(df_pivot, 15, f"st_heatmap_datadriven_{strategy}.png")

        # =========================================================
        # BLOCCO 5: EXPLAINABLE AI (XAI) SUI CASI STUDIO
        # =========================================================
        print(f"\n[BLOCCO 5 - {strategy.upper()}] Generazione Casi Studio XAI (Grad-CAM & UMAP)...")
        cam = GradCAM(model=resnet, target_layers=[resnet.block8])

        def run_xai_pipeline(identity_name, case_folder_name):
            sample_df = df_pivot[df_pivot['identity_name'] == identity_name]
            if sample_df.empty: return
            
            case_dir = explain_dir / case_folder_name
            case_dir.mkdir(exist_ok=True)
            
            # --- GRAD-CAM ---
            # Prendiamo il campione con L_inf <= 0.10 più alto (il più vicino al confine)
            valid_samples = sample_df[sample_df['success'] == 1]
            if valid_samples.empty: valid_samples = sample_df
            sample = valid_samples.sort_values(by='linf', ascending=False).iloc[0]
            
            c_bgr = cv2.imread(str(base_dir / sample['source_image_path']), cv2.IMREAD_UNCHANGED)
            a_bgr = cv2.imread(str(base_dir / sample['adversarial_image_path']), cv2.IMREAD_UNCHANGED)
            
            c_rgb = cv2.cvtColor(c_bgr, cv2.COLOR_BGR2RGB)
            a_rgb = cv2.cvtColor(a_bgr, cv2.COLOR_BGR2RGB)
            
            c_chw = np.transpose(c_rgb, (2, 0, 1))
            a_chw = np.transpose(a_rgb, (2, 0, 1))
            
            t_clean = torch.tensor(np.expand_dims(c_chw, 0) * 2 - 1).to(device).double()
            t_adv = torch.tensor(np.expand_dims(a_chw, 0) * 2 - 1).to(device).double()
            
            clean_cam = cam(input_tensor=t_clean, targets=[ClassifierOutputTarget(sample['clean_pred_class'])])[0, :]
            adv_cam = cam(input_tensor=t_adv, targets=[ClassifierOutputTarget(sample['adv_pred_class'])])[0, :]
            
            plot_gradcam_shift(c_rgb, a_rgb, clean_cam, adv_cam, True, str(case_dir / "1_attention_shift.png"))
            
            # --- UMAP ---
            resnet.classify = False
            
            bg_identities = np.random.choice([i for i in df['identity_name'].unique() if i != identity_name], 4, replace=False)
            bg_df = df[df['identity_name'].isin(bg_identities)].drop_duplicates('source_image_path')
            
            bg_emb, bg_labels = [], []
            src_clean_emb, src_adv_emb = [], []
            
            with torch.no_grad():
                for _, row in bg_df.iterrows():
                    img = np.transpose(cv2.cvtColor(cv2.imread(str(base_dir / row['source_image_path']), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB), (2, 0, 1))
                    bg_emb.append(resnet(torch.tensor(np.expand_dims(img, 0) * 2 - 1).to(device).double()).cpu().numpy()[0])
                    bg_labels.append(row['identity_name'])
                    
                for _, row in sample_df.iterrows():
                    c_img = np.transpose(cv2.cvtColor(cv2.imread(str(base_dir / row['source_image_path']), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB), (2, 0, 1))
                    a_img = np.transpose(cv2.cvtColor(cv2.imread(str(base_dir / row['adversarial_image_path']), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB), (2, 0, 1))
                    src_clean_emb.append(resnet(torch.tensor(np.expand_dims(c_img, 0) * 2 - 1).to(device).double()).cpu().numpy()[0])
                    src_adv_emb.append(resnet(torch.tensor(np.expand_dims(a_img, 0) * 2 - 1).to(device).double()).cpu().numpy()[0])
            
            resnet.classify = True
            
            adv_success_flags = sample_df['success'].values
            adv_target_names, adv_actual_pred_names = [], []
            
            for _, row in sample_df.iterrows():
                tgt_id = row['target_class']
                pred_id = row['adv_pred_class']
                
                t_name_df = df[df['clean_pred_class'] == tgt_id]
                p_name_df = df[df['clean_pred_class'] == pred_id]
                
                t_str = t_name_df['identity_name'].iloc[0] if not t_name_df.empty else f"Class {tgt_id}"
                p_str = p_name_df['identity_name'].iloc[0] if not p_name_df.empty else f"Class {pred_id}"
                
                adv_target_names.append(t_str)
                adv_actual_pred_names.append(p_str)
            
            tgt_clean_emb = None
            unique_targets = sample_df['target_class'].unique()
            if len(unique_targets) == 1:
                tgt_id = unique_targets[0]
                tgt_df = df[df['clean_pred_class'] == tgt_id].drop_duplicates('source_image_path')
                if not tgt_df.empty:
                    tgt_clean_emb = []
                    resnet.classify = False
                    with torch.no_grad():
                        for _, row in tgt_df.iterrows():
                            img = np.transpose(cv2.cvtColor(cv2.imread(str(base_dir / row['source_image_path']), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB), (2, 0, 1))
                            tgt_clean_emb.append(resnet(torch.tensor(np.expand_dims(img, 0) * 2 - 1).to(device).double()).cpu().numpy()[0])
                    resnet.classify = True
                    tgt_clean_emb = np.array(tgt_clean_emb)
            
            custom_title = f'"{identity_name}" targeted attack'
            
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

        if len(weakest_10) > 0:
            print(f" -> Elaborazione Caso 1 per {strategy}: Identità Debole")
            run_xai_pipeline(weakest_10[0], "Case_1_Weakest")
            
        if len(strongest_10) > 0:
            print(f" -> Elaborazione Caso 2 per {strategy}: Identità Forte")
            run_xai_pipeline(strongest_10[-1], "Case_2_Strongest")

    print("\n[OK] Pipeline di Evaluation conclusa con successo per tutte le strategie C&W!")

if __name__ == "__main__":
    main()