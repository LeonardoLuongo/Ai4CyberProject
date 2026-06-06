import os
import time

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["NUMBA_NUM_THREADS"] = "1"
os.environ["TORCH_CUDNN_V8_API_ENABLED"] = "0"

import sys
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path

torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

PROJECT_ROOT = Path(__file__).resolve().parents[5]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from facenet_pytorch import InceptionResnetV1
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

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

# FUNZIONE DI LETTURA TIFF 
def load_rgb_image(path: Path, image_size: int = IMAGE_SIZE) -> np.ndarray:
    image_bgr_float32 = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image_bgr_float32 is None:
        raise FileNotFoundError(f"TIFF non leggibile: {path}")
    if image_bgr_float32.ndim != 3 or image_bgr_float32.shape[2] != 3:
        raise ValueError(f"TIFF RGB non valido: {path}, shape={image_bgr_float32.shape}")
        
    if image_bgr_float32.shape[:2] != (image_size, image_size):
        image_bgr_float32 = cv2.resize(image_bgr_float32, (image_size, image_size), interpolation=cv2.INTER_LINEAR)

    image_rgb_float32 = cv2.cvtColor(image_bgr_float32, cv2.COLOR_BGR2RGB)
    return image_rgb_float32.astype(np.float32)

def rgb_to_chw_01(image_rgb: np.ndarray) -> np.ndarray:
    return np.transpose(image_rgb, (2, 0, 1)).astype(np.float32)

def main():
    print("======================================================")
    print(" METRICHE & PLOT: PGD TARGETED (Error-Specific)       ")
    print("======================================================\n")

    base_dir = PROJECT_ROOT
    print(f"-> Project Root impostata a: {base_dir}")

    base_attacks_dir = base_dir / "dataset" / "attacks" / "NN1" / "error_specific" / "pgd"
    base_plots_dir = base_dir / "plots" / "3_Adversarial_Examples" / "error_specific" / "pgd"
    
    strategies = ["next_best", "least-likely", "random"]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"-> Inizializzazione NN1 globale su {device}...")
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    resnet.classify = True 

    logger = GoogleSheetLogger()

    for strategy in strategies:
        print(f"\n======================================================")
        print(f" AVVIO VALUTAZIONE STRATEGIA: {strategy.upper()}")
        print(f"======================================================")
        
        attacks_dir = base_attacks_dir / strategy
        output_eval_dir = base_plots_dir / strategy
        
        progression_dir = output_eval_dir / "visual_progression"
        explain_dir = output_eval_dir / "explainability"
        
        # FIX: Creazione cartelle iper-robusta
        for d in [output_eval_dir, progression_dir, explain_dir]:
            try:
                os.makedirs(str(d), exist_ok=True)
            except FileExistsError:
                print(f"[ERRORE CRITICO] Il path '{d}' esiste ma è un FILE, non una cartella! CANCELLALO manualmente da Windows e riavvia.")
                sys.exit(1)

        tracker_files = list(attacks_dir.glob("eps_*/tracker_eps_*.csv"))
        if not tracker_files:
            print(f"[WARNING] Nessun file tracker trovato in {attacks_dir}. Salto strategia.")
            continue

        print(f"-> Trovati {len(tracker_files)} file tracker. Unione in corso...")
        df_list = [pd.read_csv(f) for f in tracker_files]
        df = pd.concat(df_list, ignore_index=True)
        
        df['eps'] = pd.to_numeric(df['eps'], errors='raise').astype(float)
        
        # Salviamo gli epsilon nominali originali per l'inferenza e le matrici
        nominal_epsilons = sorted(df['eps'].unique())
        
        if 'adv_pred_class' not in df.columns:
            df['clean_pred_class'] = -1 
            df['adv_pred_class'] = -1
            df['target_confidence'] = 0.0

        # =========================================================
        # BLOCCO 1: INFERENZA IN BATCH (TIFF 32-bit)
        # =========================================================
        batch_size = 64 
        print(f"\n[BLOCCO 1 - {strategy.upper()}] Inferenza delle immagini avversarie e originali...")

        with torch.no_grad():
            for eps in nominal_epsilons:
                df_eps = df[df['eps'] == eps]
                
                for i in tqdm(range(0, len(df_eps), batch_size), desc=f"Inferenza eps={eps:.3f}"):
                    batch_df = df_eps.iloc[i : i + batch_size]
                    
                    x_adv_batch, x_clean_batch = [], []
                    for _, row in batch_df.iterrows():
                        c_path = resolve_project_path(base_dir, row['source_image_path'])
                        a_path = resolve_project_path(base_dir, row['adversarial_image_path'])
                        
                        x_clean_batch.append(rgb_to_chw_01(load_rgb_image(c_path)))
                        x_adv_batch.append(rgb_to_chw_01(load_rgb_image(a_path)))
                        
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

        evaluated_csv_path = output_eval_dir / f"pgd_targeted_evaluated_{strategy}.csv"
        
        # FIX: Doppio check della directory prima di salvare e cast in Stringa
        os.makedirs(str(output_eval_dir), exist_ok=True)
        time.sleep(0.1) # Breve attesa per permettere a Windows Defender di rilasciare il lock
        df.to_csv(str(evaluated_csv_path), index=False)

        # =========================================================
        # NUOVO CALCOLO DINAMICO DEGLI EPSILON (BASATO SUI PERCENTILI)
        # =========================================================
        print(f"\n[CALCOLO DINAMICO EPSILON TARGETED - {strategy.upper()}]")
        successful_attacks = df[(df['adv_pred_class'] == df['target_class']) & (df['clean_pred_class'] != df['target_class'])]
        
        noise_col = 'linf' if 'linf' in df.columns else 'eps'
        
        if not successful_attacks.empty:
            percentiles = np.linspace(0, 100, 6)
            epsilons_raw = np.percentile(successful_attacks[noise_col], percentiles).tolist()
        else:
            print("Attenzione: Nessun attacco ha raggiunto la classe target. Uso default.")
            epsilons_raw = [0.01, 0.02, 0.03, 0.04, 0.05, 0.10]
            
        epsilons =  [round(e, 8) for e in epsilons_raw] + [round(epsilons_raw[-1] + 0.001, 8)]
        epsilons = sorted(list(set(epsilons))) 
        print(f"-> Epsilon calcolati dinamicamente (Percentili di {noise_col}): {epsilons}")

        # =========================================================
        # BLOCCO 2: GENERAZIONE GRAFICI GLOBALI (LOGICA CUMULATIVA)
        # =========================================================
        print(f"\n[BLOCCO 2 - {strategy.upper()}] Generazione Grafici Globali...")
        asr_dict = {"PGD Targeted": []}
        confidence_data = []
        outcome_data = {"Resisted": [], "Untargeted": [], "Targeted": []}
        
        total_unique_images = df['source_image_path'].nunique()

        for eps in epsilons:
            valid_attacks = df[df[noise_col] <= eps]
            
            if valid_attacks.empty:
                robust_accuracy = 1.0
                targeted_asr = 0.0
                untargeted_asr = 0.0
                confidence_data.append(np.array([]))
            else:
                closest_attacks = valid_attacks.sort_values(noise_col).drop_duplicates('source_image_path', keep='last')
                missing_count = total_unique_images - len(closest_attacks) 
                
                resisted_mask = closest_attacks['adv_pred_class'] == closest_attacks['clean_pred_class']
                targeted_mask = (closest_attacks['adv_pred_class'] == closest_attacks['target_class']) & (~resisted_mask)
                untargeted_mask = (~resisted_mask) & (~targeted_mask)
                
                resisted = resisted_mask.sum() + missing_count
                successes = targeted_mask.sum()
                untargeted = untargeted_mask.sum()
                
                robust_accuracy = resisted / total_unique_images
                targeted_asr = successes / total_unique_images
                untargeted_asr = untargeted / total_unique_images
                
                confidence_data.append(closest_attacks['target_confidence'].values)
            
            asr_dict["PGD Targeted"].append(targeted_asr)
            outcome_data["Targeted"].append(targeted_asr * 100)
            outcome_data["Resisted"].append(robust_accuracy * 100)
            outcome_data["Untargeted"].append(untargeted_asr * 100)

            try:
                logger.log_attack_metrics(
                    tester="Francesco", 
                    attack_type="PGD Error-Specific",
                    strategy=strategy,
                    epsilon=eps,
                    defense_type="None",
                    robust_accuracy=robust_accuracy,
                    targeted_asr=targeted_asr,
                    untargeted_asr=untargeted_asr,
                    notes="Valutazione Targeted TIFF 32-bit (Dynamic)"
                )
            except Exception as e:
                pass

        plot_targeted_success_curve(epsilons, asr_dict, "NN1", True, str(output_eval_dir / "tasr_curve_global.png"))
        plot_target_confidence_growth(epsilons, confidence_data, f"PGD Targeted ({strategy})", True, str(output_eval_dir / "target_confidence_global.png"))
        plot_attack_outcome_distribution(epsilons, outcome_data, f"PGD Targeted ({strategy})", True, str(output_eval_dir / "outcome_distribution_stacked.png"))

        # =========================================================
        # BLOCCO 3: VISUAL SHOWCASE
        # =========================================================
        print(f"\n[BLOCCO 3 - {strategy.upper()}] Generazione Visual Showcase...")
        sample_source_path = df['source_image_path'].iloc[0]
        
        for eps in epsilons:
            sample_candidates = df[(df[noise_col] <= eps) & (df['source_image_path'] == sample_source_path)]
            if sample_candidates.empty: continue
            
            sample = sample_candidates.sort_values(noise_col).iloc[-1]
            
            c_rgb = load_rgb_image(resolve_project_path(base_dir, sample['source_image_path']))
            a_rgb = load_rgb_image(resolve_project_path(base_dir, sample['adversarial_image_path']))

            eps_str_fmt = f"{eps:.5f}".replace('.', '_')
            
            plot_adversarial_showcase(
                c_rgb, a_rgb, 
                f"ID {int(sample['clean_pred_class'])} -> Tgt: {int(sample['target_class'])}", 
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

            # =========================================================
            # BLOCCO 5: EXPLAINABLE AI (XAI)
            # =========================================================
            print(f"\n[BLOCCO 5 - {strategy.upper()}] Generazione Casi Studio XAI (Grad-CAM & UMAP)...")
            cam = GradCAM(model=resnet, target_layers=[resnet.block8])

            def run_xai_pipeline(identity_name, case_folder_name):
                df_010 = df[df['eps_rounded'] == round(0.10, 5)]
                sample_df = df_010[df_010['identity_name'] == identity_name]
                if sample_df.empty: return
                
                case_dir = explain_dir / case_folder_name
                os.makedirs(str(case_dir), exist_ok=True)
                
                sample = sample_df.iloc[0]
                c_rgb = load_rgb_image(resolve_project_path(base_dir, sample['source_image_path']))
                a_rgb = load_rgb_image(resolve_project_path(base_dir, sample['adversarial_image_path']))
                
                t_clean = torch.tensor(np.expand_dims(rgb_to_chw_01(c_rgb), 0) * 2 - 1).to(device)
                t_adv = torch.tensor(np.expand_dims(rgb_to_chw_01(a_rgb), 0) * 2 - 1).to(device)
                
                clean_cam = cam(input_tensor=t_clean, targets=[ClassifierOutputTarget(sample['clean_pred_class'])])[0, :]
                adv_cam = cam(input_tensor=t_adv, targets=[ClassifierOutputTarget(sample['adv_pred_class'])])[0, :]
                
                plot_gradcam_shift(c_rgb, a_rgb, clean_cam, adv_cam, True, str(case_dir / "1_attention_shift.png"))
                
                resnet.classify = False
                df_unique_clean = df[df['eps'] == nominal_epsilons[0]]
                bg_identities = np.random.choice([i for i in df_unique_clean['identity_name'].unique() if i != identity_name], 4, replace=False)
                bg_df = df_unique_clean[df_unique_clean['identity_name'].isin(bg_identities)]
                
                bg_emb, bg_labels, src_clean_emb, src_adv_emb = [], [], [], []
                
                with torch.no_grad():
                    for _, row in bg_df.iterrows():
                        img_rgb = load_rgb_image(resolve_project_path(base_dir, row['source_image_path']))
                        bg_emb.append(resnet(torch.tensor(np.expand_dims(rgb_to_chw_01(img_rgb), 0) * 2 - 1).to(device)).cpu().numpy()[0])
                        bg_labels.append(row['identity_name'])
                        
                    for _, row in sample_df.iterrows():
                        c_img_rgb = load_rgb_image(resolve_project_path(base_dir, row['source_image_path']))
                        a_img_rgb = load_rgb_image(resolve_project_path(base_dir, row['adversarial_image_path']))
                        src_clean_emb.append(resnet(torch.tensor(np.expand_dims(rgb_to_chw_01(c_img_rgb), 0) * 2 - 1).to(device)).cpu().numpy()[0])
                        src_adv_emb.append(resnet(torch.tensor(np.expand_dims(rgb_to_chw_01(a_img_rgb), 0) * 2 - 1).to(device)).cpu().numpy()[0])
                
                resnet.classify = True
                
                adv_success_flags = (sample_df['adv_pred_class'] == sample_df['target_class']).values
                adv_target_names, adv_actual_pred_names = [], []
                
                for _, row in sample_df.iterrows():
                    tgt_id = row['target_class']
                    pred_id = row['adv_pred_class']
                    
                    t_name_df = df_unique_clean[df_unique_clean['clean_pred_class'] == tgt_id]
                    p_name_df = df_unique_clean[df_unique_clean['clean_pred_class'] == pred_id]
                    
                    adv_target_names.append(t_name_df['identity_name'].iloc[0] if not t_name_df.empty else f"Class {tgt_id}")
                    adv_actual_pred_names.append(p_name_df['identity_name'].iloc[0] if not p_name_df.empty else f"Class {pred_id}")
                
                tgt_clean_emb = None
                unique_targets = sample_df['target_class'].unique()
                if len(unique_targets) == 1:
                    tgt_df = df_unique_clean[df_unique_clean['clean_pred_class'] == unique_targets[0]]
                    if not tgt_df.empty:
                        tgt_clean_emb = []
                        resnet.classify = False
                        with torch.no_grad():
                            for _, row in tgt_df.iterrows():
                                img_rgb = load_rgb_image(resolve_project_path(base_dir, row['source_image_path']))
                                tgt_clean_emb.append(resnet(torch.tensor(np.expand_dims(rgb_to_chw_01(img_rgb), 0) * 2 - 1).to(device)).cpu().numpy()[0])
                        resnet.classify = True
                        tgt_clean_emb = np.array(tgt_clean_emb)
                
                plot_latent_trajectory(
                    np.array(bg_emb), bg_labels, np.array(src_clean_emb), np.array(src_adv_emb),
                    f'"{identity_name}" attacked with \u03B5={PIVOT_EPS:.3f}',  
                    adv_success_flags, adv_target_names, adv_actual_pred_names, tgt_clean_emb,
                    True, str(case_dir / "2_umap_trajectory.png")
                )

            if strategy.startswith("rr_"):
                pass 
            else:
                pass 

    print("\n[OK] Pipeline conclusa con successo!")

if __name__ == "__main__":
    main()