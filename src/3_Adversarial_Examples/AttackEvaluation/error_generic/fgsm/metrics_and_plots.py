import os
# --- FIX PER I CRASH SILENZIOSI SU WINDOWS ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["NUMBA_NUM_THREADS"] = "1"
os.environ["TORCH_CUDNN_V8_API_ENABLED"] = "0"
os.environ["MKL_NUM_THREADS"] = "1"       
os.environ["OPENBLAS_NUM_THREADS"] = "1"  

import sys
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path

# Disabilitiamo CUDNN per evitare il mismatch di librerie
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# =========================================================================
# RISOLUZIONE ROBUSTA DEI PATH 
# =========================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[5]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from facenet_pytorch import InceptionResnetV1
from util.google_logger import GoogleSheetLogger

# Import per Untargeted (Generic)
from util.plot.utils_plot_generic import (
    plot_security_evaluation_curves,
    plot_confidence_degradation,
)
from util.plot.utils_plot_shared import (
    plot_adversarial_showcase,
    plot_frequency_spectrum,
)

IMAGE_SIZE = 160

def resolve_project_path(base_dir: Path, path_value) -> Path:
    path = Path(str(path_value))
    if path.is_absolute():
        return path
    return base_dir / path

def load_rgb_image(path: Path, image_size: int = IMAGE_SIZE) -> np.ndarray:
    image_bgr_float32 = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image_bgr_float32 is None:
        raise FileNotFoundError(f"TIFF non leggibile: {path}")
    if image_bgr_float32.ndim != 3 or image_bgr_float32.shape[2] != 3:
        raise ValueError(
            f"TIFF RGB non valido: {path}, shape={image_bgr_float32.shape}"
        )
    if image_bgr_float32.shape[:2] != (image_size, image_size):
        image_bgr_float32 = cv2.resize(
            image_bgr_float32,
            (image_size, image_size),
            interpolation=cv2.INTER_LINEAR,
        )

    image_rgb_float32 = cv2.cvtColor(image_bgr_float32, cv2.COLOR_BGR2RGB)
    return image_rgb_float32.astype(np.float32)

def rgb_to_chw_01(image_rgb: np.ndarray) -> np.ndarray:
    return np.transpose(image_rgb, (2, 0, 1)).astype(np.float32)

def main():
    print("======================================================")
    print(" METRICHE & PLOT: FGSM ERROR-GENERIC (Untargeted)     ")
    print("======================================================\n")

    base_dir = PROJECT_ROOT
    print(f"-> Project Root impostata a: {base_dir}")

    # =========================================================
    # BLOCCO 0: SETUP E CARICAMENTO CSV
    # =========================================================
    attacks_dir = base_dir / "dataset" / "attacks" / "NN1" / "error_generic" / "fgsm"
    output_eval_dir = base_dir / "plots" / "3_Adversarial_Examples" / "error_generic" / "fgsm"
    
    progression_dir = output_eval_dir / "visual_progression"
    
    for d in [output_eval_dir, progression_dir]:
        d.mkdir(parents=True, exist_ok=True)

    tracker_files = list(attacks_dir.glob("eps_*/tracker_eps_*.csv"))
    
    if not tracker_files:
        raise FileNotFoundError(f"Nessun file tracker trovato in {attacks_dir}. Esegui prima la generazione.")

    print(f"-> Trovati {len(tracker_files)} file tracker. Unione in corso...")
    df_list = [pd.read_csv(f) for f in tracker_files]
    df = pd.concat(df_list, ignore_index=True)
    
    df['eps'] = pd.to_numeric(df['eps'], errors='raise').astype(float)
    epsilons = sorted(df['eps'].unique())
    print(f"-> Epsilon rilevati: {epsilons}")

    if 'adv_pred_class' not in df.columns:
        df['clean_pred_class'] = -1 
        df['adv_pred_class'] = -1
        df['clean_class_confidence'] = 0.0

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"-> Inizializzazione NN1 globale su {device}...")
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    resnet.classify = True 

    # Inizializzazione Logger
    logger = GoogleSheetLogger()

    # =========================================================
    # BLOCCO 1: INFERENZA DELLE IMMAGINI
    # =========================================================
    batch_size = 64 
    print(f"\n[BLOCCO 1] Inferenza delle immagini avversarie (Untargeted)...")

    with torch.no_grad():
        for eps in epsilons:
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
                
                # Predizione su Clean per trovare la classe originale
                clean_logits = resnet(x_clean_tensor * 2 - 1)
                clean_preds = torch.argmax(clean_logits, dim=1).cpu().numpy()
                
                # Predizione su Adv
                adv_logits = resnet(x_adv_tensor * 2 - 1)
                adv_preds = torch.argmax(adv_logits, dim=1).cpu().numpy()
                adv_probs = F.softmax(adv_logits, dim=1).cpu().numpy()
                
                for j in range(len(adv_preds)):
                    c_pred = int(clean_preds[j])
                    a_pred = int(adv_preds[j])
                    
                    # Salviamo la confidenza residua della classe ORIGINALE 
                    # (Vogliamo vedere come crolla la confidenza della classe giusta)
                    residual_confidence = adv_probs[j, c_pred]
                    
                    original_idx = batch_df.index[j]
                    df.loc[original_idx, 'clean_pred_class'] = c_pred
                    df.loc[original_idx, 'adv_pred_class'] = a_pred
                    df.loc[original_idx, 'clean_class_confidence'] = residual_confidence

    evaluated_csv_path = output_eval_dir / "fgsm_untargeted_evaluated.csv"
    df.to_csv(evaluated_csv_path, index=False)
    print(f"-> Master Data salvato in {evaluated_csv_path}")

    # =========================================================
    # BLOCCO 2: GENERAZIONE CURVE DI VALUTAZIONE
    # =========================================================
    print(f"\n[BLOCCO 2] Generazione Grafici Globali (Robust Accuracy & Degradation)...")
    
    asr_dict = {"FGSM Untargeted (Robust Acc)": []}
    confidence_data = []

    for eps in epsilons:
        df_eps = df[df['eps'] == eps]
        total_images = len(df_eps)
        
        # In Untargeted, la rete resiste se predice ancora la classe corretta
        resisted_mask = df_eps['adv_pred_class'] == df_eps['clean_pred_class']
        resisted = resisted_mask.sum()
        
        robust_accuracy = resisted / total_images
        untargeted_asr = 1.0 - robust_accuracy
        
        asr_dict["FGSM Untargeted (Robust Acc)"].append(robust_accuracy)
        confidence_data.append(df_eps['clean_class_confidence'].values)

        # --- LOGGING SU GOOGLE SHEETS ---
        if hasattr(logger, 'log_attack_metrics'):
            logger.log_attack_metrics(
                tester="Andrea",
                attack_type="FGSM Error-Generic",
                strategy="Untargeted",
                epsilon=eps,
                defense_type="None",
                robust_accuracy=robust_accuracy,
                targeted_asr=0.0,
                untargeted_asr=untargeted_asr,
                notes="Valutazione Untargeted TIFF 32-bit"
            )
        else:
            logger.log_biometric_metrics(
                tester="Andrea", 
                phase="Error-Generic Evaluation",
                attack_type="FGSM",
                epsilon=eps,
                defense_type="None",
                accuracy=robust_accuracy, 
                eer=0.0,
                far=0.0,
                frr=0.0,
                threshold=0.0,
                notes=f"Untargeted ASR: {untargeted_asr:.1%}"
            )

    plot_security_evaluation_curves(epsilons, asr_dict, "NN1 (FaceNet)", True, str(output_eval_dir / "robust_accuracy_curve.png"))
    plot_confidence_degradation(epsilons, confidence_data, "FGSM Untargeted", True, str(output_eval_dir / "confidence_degradation.png"))

    # =========================================================
    # BLOCCO 3: VISUAL SHOWCASE
    # =========================================================
    print(f"\n[BLOCCO 3] Generazione Visual Showcase per Epsilon...")
    
    sample_source_path = df['source_image_path'].iloc[0]
    
    for eps in epsilons:
        sample_candidates = df[(df['eps'] == eps) & (df['source_image_path'] == sample_source_path)]
        if sample_candidates.empty: continue
        sample = sample_candidates.iloc[0]
        
        c_rgb = load_rgb_image(resolve_project_path(base_dir, sample['source_image_path']))
        a_rgb = load_rgb_image(resolve_project_path(base_dir, sample['adversarial_image_path']))

        eps_str_fmt = f"{eps:.3f}".replace('.', '_')
        
        # Mostriamo come l'ID predetto cambia o resiste
        status_text = "RESISTED" if sample['adv_pred_class'] == sample['clean_pred_class'] else "FOOLED"
        
        plot_adversarial_showcase(
            c_rgb, a_rgb, 
            f"Orig: ID {int(sample['clean_pred_class'])}", 
            f"Pred: ID {int(sample['adv_pred_class'])} ({status_text})", 
            True, str(progression_dir / f"showcase_eps_{eps_str_fmt}.png")
        )
        plot_frequency_spectrum(c_rgb, a_rgb, True, str(progression_dir / f"spectrum_eps_{eps_str_fmt}.png"))

    print("\n[OK] Pipeline di Evaluation Error-Generic conclusa con successo!")

if __name__ == "__main__":
    main()
