import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path
from util.google_logger import GoogleSheetLogger
from facenet_pytorch import InceptionResnetV1

# Utilizziamo PYTHONPATH=src per gli import
from util.plot.utils_plot_generic import (
    plot_security_evaluation_curves,
    plot_confidence_degradation,
)
from util.plot.utils_plot_shared import (
    plot_adversarial_showcase,
    plot_frequency_spectrum,
)

def main():
    print("======================================================")
    print(" METRICHE & PLOT: CARLINI-WAGNER L_INF (Error-Generic)")
    print("======================================================\n")

    # =========================================================
    # BLOCCO 0: SETUP E CARICAMENTO CSV
    # =========================================================
    base_dir = Path.cwd()
    tracker_csv_path = base_dir / "dataset" / "attacks" / "NN1" / "error_generic" / "cw" / "tracker_cw_untargeted.csv"
    output_eval_dir = base_dir / "plots" / "3_Adversarial_Examples" / "error_generic" / "cw"
    
    progression_dir = output_eval_dir / "visual_progression"
    
    for d in [output_eval_dir, progression_dir]:
        d.mkdir(parents=True, exist_ok=True)

    if not tracker_csv_path.exists():
        raise FileNotFoundError(f"Errore: Tracker CSV non trovato in {tracker_csv_path}. Esegui prima samples_gen.py")

    df = pd.read_csv(tracker_csv_path)
    total_images = len(df)
    print(f"-> Trovate {total_images} immagini generate da Carlini-Wagner.")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"-> Inizializzazione NN1 globale su {device}...")
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    resnet.classify = True 

    if 'adv_pred_class' not in df.columns:
        df['adv_pred_class'] = -1
        df['clean_confidence'] = 0.0
        df['adv_confidence'] = 0.0

    # =========================================================
    # BLOCCO 1: INFERENZA BATCH (Valutazione Esito)
    # =========================================================
    batch_size = 64 
    print(f"\n[BLOCCO 1] Inferenza delle immagini avversarie e originali...")

    with torch.no_grad():
        for i in tqdm(range(0, total_images, batch_size), desc="Inferenza Batch"):
            batch_df = df.iloc[i : i + batch_size]
            
            x_adv_batch, x_clean_batch = [], []
            for _, row in batch_df.iterrows():
                
                # CLEAN (È un TIFF 32-bit float [0.0-1.0])
                c_bgr_float32 = cv2.imread(str(base_dir / row['source_image_path']), cv2.IMREAD_UNCHANGED)
                c_rgb_float32 = cv2.cvtColor(c_bgr_float32, cv2.COLOR_BGR2RGB)
                x_clean_batch.append(np.transpose(c_rgb_float32, (2, 0, 1)))
                
                # ADVERSARIAL (È un TIFF 32-bit float [0.0-1.0])
                a_bgr_float32 = cv2.imread(str(base_dir / row['adversarial_image_path']), cv2.IMREAD_UNCHANGED)
                a_rgb_float32 = cv2.cvtColor(a_bgr_float32, cv2.COLOR_BGR2RGB)
                x_adv_batch.append(np.transpose(a_rgb_float32, (2, 0, 1)))
                
            x_clean_tensor = torch.tensor(np.array(x_clean_batch)).to(device)
            x_adv_tensor = torch.tensor(np.array(x_adv_batch)).to(device)
            
            # Inferenza pura PyTorch (Normalizzazione per FaceNet [-1, 1])
            clean_logits = resnet(x_clean_tensor * 2 - 1)
            adv_logits = resnet(x_adv_tensor * 2 - 1)
            
            adv_preds = torch.argmax(adv_logits, dim=1).cpu().numpy()
            
            clean_probs = F.softmax(clean_logits, dim=1).cpu().numpy()
            adv_probs = F.softmax(adv_logits, dim=1).cpu().numpy()
            
            for j in range(len(adv_preds)):
                a_pred = adv_preds[j]
                
                # La vera classe la prendiamo direttamente dal tracker per evitare mismatch
                true_pred_class = batch_df['clean_pred_class'].iloc[j]
                
                original_idx = batch_df.index[j]
                df.loc[original_idx, 'adv_pred_class'] = a_pred
                df.loc[original_idx, 'clean_confidence'] = clean_probs[j, true_pred_class]
                df.loc[original_idx, 'adv_confidence'] = adv_probs[j, true_pred_class]

    evaluated_csv_path = output_eval_dir / "cw_generic_evaluated.csv"
    df.to_csv(evaluated_csv_path, index=False)
    print(f"-> Master Data salvato in {evaluated_csv_path}")

    # =========================================================
    # BLOCCO 2: GENERAZIONE CURVE DI VALUTAZIONE E LOGGING
    # =========================================================
    print(f"\n[BLOCCO 2] Generazione Grafici Globali e Logging...")
    
    epsilons = [0.0, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20]
    asr_dict = {"C&W L_inf (Untargeted)": []}
    confidence_data = []
    
    logger = GoogleSheetLogger()

    for eps in epsilons:
        resisted_budget = df['linf'] > eps
        within_budget_mask = df['linf'] <= eps
        resisted_attack = within_budget_mask & (df['adv_pred_class'] == df['clean_pred_class'])
        
        # Le immagini che la rete ha indovinato o in cui l'attacco ha sforato l'L_inf consentito
        total_resisted = len(df[resisted_budget | resisted_attack])
        robust_accuracy = total_resisted / total_images
        
        # Le immagini che l'attacco è riuscito a far sbagliare (legalmente)
        total_untargeted_success = len(df[within_budget_mask & (df['adv_pred_class'] != df['clean_pred_class'])])
        untargeted_asr = total_untargeted_success / total_images
        
        asr_dict["C&W L_inf (Untargeted)"].append(robust_accuracy)
        
        confidences = np.where(df['linf'] > eps, df['clean_confidence'], df['adv_confidence'])
        confidence_data.append(confidences)
        
        # --- LOGGING CLOUD (FORMATO DEL TEAM) ---
        if hasattr(logger, 'log_attack_metrics'):
            logger.log_attack_metrics(
                tester="Leonardo",
                attack_type="C&W Error-Generic",
                strategy="Untargeted",
                epsilon=eps,
                defense_type="None",
                robust_accuracy=robust_accuracy,
                targeted_asr=0.0,
                untargeted_asr=untargeted_asr,
                notes="Valutazione Untargeted TIFF 32-bit"
            )

    plot_security_evaluation_curves(epsilons, asr_dict, "NN1 (FaceNet)", True, str(output_eval_dir / "robust_accuracy_curve.png"))
    plot_confidence_degradation(epsilons, confidence_data, "Carlini-Wagner Untargeted", True, str(output_eval_dir / "confidence_degradation.png"))

    # =========================================================
    # BLOCCO 3: VISUAL SHOWCASE
    # =========================================================
    print(f"\n[BLOCCO 3] Generazione Visual Showcase per Epsilon...")
    
    for eps in epsilons:
        if eps == 0.0: continue 
        
        # Scegliamo immagini che C&W è riuscito a violare posizionandosi vicinissimo a questo limite
        suitable_samples = df[
            (df['adv_pred_class'] != df['clean_pred_class']) & 
            (df['linf'] <= eps) & 
            (df['linf'] > (eps - 0.02))
        ]
        
        if not suitable_samples.empty:
            sample = suitable_samples.iloc[0]
            
            # Entrambi TIFF 32-bit!
            c_bgr_float32 = cv2.imread(str(base_dir / sample['source_image_path']), cv2.IMREAD_UNCHANGED)
            a_bgr_float32 = cv2.imread(str(base_dir / sample['adversarial_image_path']), cv2.IMREAD_UNCHANGED)
            
            if c_bgr_float32 is None or a_bgr_float32 is None: continue
            
            # Non servono /255.0 o astype. Sono già pronte.
            c_rgb = cv2.cvtColor(c_bgr_float32, cv2.COLOR_BGR2RGB)
            a_rgb = cv2.cvtColor(a_bgr_float32, cv2.COLOR_BGR2RGB)

            eps_str_fmt = f"{eps:.3f}".replace('.', '_')
            
            # plot_adversarial_showcase calcolerà un rumore perfetto grazie ai float32!
            plot_adversarial_showcase(
                c_rgb, a_rgb, 
                f"Orig: {sample['identity_name']}", f"Pred: ID {sample['adv_pred_class']}", 
                True, str(progression_dir / f"showcase_eps_limit_{eps_str_fmt}.png")
            )
            plot_frequency_spectrum(c_rgb, a_rgb, True, str(progression_dir / f"spectrum_eps_limit_{eps_str_fmt}.png"))
        else:
            print(f" -> [SKIP] Nessun campione rappresentativo C&W trovato vicino a L_inf = {eps:.3f}")

    print("\n[OK] Pipeline di Evaluation C&W Error-Generic conclusa con successo!")

if __name__ == "__main__":
    main()