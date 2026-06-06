import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path

# Utilizziamo PYTHONPATH=src per gli import
from util.plot.utils_plot_generic import (
    plot_security_evaluation_curves,
    plot_confidence_degradation,
)
from util.plot.utils_plot_shared import (
    plot_adversarial_showcase,
    plot_frequency_spectrum,
)
from util.google_logger import GoogleSheetLogger 

def main():
    print("======================================================")
    print(" METRICHE & PLOT: DEEPFOOL (Error-Generic / Untargeted) ")
    print("======================================================\n")

    # =========================================================
    # BLOCCO 0: SETUP E CARICAMENTO CSV
    # =========================================================
    base_dir = Path.cwd()
    tracker_csv_path = base_dir / "dataset" / "attacks" / "NN1" / "error_generic" / "deepfool" / "tracker_deepfool.csv"
    output_eval_dir = base_dir / "plots" / "3_Adversarial_Examples" / "error_generic" / "deepfool"
    
    progression_dir = output_eval_dir / "visual_progression"
    
    for d in [output_eval_dir, progression_dir]:
        d.mkdir(parents=True, exist_ok=True)

    if not tracker_csv_path.exists():
        raise FileNotFoundError(f"Errore: Tracker CSV non trovato in {tracker_csv_path}. Esegui prima samples_gen.py")

    df = pd.read_csv(tracker_csv_path)
    total_images = len(df)
    print(f"-> Trovate {total_images} immagini generate da DeepFool.")

    # Il file contiene già: clean_pred_class, adv_pred_class, clean_confidence, adv_confidence

    # --- INIZIALIZZAZIONE LOGGER ---
    logger = GoogleSheetLogger() 

    # =========================================================
    # BLOCCO 2: GENERAZIONE CURVE DI VALUTAZIONE
    # =========================================================
    print(f"\n[BLOCCO 2] Generazione Grafici Globali (Robust Accuracy Curve)...")
    
    epsilons = [0.0, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20]
    asr_dict = {"DeepFool Untargeted": []}
    confidence_data = []

    # Calcoliamo la robustezza in modo "retroattivo"
    for eps in epsilons:
        # 1. Fallimenti per sforamento budget
        resisted_budget = df['linf'] > eps
        
        # 2. Fallimenti matematici (Sotto budget ma la classe non cambia)
        within_budget_mask = df['linf'] <= eps
        resisted_attack = within_budget_mask & (df['adv_pred_class'] == df['clean_pred_class'])
        
        total_resisted = len(df[resisted_budget | resisted_attack])
        robust_accuracy = total_resisted / total_images
        
        # --- LOGGING SU GOOGLE SHEETS ---
        try:
            logger.log_attack_metrics(
                tester="Leonardo",
                attack_type="DeepFool Error-Generic",
                strategy="Untargeted",
                epsilon=eps,
                defense_type="None",
                robust_accuracy=robust_accuracy,
                targeted_asr=0.0, 
                untargeted_asr=1.0 - robust_accuracy, 
                notes="Valutazione retroattiva su L-inf limit"
            )
        except Exception as e:
            print(f"[WARNING] Errore Google Logger: {e}")
        # ----------------------------------------

        asr_dict["DeepFool Untargeted"].append(robust_accuracy)
        
        # Raccogliamo le confidenze. Se sfora il budget, usiamo l'immagine originale
        confidences = np.where(df['linf'] > eps, df['clean_confidence'], df['adv_confidence'])
        confidence_data.append(confidences)

    plot_security_evaluation_curves(epsilons, asr_dict, "NN1 (FaceNet)", True, str(output_eval_dir / "robust_accuracy_curve.png"))
    plot_confidence_degradation(epsilons, confidence_data, "DeepFool Untargeted", True, str(output_eval_dir / "confidence_degradation.png"))

    # =========================================================
    # BLOCCO 3: VISUAL SHOWCASE
    # =========================================================
    print(f"\n[BLOCCO 3] Generazione Visual Showcase per Epsilon...")
    
    for eps in epsilons:
        if eps == 0.0: continue 
        
        # Troviamo immagini in cui l'L_inf si è avvicinato molto a questo Epsilon (entro 0.02) e l'attacco è riuscito
        suitable_samples = df[
            (df['adv_pred_class'] != df['clean_pred_class']) & 
            (df['linf'] <= eps) & 
            (df['linf'] > (eps - 0.02))
        ]
        
        if not suitable_samples.empty:
            sample = suitable_samples.iloc[0]
            
            # --- MODIFICA TIFF 32-BIT ---
            # Carichiamo i TIFF senza perdere la precisione float32
            c_bgr_float32 = cv2.imread(str(base_dir / sample['source_image_path']), cv2.IMREAD_UNCHANGED)
            a_bgr_float32 = cv2.imread(str(base_dir / sample['adversarial_image_path']), cv2.IMREAD_UNCHANGED)
            
            # Controllo sicurezza path
            if c_bgr_float32 is None or a_bgr_float32 is None: continue
            
            # Convertiamo in RGB. Essendo già in range [0.0, 1.0] float32, non serve dividere per 255
            c_rgb = cv2.cvtColor(c_bgr_float32, cv2.COLOR_BGR2RGB)
            a_rgb = cv2.cvtColor(a_bgr_float32, cv2.COLOR_BGR2RGB)

            eps_str_fmt = f"{eps:.3f}".replace('.', '_')
            
            plot_adversarial_showcase(
                c_rgb, a_rgb, 
                f"Orig: {sample['identity_name']}", f"Pred: ID {sample['adv_pred_class']}", 
                True, str(progression_dir / f"showcase_eps_limit_{eps_str_fmt}.png")
            )
            plot_frequency_spectrum(c_rgb, a_rgb, True, str(progression_dir / f"spectrum_eps_limit_{eps_str_fmt}.png"))
        else:
            print(f" -> [SKIP] Nessun campione rappresentativo trovato vicino a L_inf = {eps:.3f}")

    print("\n[OK] Pipeline di Evaluation Error-Generic conclusa con successo!")

if __name__ == "__main__":
    main()