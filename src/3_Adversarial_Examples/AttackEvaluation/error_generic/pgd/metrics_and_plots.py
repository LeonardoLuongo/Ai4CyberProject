import os
# =========================================================================
# WORKAROUND CUDNN (Fondamentale per evitare il crash in inferenza)
# =========================================================================
os.environ["TORCH_CUDNN_V8_API_ENABLED"] = "0" 

import cv2
import numpy as np
import pandas as pd
import torch

# Disabilitiamo CUDNN per evitare il mismatch di librerie
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path
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
from util.google_logger import GoogleSheetLogger 

def main():
    print("======================================================")
    print(" METRICHE & PLOT: PGD (Error-Generic / Untargeted)    ")
    print("======================================================\n")

    # =========================================================
    # BLOCCO 0: SETUP E CARICAMENTO CSV DISTRIBUITI
    # =========================================================
    base_dir = Path.cwd()
    attacks_dir = base_dir / "dataset" / "attacks" / "error_generic" / "pgd"
    output_eval_dir = base_dir / "plots" / "3_Adversarial_Examples" / "error_generic" / "pgd"
    
    progression_dir = output_eval_dir / "visual_progression"
    
    for d in [output_eval_dir, progression_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Ricerca di tutti i CSV di tracciamento nelle cartelle eps_X_XXX
    tracker_files = list(attacks_dir.glob("eps_*/tracker_eps_*.csv"))
    
    if not tracker_files:
        raise FileNotFoundError(f"Nessun file tracker trovato in {attacks_dir}. Esegui prima samples_gen.py")

    print(f"-> Trovati {len(tracker_files)} file tracker locali. Unione in corso...")
    df_list = [pd.read_csv(f) for f in tracker_files]
    df = pd.concat(df_list, ignore_index=True)
    
    epsilons = sorted(df['eps'].unique())
    print(f"-> Epsilon rilevati: {epsilons}")

    # Pre-inizializziamo le colonne per le metriche
    if 'adv_pred_class' not in df.columns:
        df['clean_pred_class'] = -1 
        df['adv_pred_class'] = -1
        df['adv_confidence'] = 0.0

    # --- INIZIALIZZAZIONE LOGGER E MODELLO ---
    logger = GoogleSheetLogger() 
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"-> Inizializzazione NN1 globale su {device}...")
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    resnet.classify = True 

    # =========================================================
    # BLOCCO 1: INFERENZA IN BATCH (Per calcolare predizioni e confidenze)
    # =========================================================
    batch_size = 64 
    print(f"\n[BLOCCO 1] Inferenza delle immagini avversarie e originali...")

    with torch.no_grad():
        for eps in epsilons:
            df_eps = df[df['eps'] == eps]
            
            for i in tqdm(range(0, len(df_eps), batch_size), desc=f"Inferenza eps={eps:.3f}"):
                batch_df = df_eps.iloc[i : i + batch_size]
                
                x_adv_batch, x_clean_batch = [], []
                for _, row in batch_df.iterrows():
                    c_rgb = cv2.cvtColor(cv2.resize(cv2.imread(str(base_dir / row['source_image_path'])), (160, 160)), cv2.COLOR_BGR2RGB)
                    a_rgb = cv2.cvtColor(cv2.imread(str(base_dir / row['adversarial_image_path'])), cv2.COLOR_BGR2RGB)
                    x_clean_batch.append(np.transpose(c_rgb, (2, 0, 1)).astype(np.float32) / 255.0)
                    x_adv_batch.append(np.transpose(a_rgb, (2, 0, 1)).astype(np.float32) / 255.0)
                    
                x_clean_tensor = torch.tensor(np.array(x_clean_batch)).to(device)
                x_adv_tensor = torch.tensor(np.array(x_adv_batch)).to(device)
                
                clean_preds = torch.argmax(resnet(x_clean_tensor * 2 - 1), dim=1).cpu().numpy()
                
                adv_logits = resnet(x_adv_tensor * 2 - 1)
                adv_preds = torch.argmax(adv_logits, dim=1).cpu().numpy()
                adv_probs = F.softmax(adv_logits, dim=1).cpu().numpy()
                
                for j in range(len(adv_preds)):
                    c_pred = clean_preds[j]
                    a_pred = adv_preds[j]
                    
                    # Nell'Error-Generic, la confidenza che ci interessa è:
                    # "Quanto la rete crede ANCORA che l'immagine sia la classe vera (c_pred)?"
                    conf_on_true_class = adv_probs[j, c_pred]
                    
                    original_idx = batch_df.index[j]
                    df.loc[original_idx, 'clean_pred_class'] = c_pred
                    df.loc[original_idx, 'adv_pred_class'] = a_pred
                    df.loc[original_idx, 'adv_confidence'] = conf_on_true_class

    evaluated_csv_path = output_eval_dir / "pgd_untargeted_evaluated.csv"
    df.to_csv(evaluated_csv_path, index=False)
    print(f"-> Master Data salvato in {evaluated_csv_path}")

    # =========================================================
    # BLOCCO 2: GENERAZIONE CURVE DI VALUTAZIONE
    # =========================================================
    print(f"\n[BLOCCO 2] Generazione Grafici Globali (Robust Accuracy Curve)...")
    
    asr_dict = {"PGD Untargeted": []}
    confidence_data = []

    for eps in epsilons:
        df_eps = df[df['eps'] == eps]
        total_images = len(df_eps)
        
        # Le immagini in cui la rete NON è stata ingannata (L'attacco ha fallito)
        resisted_attack = df_eps['adv_pred_class'] == df_eps['clean_pred_class']
        
        total_resisted = resisted_attack.sum()
        robust_accuracy = total_resisted / total_images
        
        # --- LOGGING SU GOOGLE SHEETS ---
        try:
            logger.log_attack_metrics(
                tester="Francesco",  
                attack_type="PGD Error-Generic",
                strategy="Untargeted",
                epsilon=eps,
                defense_type="None",
                robust_accuracy=robust_accuracy,
                targeted_asr=0.0, 
                untargeted_asr=1.0 - robust_accuracy, 
                notes="Valutazione Clean -> Adv"
            )
        except Exception as e:
            print(f"[WARNING] Errore Google Logger: {e}")
        # ----------------------------------------

        asr_dict["PGD Untargeted"].append(robust_accuracy)
        confidence_data.append(df_eps['adv_confidence'].values)

    plot_security_evaluation_curves(epsilons, asr_dict, "NN1 (FaceNet)", True, str(output_eval_dir / "robust_accuracy_curve.png"))
    plot_confidence_degradation(epsilons, confidence_data, "PGD Untargeted", True, str(output_eval_dir / "confidence_degradation.png"))

    # =========================================================
    # BLOCCO 3: VISUAL SHOWCASE
    # =========================================================
    print(f"\n[BLOCCO 3] Generazione Visual Showcase per Epsilon...")
    
    sample_source_path = df['source_image_path'].iloc[0]
    
    for eps in epsilons:
        # Prende lo stesso campione per tutti gli Epsilon per vedere la progressione
        sample_df = df[(df['eps'] == eps) & (df['source_image_path'] == sample_source_path)]
        
        if not sample_df.empty:
            sample = sample_df.iloc[0]
            
            c_bgr = cv2.imread(str(base_dir / sample['source_image_path']))
            a_bgr = cv2.imread(str(base_dir / sample['adversarial_image_path']))
            
            if c_bgr is None or a_bgr is None: continue
            
            c_rgb = cv2.cvtColor(cv2.resize(c_bgr, (160, 160)), cv2.COLOR_BGR2RGB)
            a_rgb = cv2.cvtColor(cv2.resize(a_bgr, (160, 160)), cv2.COLOR_BGR2RGB)

            eps_str_fmt = f"{eps:.3f}".replace('.', '_')
            
            # --- FIX NOMI ---
            plot_adversarial_showcase(
                c_rgb, a_rgb, 
                f"ID {int(sample['clean_pred_class'])}", 
                f"ID {int(sample['adv_pred_class'])}", 
                True, str(progression_dir / f"showcase_eps_{eps_str_fmt}.png")
            )
            plot_frequency_spectrum(c_rgb, a_rgb, True, str(progression_dir / f"spectrum_eps_{eps_str_fmt}.png"))

    print("\n[OK] Pipeline di Evaluation Error-Generic conclusa con successo!")

if __name__ == "__main__":
    main()