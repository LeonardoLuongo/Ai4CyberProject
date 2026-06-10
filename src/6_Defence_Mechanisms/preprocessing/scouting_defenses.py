import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Impostazioni estetiche per i plot
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

from facenet_pytorch import InceptionResnetV1
from art.defences.preprocessor import JpegCompression, FeatureSqueezing, SpatialSmoothing

# =========================================================================
# FUNZIONI DI SUPPORTO PER ESTRAZIONE DATI
# =========================================================================
def get_best_attack_samples(tracker_path: Path, targeted: bool, max_eps: float = 0.10, sample_size: int = 50):
    df = pd.read_csv(tracker_path)
    
    # 1. Filtriamo solo gli attacchi sotto il budget
    df_valid = df[df['linf'] <= max_eps].copy()
    
    # 2. Definiamo il "Successo" dell'attacco
    if targeted:
        df_valid['success'] = df_valid['adv_pred_class'] == df_valid['target_class']
    else:
        df_valid['success'] = df_valid['adv_pred_class'] != df_valid['clean_pred_class']
        
    # 3. Troviamo l'Epsilon con il Success Rate più alto
    asr_by_eps = df_valid.groupby('linf')['success'].mean()
    best_eps = asr_by_eps.idxmax()
    max_asr = asr_by_eps.max()
    
    print(f"   -> Epsilon ottimale trovato: {best_eps} (ASR: {max_asr*100:.1f}%)")
    
    # 4. Filtriamo solo i successi in quell'epsilon
    successful_samples = df_valid[(df_valid['linf'] == best_eps) & (df_valid['success'] == True)]
    
    # 5. Campionamento casuale
    if len(successful_samples) > sample_size:
        successful_samples = successful_samples.sample(n=sample_size, random_state=42)
        
    return successful_samples

def load_images_from_df(df, base_dir, is_clean=False):
    images = []
    labels = []
    
    for _, row in df.iterrows():
        path = str(base_dir / (row['source_image_path'] if is_clean else row['adversarial_image_path']))
        
        # Carichiamo il TIFF a 32-bit o il PNG a 8-bit e lo portiamo in float32 [0, 1] RGB
        img_bgr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img_bgr is None: continue
        
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        if img_rgb.dtype == np.uint8:
            img_rgb = img_rgb.astype(np.float32) / 255.0
            
        images.append(img_rgb)
        labels.append(row['clean_pred_class']) # Vogliamo sempre recuperare la VERA identità!
        
    # ART si aspetta (N, H, W, C) per i preprocessor di default se non si specifica channels_first
    # Ma per sicurezza usiamo il formato nativo PyTorch CHW e diciamo ad ART di usare channels_first
    return np.transpose(np.stack(images), (0, 3, 1, 2)), np.array(labels)

# =========================================================================
# MAIN SCOUTING SCRIPT
# =========================================================================
def main():
    print("======================================================")
    print(" SCOUTING DIFESE: PRE-PROCESSING HYPERPARAMETER TUNING")
    print("======================================================\n")

    base_dir = Path.cwd()
    output_dir = base_dir / "plots" / "6_Defence_Mechanisms" / "scouting"
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"-> Inizializzazione Rete NN1 su {device}...")
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    resnet.classify = True 

    # --- 1. PREPARAZIONE DATASET ---
    print("\n[FASE 1] Estrazione Campioni Strategici...")
    
    # Untargeted
    untarg_tracker = base_dir / "dataset" / "attacks" / "NN1" / "error_generic" / "cw" / "tracker_cw_untargeted.csv"
    print(" -> Analisi C&W Untargeted:")
    df_untarg = get_best_attack_samples(untarg_tracker, targeted=False, max_eps=0.10, sample_size=50)
    
    # Targeted Next-Best
    targ_tracker = base_dir / "dataset" / "attacks" / "NN1" / "error_specific" / "cw" / "next_best" / "tracker_next_best.csv"
    print(" -> Analisi C&W Targeted (Next-Best):")
    df_targ = get_best_attack_samples(targ_tracker, targeted=True, max_eps=0.10, sample_size=50)
    
    # Clean
    print(" -> Selezione 100 immagini Pulite...")
    df_clean = pd.read_csv(base_dir / "dataset" / "clean" / "splits" / "manifest.csv")
    # Facciamo un merge rapido con uno dei tracker per avere le vere label pre-calcolate
    df_clean_merged = df_untarg[['source_image_path', 'clean_pred_class']].drop_duplicates().sample(n=min(100, len(df_untarg)), random_state=42)

    # Caricamento in VRAM/RAM
    print(" -> Caricamento Tensori...")
    x_clean, y_clean = load_images_from_df(df_clean_merged, base_dir, is_clean=True)
    x_untarg, y_untarg = load_images_from_df(df_untarg, base_dir, is_clean=False)
    x_targ, y_targ = load_images_from_df(df_targ, base_dir, is_clean=False)

    # --- 2. SETUP GRIGLIA DIFESE ---
    # ART preprocessors ritornano una tupla (x_defended, y_defended)
    defenses = {
        "JPEG Compression": {
            "param_name": "Quality",
            "configs": [
                ("10", JpegCompression(clip_values=(0.0, 1.0), apply_predict=True, quality=10, channels_first=True)),
                ("30", JpegCompression(clip_values=(0.0, 1.0), apply_predict=True, quality=30, channels_first=True)),
                ("50", JpegCompression(clip_values=(0.0, 1.0), apply_predict=True, quality=50, channels_first=True)),
                ("70", JpegCompression(clip_values=(0.0, 1.0), apply_predict=True, quality=70, channels_first=True)),
                ("90", JpegCompression(clip_values=(0.0, 1.0), apply_predict=True, quality=90, channels_first=True))
            ]
        },
        "Feature Squeezing": {
            "param_name": "Bit Depth",
            "configs": [
                ("3", FeatureSqueezing(clip_values=(0.0, 1.0), apply_predict=True, bit_depth=3)),
                ("4", FeatureSqueezing(clip_values=(0.0, 1.0), apply_predict=True, bit_depth=4)),
                ("5", FeatureSqueezing(clip_values=(0.0, 1.0), apply_predict=True, bit_depth=5))
            ]
        },
        "Spatial Smoothing": {
            "param_name": "Window Size",
            "configs": [
                ("3x3", SpatialSmoothing(window_size=3, channels_first=True)),
                ("5x5", SpatialSmoothing(window_size=5, channels_first=True)),
                ("7x7", SpatialSmoothing(window_size=7, channels_first=True))
            ]
        }
    }

    # Funzione Helper per Inferenza e Accuracy
    def evaluate_tensors(x_np, y_true):
        with torch.no_grad():
            # Forziamo il cast a float32 (.float()) e lo passiamo su device
            x_tensor = torch.tensor(x_np).float().to(device)
            # Rete lavora in [-1, 1]
            preds = torch.argmax(resnet(x_tensor * 2.0 - 1.0), dim=1).cpu().numpy()
            return np.mean(preds == y_true) * 100.0


    # --- 3. SCOUTING LOOP ---
    print("\n[FASE 2] Avvio Grid Search delle Difese...")
    
    # Baseline (Nessuna Difesa)
    base_clean = evaluate_tensors(x_clean, y_clean)
    base_untarg = evaluate_tensors(x_untarg, y_untarg) # Dovrebbe essere 0% essendo attacchi di successo
    base_targ = evaluate_tensors(x_targ, y_targ)       # Dovrebbe essere 0%
    
    for def_name, def_info in defenses.items():
        print(f"\n--- Analisi: {def_name} ---")
        
        param_labels = []
        res_clean = []
        res_untarg = []
        res_targ = []
        
        for param_val, preprocessor in def_info["configs"]:
            print(f" -> Test {def_info['param_name']} = {param_val}")
            
            # Applichiamo la difesa (ART restituisce x_def, y_def)
            x_clean_def, _ = preprocessor(x_clean)
            x_untarg_def, _ = preprocessor(x_untarg)
            x_targ_def, _ = preprocessor(x_targ)
            
            # Valutiamo l'Accuracy
            acc_c = evaluate_tensors(x_clean_def, y_clean)
            acc_u = evaluate_tensors(x_untarg_def, y_untarg)
            acc_t = evaluate_tensors(x_targ_def, y_targ)
            
            param_labels.append(param_val)
            res_clean.append(acc_c)
            res_untarg.append(acc_u)
            res_targ.append(acc_t)
            
            print(f"    Clean Ret: {acc_c:.1f}% | Untarg Rec: {acc_u:.1f}% | Targ Rec: {acc_t:.1f}%")

        # --- 4. PLOTTING INDIVIDUALE PER DIFESA ---
        x = np.arange(len(param_labels))
        width = 0.25

        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Barre
        ax.bar(x - width, res_clean, width, label='Clean Retention (No Attack)', color='forestgreen', edgecolor='white')
        ax.bar(x, res_untarg, width, label='Untargeted Recovery', color='dodgerblue', edgecolor='white')
        ax.bar(x + width, res_targ, width, label='Targeted Recovery (Next-Best)', color='darkorange', edgecolor='white')

        # Linea orizzontale per la baseline pulita
        ax.axhline(y=base_clean, color='forestgreen', linestyle='--', alpha=0.7, label=f'Clean Baseline ({base_clean:.1f}%)')

        ax.set_ylabel('Accuracy (%)', fontsize=12)
        ax.set_xlabel(def_info["param_name"], fontsize=12)
        ax.set_title(f'Defense Scouting: {def_name}', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(param_labels)
        ax.set_ylim(0, 105)
        ax.legend(loc='upper right', bbox_to_anchor=(1.05, 1))

        # Aggiunta etichette sopra le barre
        for bars in ax.containers:
            ax.bar_label(bars, fmt='%.1f%%', padding=3, fontsize=9)

        plt.tight_layout()
        plot_filename = output_dir / f"scouting_{def_name.replace(' ', '_').lower()}.png"
        plt.savefig(plot_filename, dpi=300)
        plt.close()
        
    print(f"\n[OK] Scouting Completato! I 3 grafici a barre sono in: {output_dir}")

if __name__ == "__main__":
    main()