import os
import cv2
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from pathlib import Path

# Impostazioni estetiche per i plot
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

from facenet_pytorch import InceptionResnetV1
from art.defences.preprocessor import JpegCompression, SpatialSmoothing, FeatureSqueezing

# =========================================================================
# FUNZIONI DI SUPPORTO PER ESTRAZIONE E PLOT
# =========================================================================
def get_best_attack_samples(tracker_path: Path, targeted: bool, max_eps: float = 0.10, sample_size: int = 50):
    df = pd.read_csv(tracker_path)
    df_valid = df[df['linf'] <= max_eps].copy()
    
    if targeted:
        df_valid['success'] = df_valid['adv_pred_class'] == df_valid['target_class']
    else:
        df_valid['success'] = df_valid['adv_pred_class'] != df_valid['clean_pred_class']
        
    asr_by_eps = df_valid.groupby('linf')['success'].mean()
    best_eps = asr_by_eps.idxmax()
    
    successful_samples = df_valid[(df_valid['linf'] == best_eps) & (df_valid['success'] == True)]
    if len(successful_samples) > sample_size:
        successful_samples = successful_samples.sample(n=sample_size, random_state=42)
        
    return successful_samples

def load_images_from_df(df, base_dir, is_clean=False):
    images, clean_labels, adv_labels, tgt_labels = [], [], [], []
    for _, row in df.iterrows():
        path = str(base_dir / (row['source_image_path'] if is_clean else row['adversarial_image_path']))
        img_bgr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img_bgr is None: continue
        
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        if img_rgb.dtype == np.uint8:
            img_rgb = img_rgb.astype(np.float32) / 255.0
            
        images.append(img_rgb)
        clean_labels.append(row['clean_pred_class'])
        
        if not is_clean:
            adv_labels.append(row['adv_pred_class'])
            tgt_labels.append(row.get('target_class', -1))
            
    x_chw = np.transpose(np.stack(images), (0, 3, 1, 2))
    return x_chw, np.array(clean_labels), np.array(adv_labels), np.array(tgt_labels)

def get_color(pred, clean_label, tgt_label=-1):
    if pred == clean_label: return 'green'
    if pred == tgt_label: return 'firebrick' 
    return 'red'

def plot_progression_attack(clean_img, adv_img, s1_img, s2_img, s3_img, 
                            preds, clean_lbl, tgt_lbl,
                            title, save_path):
    # ORA ABBIAMO 5 COLONNE
    fig, axes = plt.subplots(2, 5, figsize=(25, 10))
    fig.suptitle(title, fontsize=18, fontweight='bold', y=1.02)
    
    imgs = [clean_img, adv_img, s1_img, s2_img, s3_img]
    titles = ["1. Original Clean", "2. Adversarial (Attacked)", "3. Stage 1: Smoothing", "4. Stage 2: Squeezing", "5. Stage 3: JPEG (Final)"]
    
    for i in range(5):
        # Riga 1: Immagini
        axes[0, i].imshow(np.transpose(imgs[i], (1, 2, 0)))
        pred_lbl = preds[i]
        color = get_color(pred_lbl, clean_lbl, tgt_lbl)
        axes[0, i].set_title(f"{titles[i]}\nPred: ID {pred_lbl}", color=color, fontweight='bold')
        axes[0, i].axis('off')
        
        # Riga 2: Rumore residuo (Amplificato x10)
        if i == 0:
            axes[1, i].axis('off')
        else:
            noise = np.abs(imgs[i] - clean_img)
            noise_vis = np.clip(np.transpose(noise * 10.0, (1, 2, 0)), 0, 1)
            axes[1, i].imshow(noise_vis)
            axes[1, i].set_title(f"Residual Noise (x10)\nMax Diff: {np.max(noise):.4f}")
            axes[1, i].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def plot_progression_clean(clean_img, s1_img, s2_img, s3_img, preds, clean_lbl, title, save_path):
    # 4 COLONNE
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(title, fontsize=18, fontweight='bold', y=1.05)
    
    imgs = [clean_img, s1_img, s2_img, s3_img]
    titles = ["1. Original Clean", "2. Stage 1: Smoothing", "3. Stage 2: Squeezing", "4. Stage 3: JPEG (Final)"]
    
    for i in range(4):
        axes[i].imshow(np.transpose(imgs[i], (1, 2, 0)))
        color = 'green' if preds[i] == clean_lbl else 'red'
        axes[i].set_title(f"{titles[i]}\nPred: ID {preds[i]}", color=color, fontweight='bold')
        axes[i].axis('off')
        
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

# =========================================================================
# MAIN SCRIPT
# =========================================================================
def main():
    print("======================================================")
    print(" TRIPLE-CASCADING DEFENSE: PIPELINE GRID SEARCH       ")
    print("======================================================\n")

    base_dir = Path.cwd()
    out_dir = base_dir / "plots" / "6_Defence_Mechanisms" / "pipeline_showcase"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    resnet.classify = True 

    def infer(x_np):
        with torch.no_grad():
            t = torch.tensor(x_np).float().to(device)
            return torch.argmax(resnet(t * 2.0 - 1.0), dim=1).cpu().numpy()

    # --- 1. CARICAMENTO DATI ---
    print("-> Lettura e Caricamento Dati...")
    df_untarg = get_best_attack_samples(base_dir / "dataset" / "attacks" / "NN1" / "error_generic" / "cw" / "tracker_cw_untargeted.csv", False)
    df_targ = get_best_attack_samples(base_dir / "dataset" / "attacks" / "NN1" / "error_specific" / "cw" / "next_best" / "tracker_next_best.csv", True)
    
    df_clean_sample = df_untarg[['source_image_path', 'clean_pred_class']].drop_duplicates().sample(n=min(100, len(df_untarg)), random_state=42)

    x_c_orig, y_c_true, _, _ = load_images_from_df(df_clean_sample, base_dir, is_clean=True)
    x_u_adv, y_u_true, _, _ = load_images_from_df(df_untarg, base_dir, is_clean=False)
    x_t_adv, y_t_true, _, y_t_tgt = load_images_from_df(df_targ, base_dir, is_clean=False)

    x_u_orig, _, _, _ = load_images_from_df(df_untarg, base_dir, is_clean=True)
    x_t_orig, _, _, _ = load_images_from_df(df_targ, base_dir, is_clean=True)

    # --- 2. GRID SEARCH 3D DELLA PIPELINE ---
    print("\n[FASE 1] Avvio Grid Search 3D (Smoothing -> Squeezing -> JPEG)...")
    print("         Attenzione: 27 combinazioni in fase di test, ci vorrà un minutino.\n")
    
    windows = [3, 5, 7]
    bits = [4, 5, 6]      # 3 bit è troppo distruttivo, usiamo 4, 5 e 6
    qualities = [50, 70, 90]
    
    results = []
    
    # Progress bar unica per le 27 iterazioni
    pbar = tqdm(total=len(windows)*len(bits)*len(qualities), desc="Grid Search")
    
    for w in windows:
        for b in bits:
            for q in qualities:
                # Inizializzazione difese ART
                def_smooth = SpatialSmoothing(window_size=w, channels_first=True)
                def_squeeze = FeatureSqueezing(clip_values=(0.0, 1.0), apply_predict=True, bit_depth=b)
                def_jpeg = JpegCompression(clip_values=(0.0, 1.0), apply_predict=True, quality=q, channels_first=True)
                
                def apply_pipeline(x_in):
                    x_s1, _ = def_smooth(x_in)
                    x_s2, _ = def_squeeze(x_s1)
                    x_s3, _ = def_jpeg(x_s2)
                    return x_s3
                
                # Valutazione
                acc_c = np.mean(infer(apply_pipeline(x_c_orig)) == y_c_true) * 100
                acc_u = np.mean(infer(apply_pipeline(x_u_adv)) == y_u_true) * 100
                acc_t = np.mean(infer(apply_pipeline(x_t_adv)) == y_t_true) * 100
                
                combo_name = f"Sm{w}-Sq{b}-Jp{q}"
                
                results.append({
                    "combo": combo_name, "w": w, "b": b, "q": q,
                    "acc_c": acc_c, "acc_u": acc_u, "acc_t": acc_t
                })
                pbar.update(1)
                
    pbar.close()
    df_res = pd.DataFrame(results)

    # --- 3. PLOT DELLA GRID SEARCH A 27 COMBINAZIONI ---
    # Usiamo una larghezza elevata per farci stare 27 tuple
    fig, ax = plt.subplots(figsize=(24, 7))
    x = np.arange(len(df_res))
    width = 0.25

    ax.bar(x - width, df_res['acc_c'], width, label='Clean Retention', color='forestgreen', edgecolor='white')
    ax.bar(x, df_res['acc_u'], width, label='Untargeted Recovery', color='dodgerblue', edgecolor='white')
    ax.bar(x + width, df_res['acc_t'], width, label='Targeted Recovery', color='darkorange', edgecolor='white')

    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Pipeline 3D Grid Search: Smoothing $\\rightarrow$ Squeezing $\\rightarrow$ JPEG', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(df_res['combo'], rotation=90, fontsize=10) # Rotazione 90 gradi per farcele stare
    ax.set_ylim(0, 105)
    ax.legend(loc='upper right', bbox_to_anchor=(1.10, 1))

    # Aggiunta etichette sopra le barre (girate a 90 per risparmiare spazio orizzontale)
    for bars in ax.containers:
        ax.bar_label(bars, fmt='%.0f', padding=3, fontsize=8, rotation=90)

    plt.tight_layout()
    plt.savefig(out_dir / "pipeline_triple_grid_search.png", dpi=300)
    plt.close()

    # --- 4. SELEZIONE AUTOMATICA DEL VINCITORE ---
    valid_combos = df_res[df_res['acc_c'] >= 90.0]
    if valid_combos.empty:
        best_row = df_res.loc[df_res['acc_c'].idxmax()]
        print("\n[!] Nessuna combo ha Clean >= 90%. Scelgo quella col Clean maggiore.")
    else:
        valid_combos = valid_combos.copy()
        valid_combos['total_recovery'] = valid_combos['acc_u'] + valid_combos['acc_t']
        best_row = valid_combos.loc[valid_combos['total_recovery'].idxmax()]

    best_w, best_b, best_q = int(best_row['w']), int(best_row['b']), int(best_row['q'])
    print(f"\n[VINCITORE] Configurazione: Smoothing {best_w}x{best_w} -> Squeezing {best_b}-Bit -> JPEG {best_q}")

    # --- 5. ESTRAZIONE DEI 6 CASI STUDIO ---
    print("\n[FASE 2] Estrazione dei 6 Casi Studio per la Relazione...")

    def_smooth_best = SpatialSmoothing(window_size=best_w, channels_first=True)
    def_squeeze_best = FeatureSqueezing(clip_values=(0.0, 1.0), apply_predict=True, bit_depth=best_b)
    def_jpeg_best = JpegCompression(clip_values=(0.0, 1.0), apply_predict=True, quality=best_q, channels_first=True)

    def apply_best_pipeline_steps(x_in):
        x_s1, _ = def_smooth_best(x_in)
        x_s2, _ = def_squeeze_best(x_s1)
        x_s3, _ = def_jpeg_best(x_s2)
        return x_s1, x_s2, x_s3

    # Clean
    pred_c_orig = infer(x_c_orig)
    x_c_s1, x_c_s2, x_c_s3 = apply_best_pipeline_steps(x_c_orig)
    pred_c_s1, pred_c_s2, pred_c_s3 = infer(x_c_s1), infer(x_c_s2), infer(x_c_s3)

    # Untargeted
    pred_u_adv = infer(x_u_adv)
    x_u_s1, x_u_s2, x_u_s3 = apply_best_pipeline_steps(x_u_adv)
    pred_u_s1, pred_u_s2, pred_u_s3 = infer(x_u_s1), infer(x_u_s2), infer(x_u_s3)

    # Targeted
    pred_t_adv = infer(x_t_adv)
    x_t_s1, x_t_s2, x_t_s3 = apply_best_pipeline_steps(x_t_adv)
    pred_t_s1, pred_t_s2, pred_t_s3 = infer(x_t_s1), infer(x_t_s2), infer(x_t_s3)

    # 1. Clean Success
    idx_c_ok = np.where((pred_c_orig == y_c_true) & (pred_c_s3 == y_c_true))[0]
    if len(idx_c_ok) > 0:
        i = idx_c_ok[0]
        plot_progression_clean(x_c_orig[i], x_c_s1[i], x_c_s2[i], x_c_s3[i], 
                               [pred_c_orig[i], pred_c_s1[i], pred_c_s2[i], pred_c_s3[i]], y_c_true[i],
                               "Case 5: Clean Image - Successfully Retained", out_dir / "case_5_clean_success.png")

    # 2. Clean Failure
    idx_c_ko = np.where((pred_c_orig == y_c_true) & (pred_c_s3 != y_c_true))[0]
    if len(idx_c_ko) > 0:
        i = idx_c_ko[0]
        plot_progression_clean(x_c_orig[i], x_c_s1[i], x_c_s2[i], x_c_s3[i], 
                               [pred_c_orig[i], pred_c_s1[i], pred_c_s2[i], pred_c_s3[i]], y_c_true[i],
                               "Case 6: Clean Image - Destroyed by Defense (Collateral Damage)", out_dir / "case_6_clean_failure.png")

    # 3. Untargeted Success (Recuperata)
    idx_u_ok = np.where((pred_u_adv != y_u_true) & (pred_u_s3 == y_u_true))[0]
    if len(idx_u_ok) > 0:
        i = idx_u_ok[0]
        plot_progression_attack(x_u_orig[i], x_u_adv[i], x_u_s1[i], x_u_s2[i], x_u_s3[i],
                                [y_u_true[i], pred_u_adv[i], pred_u_s1[i], pred_u_s2[i], pred_u_s3[i]], y_u_true[i], -1,
                                "Case 3: Untargeted Attack - Successfully Recovered", out_dir / "case_3_untarg_success.png")

    # 4. Untargeted Failure
    idx_u_ko = np.where((pred_u_adv != y_u_true) & (pred_u_s3 != y_u_true))[0]
    if len(idx_u_ko) > 0:
        i = idx_u_ko[0]
        plot_progression_attack(x_u_orig[i], x_u_adv[i], x_u_s1[i], x_u_s2[i], x_u_s3[i],
                                [y_u_true[i], pred_u_adv[i], pred_u_s1[i], pred_u_s2[i], pred_u_s3[i]], y_u_true[i], -1,
                                "Case 4: Untargeted Attack - Defense Failed", out_dir / "case_4_untarg_failure.png")

    # 5. Targeted Success (Recuperata)
    idx_t_ok = np.where((pred_t_adv == y_t_tgt) & (pred_t_s3 == y_t_true))[0]
    if len(idx_t_ok) > 0:
        i = idx_t_ok[0]
        plot_progression_attack(x_t_orig[i], x_t_adv[i], x_t_s1[i], x_t_s2[i], x_t_s3[i],
                                [y_t_true[i], pred_t_adv[i], pred_t_s1[i], pred_t_s2[i], pred_t_s3[i]], y_t_true[i], y_t_tgt[i],
                                "Case 1: Targeted Attack - Successfully Recovered", out_dir / "case_1_targ_success.png")

    # 6. Targeted Failure
    idx_t_ko = np.where((pred_t_adv == y_t_tgt) & (pred_t_s3 == y_t_tgt))[0]
    if len(idx_t_ko) > 0:
        i = idx_t_ko[0]
        plot_progression_attack(x_t_orig[i], x_t_adv[i], x_t_s1[i], x_t_s2[i], x_t_s3[i],
                                [y_t_true[i], pred_t_adv[i], pred_t_s1[i], pred_t_s2[i], pred_t_s3[i]], y_t_true[i], y_t_tgt[i],
                                "Case 2: Targeted Attack - Defense Failed (Still Target)", out_dir / "case_2_targ_failure.png")

    print("\n[FINE] Tutta la logica a 3 stadi è stata processata. Immagini salvate in 'pipeline_showcase'.")

if __name__ == "__main__":
    main()