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
from art.defences.preprocessor import JpegCompression, SpatialSmoothing
from util.google_logger import GoogleSheetLogger
from util.plot.utils_plot_shared import plot_adversarial_showcase

# =========================================================================
# IMPOSTAZIONI GLOBALI
# =========================================================================
DEFENSE_NAME = "Sm:7+Jp:70"
SMOOTH_WINDOW = 7
JPEG_QUALITY = 70
BATCH_SIZE = 64

def get_color(pred, clean_label, tgt_label=-1):
    if pred == clean_label: return 'green'
    if pred == tgt_label: return 'firebrick'
    return 'red'

def plot_ultimate_showcase(clean_img, adv_img, s1_img, s2_img, preds, clean_lbl, tgt_lbl, title, save_path):
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle(title, fontsize=18, fontweight='bold', y=1.02)
    
    # Immagini in formato nativo per Matplotlib (H, W, C)
    imgs = [clean_img, adv_img, s1_img, s2_img]
    titles = ["1. Original Clean", "2. Adversarial (Bypassed)", f"3. Smoothing {SMOOTH_WINDOW}x{SMOOTH_WINDOW}", f"4. JPEG {JPEG_QUALITY} (Defended)"]
    
    for i in range(4):
        # Riga 1: Immagini
        axes[0, i].imshow(imgs[i])
        pred_lbl = preds[i]
        color = get_color(pred_lbl, clean_lbl, tgt_lbl)
        axes[0, i].set_title(f"{titles[i]}\nPred: ID {pred_lbl}", color=color, fontweight='bold')
        axes[0, i].axis('off')
        
        # Riga 2: Rumore residuo (Amplificato x10)
        if i == 0:
            axes[1, i].axis('off')
        else:
            noise = np.abs(imgs[i] - clean_img)
            noise_vis = np.clip(noise * 10.0, 0, 1)
            axes[1, i].imshow(noise_vis)
            axes[1, i].set_title(f"Residual Noise (x10)\nMax Diff: {np.max(noise):.4f}")
            axes[1, i].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()



def main():
    print("======================================================")
    print(" ULTIMATE DEFENSE EVALUATION & LOGGING                ")
    print(f" Difesa Attiva: {DEFENSE_NAME}                        ")
    print("======================================================\n")

    base_dir = Path.cwd()
    attacks_base_dir = base_dir / "dataset" / "attacks" / "NN1"
    out_dir = base_dir / "plots" / "6_Defence_Mechanisms" / "final_evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"-> Inizializzazione Rete NN1 su {device}...")
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    resnet.classify = True 

    # Inizializzazione Difese ART
    print(f"-> Inizializzazione Pipeline: Smoothing {SMOOTH_WINDOW}x{SMOOTH_WINDOW} -> JPEG {JPEG_QUALITY}")
    def_smooth = SpatialSmoothing(window_size=SMOOTH_WINDOW, channels_first=True)
    def_jpeg = JpegCompression(clip_values=(0.0, 1.0), apply_predict=True, quality=JPEG_QUALITY, channels_first=True)

    logger = GoogleSheetLogger()

    # --- 1. RICERCA DI TUTTI I TRACKER E AGGREGAZIONE ---
    trackers = list(attacks_base_dir.rglob("tracker_*.csv"))
    print(f"\n-> Trovati {len(trackers)} file tracker. Aggregazione in corso...")
    
    df_list = []
    for t in trackers:
        tmp = pd.read_csv(t)
        if not tmp.empty: df_list.append(tmp)
        
    if not df_list:
        print("[ERRORE] Nessun dato trovato nei file CSV.")
        return
        
    mega_df = pd.concat(df_list, ignore_index=True)
    
    # Pulizia colonne per raggruppamento sicuro
    if 'target_strategy' not in mega_df.columns: mega_df['target_strategy'] = 'none'
    mega_df['target_strategy'] = mega_df['target_strategy'].fillna('none')
    if 'targeted' not in mega_df.columns: mega_df['targeted'] = False

    global_impact_data = [] 
    curves_data = {}        

    # --- 2. ELABORAZIONE GRUPPO PER GRUPPO (es. "BIM Untargeted", "CW Targeted") ---
    attack_groups = mega_df.groupby(['attack_type', 'targeted', 'target_strategy'])
    
    for (atk_type, is_targeted, strategy), df_group in attack_groups:
        df_group = df_group.copy()
        atk_label = f"{str(atk_type).upper()} {'Targeted' if is_targeted else 'Untargeted'} ({strategy})"
        
        print(f"\n--- Valutazione Difesa su: {atk_label} ---")
        
        # Controlliamo se mancano le predizioni originali (es. vecchi CSV)
        needs_orig_infer = ('clean_pred_class' not in df_group.columns) or ('adv_pred_class' not in df_group.columns)
        if needs_orig_infer:
            df_group['clean_pred_class'] = -1
            df_group['adv_pred_class'] = -1
            
        df_group['defended_pred_class'] = -1
        
        # --- ESECUZIONE DELLA DIFESA IN BATCH ---
        for start_idx in tqdm(range(0, len(df_group), BATCH_SIZE), desc="Applicazione Difesa in Batch"):
            batch_df = df_group.iloc[start_idx : start_idx + BATCH_SIZE]
            
            x_adv_batch, x_clean_batch = [], []
            valid_indices = [] # <-- Novità: tracciamo gli indici sicuri
            
            for original_idx, row in batch_df.iterrows():
                path_adv = str(base_dir / row['adversarial_image_path'])
                img_adv = cv2.imread(path_adv, cv2.IMREAD_UNCHANGED)
                
                if img_adv is None: 
                    print(f"\n[WARNING] Immagine Adv corrotta o mancante: {path_adv}")
                    continue
                
                # Gestione immagine pulita (se necessaria)
                if needs_orig_infer:
                    path_clean = str(base_dir / row['source_image_path'])
                    img_clean = cv2.imread(path_clean, cv2.IMREAD_UNCHANGED)
                    if img_clean is None:
                        print(f"\n[WARNING] Immagine Clean corrotta o mancante: {path_clean}")
                        continue
                    
                    img_clean = cv2.cvtColor(img_clean, cv2.COLOR_BGR2RGB)
                    if img_clean.dtype == np.uint8: img_clean = img_clean.astype(np.float32) / 255.0
                    x_clean_batch.append(img_clean)

                img_adv = cv2.cvtColor(img_adv, cv2.COLOR_BGR2RGB)
                if img_adv.dtype == np.uint8: img_adv = img_adv.astype(np.float32) / 255.0
                x_adv_batch.append(img_adv)
                
                # Se l'immagine era buona, salviamo il suo indice originale!
                valid_indices.append(original_idx)
            
            # Se il batch era tutto corrotto, andiamo avanti
            if not x_adv_batch:
                continue

            x_adv_np = np.transpose(np.stack(x_adv_batch), (0, 3, 1, 2))
            
            x_s1, _ = def_smooth(x_adv_np)
            x_defended, _ = def_jpeg(x_s1)
            
            with torch.no_grad():
                t_def = torch.tensor(x_defended).float().to(device)
                preds_def = torch.argmax(resnet(t_def * 2.0 - 1.0), dim=1).cpu().numpy()
                
                # Assegniamo i risultati SOLO agli indici validi!
                df_group.loc[valid_indices, 'defended_pred_class'] = preds_def
                
                if needs_orig_infer:
                    t_adv = torch.tensor(x_adv_np).float().to(device)
                    t_clean = torch.tensor(np.transpose(np.stack(x_clean_batch), (0, 3, 1, 2))).float().to(device)
                    df_group.loc[valid_indices, 'adv_pred_class'] = torch.argmax(resnet(t_adv * 2.0 - 1.0), dim=1).cpu().numpy()
                    df_group.loc[valid_indices, 'clean_pred_class'] = torch.argmax(resnet(t_clean * 2.0 - 1.0), dim=1).cpu().numpy()

        # --- CALCOLO EPSILON DINAMICO O REALE ---
        is_optimized_attack = 'cw' in str(atk_type).lower() or 'deepfool' in str(atk_type).lower()
        
        if not is_optimized_attack and 'eps' in df_group.columns:
            epsilons = sorted(df_group['eps'].unique())
        else:
            noise_col = 'linf' if 'linf' in df_group.columns else 'eps'
            if is_targeted:
                succ = df_group[df_group['adv_pred_class'] == df_group['target_class']]
            else:
                succ = df_group[df_group['adv_pred_class'] != df_group['clean_pred_class']]
                
            if not succ.empty:
                percentiles = np.linspace(0, 100, 6)
                eps_raw = np.percentile(succ[noise_col], percentiles).tolist()
                epsilons = [0.0] + [round(e, 8) for e in eps_raw] + [round(eps_raw[-1] + 0.001, 8)]
                epsilons = sorted(list(set(epsilons)))
            else:
                epsilons = [0.0, 0.025, 0.05, 0.075, 0.1, 0.15, 0.20]
                
        curves_data[atk_label] = {"epsilons": epsilons, "orig_acc": [], "def_acc": []}
        
        # FIX: Calcoliamo i "tentativi totali" veri (Essenziale per il Round-Robin)
        if 'target_class' in df_group.columns:
            total_attempts = len(df_group[['source_image_path', 'target_class']].drop_duplicates())
        else:
            total_attempts = df_group['source_image_path'].nunique()
        
        # --- CALCOLO METRICHE ---
        for eps in epsilons:
            if not is_optimized_attack:
                closest_attacks = df_group[df_group['eps'] == eps]
            else:
                valid_attacks = df_group[df_group[noise_col] <= eps]
                if valid_attacks.empty:
                    closest_attacks = pd.DataFrame()
                else:
                    closest_attacks = valid_attacks.sort_values(noise_col).drop_duplicates('source_image_path', keep='last')
            
            missing_count = total_attempts - len(closest_attacks)
            
            if not closest_attacks.empty:
                orig_res = (closest_attacks['adv_pred_class'] == closest_attacks['clean_pred_class']).sum()
                def_res = (closest_attacks['defended_pred_class'] == closest_attacks['clean_pred_class']).sum()
            else:
                orig_res, def_res = 0, 0
                
            orig_rob_acc = (orig_res + missing_count) / total_attempts
            def_rob_acc = (def_res + missing_count) / total_attempts
            
            if is_targeted and not closest_attacks.empty:
                def_targ_succ = (closest_attacks['defended_pred_class'] == closest_attacks['target_class']).sum() / total_attempts
                def_untarg_succ = 1.0 - def_rob_acc - def_targ_succ
            elif not is_targeted and not closest_attacks.empty:
                def_targ_succ = 0.0
                def_untarg_succ = (closest_attacks['defended_pred_class'] != closest_attacks['clean_pred_class']).sum() / total_attempts
            else:
                def_targ_succ, def_untarg_succ = 0.0, 0.0
        
        # --- CALCOLO TENTATIVI TOTALI (Supporto Round-Robin) ---
        if is_targeted and 'target_class' in df_group.columns:
            total_attempts = len(df_group[['source_image_path', 'target_class']].drop_duplicates())
            dup_subset = ['source_image_path', 'target_class'] # Distinguiamo gli attacchi RR!
        else:
            total_attempts = df_group['source_image_path'].nunique()
            dup_subset = ['source_image_path']
        
        # --- CALCOLO METRICHE ---
        for eps in epsilons:
            if not is_optimized_attack:
                closest_attacks = df_group[df_group['eps'] == eps]
            else:
                valid_attacks = df_group[df_group[noise_col] <= eps]
                if valid_attacks.empty:
                    closest_attacks = pd.DataFrame()
                else:
                    closest_attacks = valid_attacks.sort_values(noise_col).drop_duplicates(dup_subset, keep='last')
            
            missing_count = total_attempts - len(closest_attacks)
            
            if not closest_attacks.empty:
                orig_res = (closest_attacks['adv_pred_class'] == closest_attacks['clean_pred_class']).sum()
                def_res = (closest_attacks['defended_pred_class'] == closest_attacks['clean_pred_class']).sum()
            else:
                orig_res, def_res = 0, 0
                
            orig_rob_acc = (orig_res + missing_count) / total_attempts
            def_rob_acc = (def_res + missing_count) / total_attempts
            
            if is_targeted and not closest_attacks.empty:
                def_targ_succ = (closest_attacks['defended_pred_class'] == closest_attacks['target_class']).sum() / total_attempts
                def_untarg_succ = 1.0 - def_rob_acc - def_targ_succ
            elif not is_targeted and not closest_attacks.empty:
                def_targ_succ = 0.0
                def_untarg_succ = (closest_attacks['defended_pred_class'] != closest_attacks['clean_pred_class']).sum() / total_attempts
            else:
                def_targ_succ, def_untarg_succ = 0.0, 0.0

            if hasattr(logger, 'log_attack_metrics'):
                logger.log_attack_metrics(
                    tester="Leonardo", 
                    attack_type=atk_type.upper(),
                    strategy=strategy,
                    epsilon=eps,
                    defense_type=DEFENSE_NAME,
                    robust_accuracy=def_rob_acc,
                    targeted_asr=def_targ_succ,
                    untargeted_asr=def_untarg_succ,
                    notes=f"Original Rob_Acc: {orig_rob_acc:.1%}"
                )
                
            curves_data[atk_label]["orig_acc"].append(orig_rob_acc * 100)
            curves_data[atk_label]["def_acc"].append(def_rob_acc * 100)
            
            # Scelta dell'epsilon mediano per il Global Impact Chart
            med_eps = epsilons[len(epsilons)//2]
            if abs(eps - med_eps) < 1e-4 or len(epsilons) == 1:
                if not any(d['Attack'] == atk_label for d in global_impact_data):
                    global_impact_data.append({
                        "Attack": atk_label,
                        "Original Robust Acc": orig_rob_acc * 100,
                        "Defended Robust Acc": def_rob_acc * 100
                    })

        # --- ESTRAZIONE ULTIMATE SHOWCASE (Solo C&W Targeted) ---
        if 'cw' in str(atk_type).lower() and is_targeted and strategy == 'next_best':
            print(" -> Generazione Ultimate Visual Showcase e Heatmap Difesa...")
            
            df_pivot = df_group[df_group['linf'] <= 0.10].copy()
            df_pivot = df_pivot.sort_values('linf').drop_duplicates('source_image_path', keep='last')
            
            if not df_pivot.empty:
                df_pivot['success'] = (df_pivot['defended_pred_class'] == df_pivot['target_class']).astype(int)
                
                top_srcs = df_pivot['identity_name'].unique()[:10] 
                matrix = np.zeros((10, 10))
                for i, src in enumerate(top_srcs):
                    for j, tgt in enumerate(top_srcs):
                        tgt_df = df_pivot[df_pivot['identity_name'] == tgt]
                        if not tgt_df.empty:
                            tgt_id = tgt_df['clean_pred_class'].iloc[0]
                            att = df_pivot[(df_pivot['identity_name'] == src) & (df_pivot['target_class'] == tgt_id)]
                            if not att.empty:
                                matrix[i, j] = att['success'].mean() * 100
                
                plt.figure(figsize=(10, 8))
                ax = sns.heatmap(matrix, annot=True, fmt=".0f", cmap="Reds", vmin=0, vmax=100)
                ax.set_facecolor('lightgray')
                plt.title(f"Defended Impersonation Matrix ({DEFENSE_NAME})", fontsize=16, pad=20)
                plt.savefig(out_dir / "defended_heatmap_cw_targeted.png", bbox_inches='tight', dpi=300)
                plt.close()

                idx_ok = df_group[(df_group['adv_pred_class'] == df_group['target_class']) & (df_group['defended_pred_class'] == df_group['clean_pred_class'])]
                if not idx_ok.empty:
                    sample = idx_ok.iloc[-1] 
                    
                    c_bgr = cv2.imread(str(base_dir / sample['source_image_path']), cv2.IMREAD_UNCHANGED)
                    a_bgr = cv2.imread(str(base_dir / sample['adversarial_image_path']), cv2.IMREAD_UNCHANGED)
                    c_rgb = cv2.cvtColor(c_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0 if c_bgr.dtype == np.uint8 else cv2.cvtColor(c_bgr, cv2.COLOR_BGR2RGB)
                    a_rgb = cv2.cvtColor(a_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0 if a_bgr.dtype == np.uint8 else cv2.cvtColor(a_bgr, cv2.COLOR_BGR2RGB)
                    
                    a_chw = np.expand_dims(np.transpose(a_rgb, (2, 0, 1)), 0)
                    s1_chw, _ = def_smooth(a_chw)
                    s2_chw, _ = def_jpeg(s1_chw)
                    
                    plot_ultimate_showcase(
                        c_rgb, a_rgb, np.transpose(s1_chw[0], (1, 2, 0)), np.transpose(s2_chw[0], (1, 2, 0)),
                        [sample['clean_pred_class'], sample['adv_pred_class'], -1, sample['defended_pred_class']],
                        sample['clean_pred_class'], sample['target_class'],
                        f"Ultimate Defense Showcase: Neutralizing C&W Targeted",
                        out_dir / "ultimate_showcase.png"
                    )

    # =========================================================================
    # GENERAZIONE GRAFICI GLOBALI
    # =========================================================================
    print("\n-> Generazione Grafici Globali...")
    
    # 1. GLOBAL IMPACT BAR CHART
    if global_impact_data:
        df_impact = pd.DataFrame(global_impact_data)
        fig, ax = plt.subplots(figsize=(14, 7))
        x = np.arange(len(df_impact))
        width = 0.35
        
        ax.bar(x - width/2, df_impact['Original Robust Acc'], width, label='Model WITHOUT Defense', color='firebrick')
        ax.bar(x + width/2, df_impact['Defended Robust Acc'], width, label=f'Model WITH {DEFENSE_NAME}', color='forestgreen')
        
        ax.set_ylabel('Robust Accuracy (%)', fontsize=12)
        ax.set_title('Global Defense Impact Analysis', fontsize=16, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(df_impact['Attack'], rotation=45, ha='right')
        ax.set_ylim(0, 105)
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=2)
        plt.tight_layout()
        plt.savefig(out_dir / "global_impact_barchart.png", dpi=300)
        plt.close()

    # 2. DEFENDED SECURITY CURVES
    for atk_label, data in curves_data.items():
        if len(data['epsilons']) <= 1: continue 
        
        plt.figure(figsize=(10, 6))
        plt.plot(data['epsilons'], data['orig_acc'], marker='o', color='red', linestyle='-', linewidth=2, label="Original Accuracy (Undefended)")
        plt.plot(data['epsilons'], data['def_acc'], marker='D', color='green', linestyle='--', linewidth=2.5, label=f"Defended Accuracy ({DEFENSE_NAME})")
        
        plt.title(f"Defended Security Evaluation Curve\n{atk_label}", fontsize=14, fontweight='bold')
        plt.xlabel(r"Perturbation Budget ($L_\infty$ $\epsilon$)", fontsize=12)
        plt.ylabel("Robust Accuracy (%)", fontsize=12)
        plt.ylim(-5, 105)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # --- FIX: Legenda spostata in basso fuori dal grafico ---
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=True)
        
        plt.tight_layout()
        
        safe_name = atk_label.replace(" ", "_").replace("/", "").replace("(", "").replace(")", "").lower()
        # Aggiunto bbox_inches='tight' per non far tagliare la legenda salvata
        plt.savefig(out_dir / f"defended_curve_{safe_name}.png", bbox_inches='tight', dpi=300)
        plt.close()


    print(f"\n[FINE] Tutta l'Evaluation è stata completata e i log sono stati inviati a Google Sheets.")
    print(f"I grafici finali per il report si trovano in: {out_dir}")

if __name__ == "__main__":
    main()