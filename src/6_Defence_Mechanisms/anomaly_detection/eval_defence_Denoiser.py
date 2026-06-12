import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from pathlib import Path

# Impostazioni estetiche per i plot
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

from facenet_pytorch import InceptionResnetV1
from util.google_logger import GoogleSheetLogger
from util.plot.utils_plot_shared import plot_adversarial_showcase

# =========================================================================
# IMPOSTAZIONI GLOBALI
# =========================================================================
DEFENSE_NAME = "Autoencoder (Denoiser)"
BATCH_SIZE = 64

# [!] IMPORTANTE: MODIFICA QUESTO PATH CON IL TUO FILE SALVATO DA COLAB [!]
AUTOENCODER_WEIGHTS_PATH = r'src\6_Defence_Mechanisms\anomaly_detection\best_autoencoder_defense.pth' 

# =========================================================================
# CLASSE AUTOENCODER
# =========================================================================
class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid() 
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

# =========================================================================
# FUNZIONI DI UTILITA'
# =========================================================================
def get_color(pred, clean_label, tgt_label=-1):
    if pred == clean_label: return 'green'
    if pred == tgt_label: return 'firebrick'
    return 'red'

def plot_ultimate_showcase(clean_img, adv_img, rec_clean_img, rec_adv_img, preds, clean_lbl, tgt_lbl, title, save_path):
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle(title, fontsize=18, fontweight='bold', y=1.02)
    
    imgs = [clean_img, adv_img, rec_clean_img, rec_adv_img]
    titles = ["1. Original Clean", "2. Adversarial (Bypassed)", "3. Reconstructed Clean", "4. Reconstructed Adv (Defended)"]
    
    for i in range(4):
        axes[0, i].imshow(imgs[i])
        pred_lbl = preds[i]
        color = get_color(pred_lbl, clean_lbl, tgt_lbl)
        axes[0, i].set_title(f"{titles[i]}\nPred: ID {pred_lbl}", color=color, fontweight='bold')
        axes[0, i].axis('off')
        
        if i == 0 or i == 2:
            axes[1, i].axis('off')
        else:
            base_img = clean_img if i == 1 else rec_clean_img
            noise = np.abs(imgs[i] - base_img)
            noise_vis = np.clip(noise * 10.0, 0, 1)
            axes[1, i].imshow(noise_vis)
            axes[1, i].set_title(f"Residual Noise (x10)\nMax Diff: {np.max(noise):.4f}")
            axes[1, i].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def main():
    print("======================================================")
    print(" ULTIMATE DEFENSE EVALUATION (ZERO-TRUST ON-THE-FLY)  ")
    print(f" Difesa Attiva: {DEFENSE_NAME}                        ")
    print("======================================================\n")

    base_dir = Path.cwd()
    attacks_base_dir = base_dir / "dataset" / "attacks" / "NN1"
    out_dir = base_dir / "plots" / "6_Defence_Mechanisms" / "autoencoder_evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"-> Inizializzazione Rete Target NN1 su {device}...")
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    resnet.classify = True 

    print(f"-> Inizializzazione Denoising Autoencoder...")
    autoencoder = DenoisingAutoencoder().to(device)
    try:
        autoencoder.load_state_dict(torch.load(AUTOENCODER_WEIGHTS_PATH, map_location=device))
        autoencoder.eval()
        print("   [OK] Pesi Autoencoder caricati con successo.")
    except Exception as e:
        print(f"   [ERRORE CRITICO] Impossibile caricare i pesi da {AUTOENCODER_WEIGHTS_PATH}.\n   Dettagli: {e}")
        return

    logger = GoogleSheetLogger()
    criterion_mse = nn.MSELoss(reduction='none')

    # --- 1. RICERCA DI TUTTI I TRACKER ---
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
    
    if 'target_strategy' not in mega_df.columns: mega_df['target_strategy'] = 'none'
    mega_df['target_strategy'] = mega_df['target_strategy'].fillna('none')
    if 'targeted' not in mega_df.columns: mega_df['targeted'] = False

    global_impact_data = [] 
    curves_data = {}        

    # Variabili per KDE Plot delle Gaussiane
    mse_records = [] 
    clean_mse_calculated = False 

    # --- 2. ELABORAZIONE ON-THE-FLY GRUPPO PER GRUPPO ---
    attack_groups = mega_df.groupby(['attack_type', 'targeted', 'target_strategy'])
    
    for (atk_type, is_targeted, strategy), df_group in attack_groups:
        df_group = df_group.copy()
        atk_label = f"{str(atk_type).upper()} {'Targeted' if is_targeted else 'Untargeted'} ({strategy})"
        print(f"\n--- Valutazione Tabula-Rasa su: {atk_label} ---")
        
        df_group['eval_clean_pred'] = -1
        df_group['eval_adv_pred'] = -1
        df_group['eval_def_pred'] = -1
        
        # ESECUZIONE DELLA DIFESA IN BATCH
        for start_idx in tqdm(range(0, len(df_group), BATCH_SIZE), desc="Inferenza & Denoising in Batch"):
            batch_df = df_group.iloc[start_idx : start_idx + BATCH_SIZE]
            
            x_adv_batch, x_clean_batch = [], []
            valid_indices = []
            
            for original_idx, row in batch_df.iterrows():
                path_adv = str(base_dir / row['adversarial_image_path']).replace('.png', '.tiff').replace('.jpg', '.tiff')
                path_clean = str(base_dir / row['source_image_path']).replace('.png', '.tiff').replace('.jpg', '.tiff')
                
                img_adv = cv2.imread(path_adv, cv2.IMREAD_UNCHANGED)
                img_clean = cv2.imread(path_clean, cv2.IMREAD_UNCHANGED)
                
                if img_adv is None or img_clean is None: continue
                
                img_adv = cv2.cvtColor(img_adv, cv2.COLOR_BGR2RGB)
                img_clean = cv2.cvtColor(img_clean, cv2.COLOR_BGR2RGB)
                
                x_adv_batch.append(np.transpose(img_adv, (2, 0, 1)))
                x_clean_batch.append(np.transpose(img_clean, (2, 0, 1)))
                valid_indices.append(original_idx)
            
            if not x_adv_batch: continue

            x_adv_np = np.stack(x_adv_batch)
            x_clean_np = np.stack(x_clean_batch)
            
            with torch.no_grad():
                t_clean = torch.tensor(x_clean_np).float().to(device)
                t_adv = torch.tensor(x_adv_np).float().to(device)
                
                # --- RICOSTRUZIONE AUTOENCODER E CALCOLO MSE ---
                rec_adv = autoencoder(t_adv)
                
                # Calcolo MSE Reshape-Safe per le curve gaussiane
                mse_adv_batch = criterion_mse(rec_adv, t_adv).reshape(t_adv.size(0), -1).mean(dim=1).cpu().numpy()
                for mse_val in mse_adv_batch:
                    mse_records.append({'Group': atk_label, 'MSE': mse_val})
                
                if not clean_mse_calculated:
                    rec_clean = autoencoder(t_clean)
                    mse_clean_batch = criterion_mse(rec_clean, t_clean).reshape(t_clean.size(0), -1).mean(dim=1).cpu().numpy()
                    for mse_val in mse_clean_batch:
                        mse_records.append({'Group': 'Immagini Pulite (Clean)', 'MSE': mse_val})

                # --- INFERENZA NN1 MULTIPLA VETTORIZZATA ---
                preds_clean = torch.argmax(resnet(t_clean * 2.0 - 1.0), dim=1).cpu().numpy()
                preds_adv = torch.argmax(resnet(t_adv * 2.0 - 1.0), dim=1).cpu().numpy()
                # Prediciamo l'immagine PURIFICATA dall'Autoencoder
                preds_def = torch.argmax(resnet(rec_adv * 2.0 - 1.0), dim=1).cpu().numpy()
                
                df_group.loc[valid_indices, 'eval_clean_pred'] = preds_clean
                df_group.loc[valid_indices, 'eval_adv_pred'] = preds_adv
                df_group.loc[valid_indices, 'eval_def_pred'] = preds_def

        clean_mse_calculated = True # Evitiamo ricalcoli inutili
        df_group = df_group[df_group['eval_clean_pred'] != -1].copy()
        if df_group.empty: continue

        # --- 3. CALCOLO EPSILON E METRICHE (NATIVE MASKING ALIGNMENT) ---
        is_optimized_attack = 'cw' in str(atk_type).lower() or 'deepfool' in str(atk_type).lower()
        noise_col = 'linf' if 'linf' in df_group.columns else 'eps'
        
        if not is_optimized_attack and 'eps' in df_group.columns:
            epsilons = sorted(df_group['eps'].unique())
        else:
            if is_targeted:
                succ = df_group[df_group['eval_adv_pred'] == df_group['target_class']]
            else:
                succ = df_group[df_group['eval_adv_pred'] != df_group['eval_clean_pred']]
                
            if not succ.empty:
                percentiles = np.linspace(0, 100, 6)
                eps_raw = np.percentile(succ[noise_col], percentiles).tolist()
                epsilons = [0.0] + [round(e, 8) for e in eps_raw] + [round(eps_raw[-1] + 0.001, 8)]
                epsilons = sorted(list(set(epsilons)))
            else:
                epsilons = [0.0, 0.025, 0.05, 0.075, 0.1, 0.15, 0.20]
                
        curves_data[atk_label] = {"epsilons": epsilons, "orig_acc": [], "def_acc": []}
        
        for eps in epsilons:
            if not is_optimized_attack:
                # A: LOGICA STRICT PER-EPSILON
                df_eval = df_group[df_group['eps'] == eps]
                total_attempts = len(df_eval)
                if total_attempts == 0: continue
                
                orig_resisted = (df_eval['eval_adv_pred'] == df_eval['eval_clean_pred']).sum()
                orig_rob_acc = orig_resisted / total_attempts
                
                def_resisted = (df_eval['eval_def_pred'] == df_eval['eval_clean_pred']).sum()
                def_rob_acc = def_resisted / total_attempts
                
                if is_targeted:
                    def_targ_succ = (df_eval['eval_def_pred'] == df_eval['target_class']).sum() / total_attempts
                    def_untarg_succ = 1.0 - def_rob_acc - def_targ_succ
                else:
                    def_targ_succ = 0.0
                    def_untarg_succ = (df_eval['eval_def_pred'] != df_eval['eval_clean_pred']).sum() / total_attempts

            else:
                # B: LOGICA NATIVE MASKING 
                total_attempts = len(df_group)
                if total_attempts == 0: continue
                
                within_budget = df_group[noise_col] <= eps
                out_of_budget = df_group[noise_col] > eps
                
                orig_resisted_attack = within_budget & (df_group['eval_adv_pred'] == df_group['eval_clean_pred'])
                orig_resisted = out_of_budget.sum() + orig_resisted_attack.sum()
                orig_rob_acc = orig_resisted / total_attempts
                
                def_resisted_attack = within_budget & (df_group['eval_def_pred'] == df_group['eval_clean_pred'])
                def_resisted = out_of_budget.sum() + def_resisted_attack.sum()
                def_rob_acc = def_resisted / total_attempts
                
                if is_targeted:
                    def_targ_succ = (within_budget & (df_group['eval_def_pred'] == df_group['target_class']) & (~def_resisted_attack)).sum() / total_attempts
                    def_untarg_succ = 1.0 - def_rob_acc - def_targ_succ
                else:
                    def_targ_succ = 0.0
                    def_untarg_succ = (within_budget & (df_group['eval_def_pred'] != df_group['eval_clean_pred'])).sum() / total_attempts

            print(f"   Eps: {eps:.3f} | Orig Rob_Acc: {orig_rob_acc*100:5.1f}% | Defended Rob_Acc: {def_rob_acc*100:5.1f}%")

            if hasattr(logger, 'log_attack_metrics'):
                logger.log_attack_metrics(
                    tester="Leonardo", 
                    attack_type=str(atk_type).upper(),
                    strategy=strategy,
                    epsilon=eps,
                    defense_type=DEFENSE_NAME,
                    robust_accuracy=def_rob_acc,
                    targeted_asr=def_targ_succ,
                    untargeted_asr=def_untarg_succ,
                    notes=f"Original Rob_Acc: {orig_rob_acc:.1%} (Native Masking Alignment)"
                )
                
            curves_data[atk_label]["orig_acc"].append(orig_rob_acc * 100)
            curves_data[atk_label]["def_acc"].append(def_rob_acc * 100)
            
            med_eps = epsilons[len(epsilons)//2]
            if abs(eps - med_eps) < 1e-4 or len(epsilons) == 1:
                if not any(d['Attack'] == atk_label for d in global_impact_data):
                    global_impact_data.append({
                        "Attack": atk_label,
                        "Original Robust Acc": orig_rob_acc * 100,
                        "Defended Robust Acc": def_rob_acc * 100
                    })

        # --- 4. ESTRAZIONE ULTIMATE SHOWCASE E HEATMAP (Cw Targeted) ---
        if 'cw' in str(atk_type).lower() and is_targeted and strategy == 'next_best':
            print(" -> Generazione Ultimate Visual Showcase e Heatmap Difesa...")
            
            df_pivot = df_group[df_group['linf'] <= 0.10].copy()
            df_pivot['success'] = (df_pivot['eval_def_pred'] == df_pivot['target_class']).astype(int)
            
            top_srcs = df_pivot['identity_name'].unique()[:10] 
            matrix = np.zeros((10, 10))
            for i, src in enumerate(top_srcs):
                for j, enumerate_tgt in enumerate(top_srcs):
                    tgt_df = df_pivot[df_pivot['identity_name'] == enumerate_tgt]
                    if not tgt_df.empty:
                        tgt_id = tgt_df['eval_clean_pred'].iloc[0]
                        att = df_pivot[(df_pivot['identity_name'] == src) & (df_pivot['target_class'] == tgt_id)]
                        if not att.empty:
                            matrix[i, j] = att['success'].mean() * 100
            
            plt.figure(figsize=(10, 8))
            ax = sns.heatmap(matrix, annot=True, fmt=".0f", cmap="Reds", vmin=0, vmax=100)
            ax.set_facecolor('lightgray')
            plt.title(f"Defended Impersonation Matrix ({DEFENSE_NAME})", fontsize=16, pad=20)
            plt.savefig(out_dir / "defended_heatmap_cw_targeted.png", bbox_inches='tight', dpi=300)
            plt.close()

            idx_ok = df_group[(df_group['eval_adv_pred'] == df_group['target_class']) & (df_group['eval_def_pred'] == df_group['eval_clean_pred'])]
            if not idx_ok.empty:
                sample = idx_ok.iloc[-1] 
                
                path_adv = str(base_dir / sample['adversarial_image_path']).replace('.png', '.tiff').replace('.jpg', '.tiff')
                path_clean = str(base_dir / sample['source_image_path']).replace('.png', '.tiff').replace('.jpg', '.tiff')
                
                c_bgr = cv2.imread(path_clean, cv2.IMREAD_UNCHANGED)
                a_bgr = cv2.imread(path_adv, cv2.IMREAD_UNCHANGED)
                c_rgb = cv2.cvtColor(c_bgr, cv2.COLOR_BGR2RGB)
                a_rgb = cv2.cvtColor(a_bgr, cv2.COLOR_BGR2RGB)
                
                # Ricostruzione Autoencoder per il plot visuale
                a_chw = torch.tensor(np.expand_dims(np.transpose(a_rgb, (2, 0, 1)), 0)).float().to(device)
                c_chw = torch.tensor(np.expand_dims(np.transpose(c_rgb, (2, 0, 1)), 0)).float().to(device)
                with torch.no_grad():
                    rec_a = autoencoder(a_chw)[0].cpu().numpy()
                    rec_c = autoencoder(c_chw)[0].cpu().numpy()
                
                plot_ultimate_showcase(
                    c_rgb, a_rgb, np.transpose(rec_c, (1, 2, 0)), np.transpose(rec_a, (1, 2, 0)),
                    [sample['eval_clean_pred'], sample['eval_adv_pred'], -1, sample['eval_def_pred']],
                    sample['eval_clean_pred'], sample['target_class'],
                    f"Ultimate Defense Showcase: Neutralizing C&W Targeted",
                    out_dir / "ultimate_showcase.png"
                )

    # =========================================================================
    # GENERAZIONE GRAFICI GLOBALI E KDE PLOT
    # =========================================================================
    print("\n-> Generazione Grafici Globali...")
    
    # --- 1. GRAFICO DELLE GAUSSIANE (KDE PLOT) ---
    if mse_records:
        df_mse = pd.DataFrame(mse_records)
        plt.figure(figsize=(12, 7))
        sns.kdeplot(data=df_mse, x='MSE', hue='Group', fill=True, common_norm=False, palette='tab10', alpha=0.5, linewidth=2)
        
        plt.title('Distribuzione Errore di Ricostruzione (MSE) per Tipologia di Attacco', fontsize=16, fontweight='bold')
        plt.xlabel('Mean Squared Error (MSE)', fontsize=14)
        plt.ylabel('Densità', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.6)
        
        sns.move_legend(plt.gca(), "center left", bbox_to_anchor=(1, 0.5), title='Legenda', frameon=True)
        plt.tight_layout()
        plt.savefig(out_dir / "mse_kde_distributions.png", bbox_inches='tight', dpi=300)
        plt.close()
        print("   [OK] Grafico KDE Distribuzioni salvato.")

    # --- 2. BAR CHART E CURVE ---
    if global_impact_data:
        df_impact = pd.DataFrame(global_impact_data)
        fig, ax = plt.subplots(figsize=(14, 7))
        x = np.arange(len(df_impact))
        width = 0.35
        
        ax.bar(x - width/2, df_impact['Original Robust Acc'], width, label='Model WITHOUT Defense', color='firebrick')
        ax.bar(x + width/2, df_impact['Defended Robust Acc'], width, label=f'Model WITH {DEFENSE_NAME}', color='forestgreen')
        
        ax.set_ylabel('Robust Accuracy (%)', fontsize=12)
        ax.set_title('Global Defense Impact Analysis (Autoencoder Denoising)', fontsize=16, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(df_impact['Attack'], rotation=45, ha='right')
        ax.set_ylim(0, 105)
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=2)
        plt.tight_layout()
        plt.savefig(out_dir / "global_impact_barchart.png", dpi=300)
        plt.close()

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
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=True)
        plt.tight_layout()
        
        safe_name = atk_label.replace(" ", "_").replace("/", "").replace("(", "").replace(")", "").lower()
        plt.savefig(out_dir / f"defended_curve_{safe_name}.png", bbox_inches='tight', dpi=300)
        plt.close()

    print(f"\n[FINE] Tutta l'Evaluation On-The-Fly è completata. Log inviati e grafici salvati in: {out_dir}")

if __name__ == "__main__":
    main()