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
from sklearn.metrics import roc_auc_score

# Impostazioni estetiche per i plot
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

from facenet_pytorch import InceptionResnetV1
from util.google_logger import GoogleSheetLogger

# =========================================================================
# IMPOSTAZIONI GLOBALI
# =========================================================================
DEFENSE_NAME = "AE Anomaly Detector"
BATCH_SIZE = 64
FPR_TOLERANCE = 0.15 # Tolleriamo il 5% di falsi positivi (Clean bloccate)

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
# FUNZIONI DI PLOT E UTILITA'
# =========================================================================
def get_color(pred, clean_label, tgt_label=-1):
    if pred == clean_label: return 'green'
    if pred == tgt_label: return 'firebrick'
    return 'red'

def plot_anomaly_showcase(clean_img, adv_img, clean_mse_map, adv_mse_map, 
                          clean_mse_val, adv_mse_val, threshold, 
                          preds, clean_lbl, tgt_lbl, title, save_path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(title, fontsize=18, fontweight='bold', y=1.02)
    
    # Riga 1: Immagini a confronto
    axes[0, 0].imshow(clean_img)
    axes[0, 0].set_title(f"Original Clean\nPred: ID {preds[0]}", color=get_color(preds[0], clean_lbl, tgt_lbl), fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(adv_img)
    axes[0, 1].set_title(f"Adversarial\nPred: ID {preds[1]}", color=get_color(preds[1], clean_lbl, tgt_lbl), fontweight='bold')
    axes[0, 1].axis('off')
    
    # Riga 2: Mappe MSE (Amplificate per renderle visibili)
    cmap = 'hot'
    c_status = "ACCEPTED" if clean_mse_val <= threshold else "FALSE ALARM"
    axes[1, 0].imshow(np.clip(clean_mse_map * 20.0, 0, 1), cmap=cmap)
    axes[1, 0].set_title(f"Clean MSE Map (Score: {clean_mse_val:.4f})\nDetector: {c_status}", color='green' if c_status=="ACCEPTED" else 'orange')
    axes[1, 0].axis('off')
    
    a_status = "DETECTED & BLOCKED" if adv_mse_val > threshold else "BYPASSED"
    axes[1, 1].imshow(np.clip(adv_mse_map * 20.0, 0, 1), cmap=cmap)
    axes[1, 1].set_title(f"Adv MSE Map (Score: {adv_mse_val:.4f})\nDetector: {a_status}", color='blue' if a_status=="DETECTED & BLOCKED" else 'red')
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def main():
    print("======================================================")
    print(" ANOMALY DETECTION EVALUATION (ZERO-TRUST GATEKEEPER) ")
    print(f" Difesa Attiva: {DEFENSE_NAME}                        ")
    print("======================================================\n")

    base_dir = Path.cwd()
    attacks_base_dir = base_dir / "dataset" / "attacks" / "NN1"
    out_dir = base_dir / "plots" / "6_Defence_Mechanisms" / "anomaly_evaluation_15_perc"
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

    # --- 1. RICERCA E AGGREGAZIONE TRACKER ---
    trackers = list(attacks_base_dir.rglob("tracker_*.csv"))
    print(f"\n-> Trovati {len(trackers)} file tracker. Aggregazione in corso...")
    df_list = [pd.read_csv(t) for t in trackers if not pd.read_csv(t).empty]
    if not df_list: return
    mega_df = pd.concat(df_list, ignore_index=True)
    if 'target_strategy' not in mega_df.columns: mega_df['target_strategy'] = 'none'
    mega_df['target_strategy'] = mega_df['target_strategy'].fillna('none')
    if 'targeted' not in mega_df.columns: mega_df['targeted'] = False

    global_impact_data = [] 
    curves_data = {}
    auroc_data = {}
    mse_records = [] 
    GLOBAL_THRESHOLD = 0.0

    # --- 2. ELABORAZIONE ON-THE-FLY GRUPPO PER GRUPPO ---
    attack_groups = mega_df.groupby(['attack_type', 'targeted', 'target_strategy'])
    
    for (atk_type, is_targeted, strategy), df_group in attack_groups:
        clean_mse_calculated = False 
        df_group = df_group.copy()
        atk_label = f"{str(atk_type).upper()} {'Targeted' if is_targeted else 'Untargeted'} ({strategy})"
        print(f"\n--- Valutazione Tabula-Rasa su: {atk_label} ---")
        
        df_group['eval_clean_pred'] = -1
        df_group['eval_adv_pred'] = -1
        df_group['clean_mse'] = 0.0
        df_group['adv_mse'] = 0.0
        
        # ESECUZIONE DELL'INFERENZA E CALCOLO ANOMALIE
        for start_idx in tqdm(range(0, len(df_group), BATCH_SIZE), desc="Inferenza & Rilevamento in Batch"):
            batch_df = df_group.iloc[start_idx : start_idx + BATCH_SIZE]
            x_adv_batch, x_clean_batch, valid_indices = [], [], []
            
            for original_idx, row in batch_df.iterrows():
                path_adv = str(base_dir / row['adversarial_image_path']).replace('.png', '.tiff').replace('.jpg', '.tiff')
                path_clean = str(base_dir / row['source_image_path']).replace('.png', '.tiff').replace('.jpg', '.tiff')
                img_adv = cv2.imread(path_adv, cv2.IMREAD_UNCHANGED)
                img_clean = cv2.imread(path_clean, cv2.IMREAD_UNCHANGED)
                if img_adv is None or img_clean is None: continue
                
                x_adv_batch.append(np.transpose(cv2.cvtColor(img_adv, cv2.COLOR_BGR2RGB), (2, 0, 1)))
                x_clean_batch.append(np.transpose(cv2.cvtColor(img_clean, cv2.COLOR_BGR2RGB), (2, 0, 1)))
                valid_indices.append(original_idx)
            
            if not x_adv_batch: continue

            x_adv_np = np.stack(x_adv_batch)
            x_clean_np = np.stack(x_clean_batch)
            
            with torch.no_grad():
                t_clean = torch.tensor(x_clean_np).float().to(device)
                t_adv = torch.tensor(x_adv_np).float().to(device)
                
                # --- CALCOLO MSE (ANOMALY SCORE) ---
                rec_adv = autoencoder(t_adv)
                mse_adv_batch = criterion_mse(rec_adv, t_adv).reshape(t_adv.size(0), -1).mean(dim=1).cpu().numpy()
                df_group.loc[valid_indices, 'adv_mse'] = mse_adv_batch
                
                for mse_val in mse_adv_batch:
                    mse_records.append({'Group': atk_label, 'MSE': mse_val})
                
                if not clean_mse_calculated:
                    rec_clean = autoencoder(t_clean)
                    mse_clean_batch = criterion_mse(rec_clean, t_clean).reshape(t_clean.size(0), -1).mean(dim=1).cpu().numpy()
                    df_group.loc[valid_indices, 'clean_mse'] = mse_clean_batch
                    for mse_val in mse_clean_batch:
                        mse_records.append({'Group': 'Immagini Pulite (Clean)', 'MSE': mse_val})

                # --- INFERENZA NN1 SULLE IMMAGINI ORIGINALI (NESSUN DENOISING APPLICATO A FACENET) ---
                preds_clean = torch.argmax(resnet(t_clean * 2.0 - 1.0), dim=1).cpu().numpy()
                preds_adv = torch.argmax(resnet(t_adv * 2.0 - 1.0), dim=1).cpu().numpy()
                
                df_group.loc[valid_indices, 'eval_clean_pred'] = preds_clean
                df_group.loc[valid_indices, 'eval_adv_pred'] = preds_adv

        # CALCOLO DELLA SOGLIA GLOBALE UNA SOLA VOLTA
        if not clean_mse_calculated:
            clean_mses = df_group['clean_mse'].dropna().values
            GLOBAL_THRESHOLD = np.percentile(clean_mses, 100 - (FPR_TOLERANCE * 100))
            print(f"\n   [GATEKEEPER] Soglia di Anomalia Impostata a: {GLOBAL_THRESHOLD:.6f} (FPR: {FPR_TOLERANCE*100:.1f}%)")
            clean_mse_calculated = True

        df_group = df_group[df_group['eval_clean_pred'] != -1].copy()
        if df_group.empty: continue

        # --- 3. CALCOLO EPSILON E METRICHE SYSTEM-LEVEL ---
        is_optimized_attack = 'cw' in str(atk_type).lower() or 'deepfool' in str(atk_type).lower()
        noise_col = 'linf' if 'linf' in df_group.columns else 'eps'
        
        if not is_optimized_attack and 'eps' in df_group.columns:
            epsilons = sorted(df_group['eps'].unique())
        else:
            if is_targeted: succ = df_group[df_group['eval_adv_pred'] == df_group['target_class']]
            else: succ = df_group[df_group['eval_adv_pred'] != df_group['eval_clean_pred']]
            if not succ.empty:
                percentiles = np.linspace(0, 100, 6)
                eps_raw = np.percentile(succ[noise_col], percentiles).tolist()
                epsilons = [0.0] + [round(e, 8) for e in eps_raw] + [round(eps_raw[-1] + 0.001, 8)]
                epsilons = sorted(list(set(epsilons)))
            else:
                epsilons = [0.0, 0.025, 0.05, 0.075, 0.1, 0.15, 0.20]
                
        curves_data[atk_label] = {"epsilons": epsilons, "orig_acc": [], "sys_acc": [], "detected": [], "bypassed_fatal": []}
        auroc_data[atk_label] = {"epsilons": epsilons, "auroc": [], "tpr": []}
        
        for eps in epsilons:
            if not is_optimized_attack:
                df_eval = df_group[df_group['eps'] == eps]
            else:
                within_budget = df_group[noise_col] <= eps
                df_eval = df_group[within_budget].sort_values(noise_col).drop_duplicates('source_image_path', keep='last')
            
            total_attempts = len(df_group) if is_optimized_attack else len(df_eval)
            if total_attempts == 0: continue
            
            # Calcola l'AUC solo se abbiamo un campione statistico minimo (es. > 5)
            # per evitare crolli artificiali a 0.0 a epsilon bassissimi
            if len(df_eval) > 5: 
                y_true_roc = np.concatenate([np.zeros(len(df_eval)), np.ones(len(df_eval))])
                y_scores_roc = np.concatenate([df_eval['clean_mse'].values, df_eval['adv_mse'].values])
                try:
                    auc_score = roc_auc_score(y_true_roc, y_scores_roc)
                except ValueError:
                    auc_score = 0.5
            else:
                auc_score = 0.5 # Default a "tirare a caso" se non ci sono dati sufficienti
            auroc_data[atk_label]["auroc"].append(auc_score * 100)

            # METRICHE ZERO-TRUST (Native Masking per gli ottimizzati)
            if is_optimized_attack:
                out_of_budget = df_group[noise_col] > eps
                valid_mask = within_budget
                df_scope = df_group
            else:
                out_of_budget = pd.Series([False]*len(df_eval), index=df_eval.index)
                valid_mask = pd.Series([True]*len(df_eval), index=df_eval.index)
                df_scope = df_eval

            # Original Undefended Baseline
            orig_resisted = out_of_budget.sum() + (valid_mask & (df_scope['eval_adv_pred'] == df_scope['eval_clean_pred'])).sum()
            orig_rob_acc = orig_resisted / total_attempts

            # System Level Metrics (Detector + Network)
            detected_by_ae = valid_mask & (df_scope['adv_mse'] > GLOBAL_THRESHOLD)
            bypassed = valid_mask & (df_scope['adv_mse'] <= GLOBAL_THRESHOLD)
            
            resisted_by_net = bypassed & (df_scope['eval_adv_pred'] == df_scope['eval_clean_pred'])
            
            # Robustezza Totale (Fuori Budget + Rilevati + Resistiti Dalla Rete)
            sys_resisted = out_of_budget.sum() + detected_by_ae.sum() + resisted_by_net.sum()
            sys_rob_acc = sys_resisted / total_attempts

            if is_targeted:
                bypassed_and_fatal = bypassed & (df_scope['eval_adv_pred'] == df_scope['target_class']) & (~resisted_by_net)
                sys_untarg = 1.0 - sys_rob_acc - (bypassed_and_fatal.sum() / total_attempts)
            else:
                bypassed_and_fatal = bypassed & (df_scope['eval_adv_pred'] != df_scope['eval_clean_pred'])
                sys_untarg = bypassed_and_fatal.sum() / total_attempts

            sys_fatal_asr = bypassed_and_fatal.sum() / total_attempts
            detection_rate = detected_by_ae.sum() / total_attempts

            curves_data[atk_label]["orig_acc"].append(orig_rob_acc * 100)
            curves_data[atk_label]["sys_acc"].append(sys_rob_acc * 100)
            curves_data[atk_label]["detected"].append(detection_rate * 100)
            curves_data[atk_label]["bypassed_fatal"].append(sys_fatal_asr * 100)

            print(f"   Eps: {eps:.3f} | AUROC: {auc_score*100:5.1f}% | Orig Acc: {orig_rob_acc*100:5.1f}% | System Acc: {sys_rob_acc*100:5.1f}% (Detected: {detection_rate*100:.1f}%)")

            if hasattr(logger, 'log_attack_metrics'):
                logger.log_attack_metrics(
                    tester="Leonardo", attack_type=str(atk_type).upper(), strategy=strategy, epsilon=eps,
                    defense_type=DEFENSE_NAME, robust_accuracy=sys_rob_acc, targeted_asr=sys_fatal_asr if is_targeted else 0.0, untargeted_asr=sys_untarg,
                    notes=f"Zero-Trust. AUROC: {auc_score:.2f} | Detected: {detection_rate:.1%} | Thr: {GLOBAL_THRESHOLD:.5f}"
                )

            med_eps = epsilons[len(epsilons)//2]
            if abs(eps - med_eps) < 1e-4 or len(epsilons) == 1:
                if not any(d['Attack'] == atk_label for d in global_impact_data):
                    global_impact_data.append({"Attack": atk_label, "Original Robust Acc": orig_rob_acc * 100, "System Robust Acc": sys_rob_acc * 100})

        # --- 4. ESTRAZIONE ANOMALY SHOWCASE ---
        if 'cw' in str(atk_type).lower() and is_targeted and strategy == 'next_best':
            print(" -> Generazione Anomaly Visual Showcase...")
            idx_bypassed = df_group[(df_group['adv_mse'] <= GLOBAL_THRESHOLD) & (df_group['eval_adv_pred'] == df_group['target_class'])]
            idx_detected = df_group[(df_group['adv_mse'] > GLOBAL_THRESHOLD) & (df_group['eval_adv_pred'] == df_group['target_class'])]
            
            samples_to_plot = []
            if not idx_detected.empty: samples_to_plot.append(("detected", idx_detected.iloc[-1]))
            if not idx_bypassed.empty: samples_to_plot.append(("bypassed", idx_bypassed.iloc[-1]))

            for status, sample in samples_to_plot:
                c_rgb = cv2.cvtColor(cv2.imread(str(base_dir / sample['source_image_path']).replace('.png', '.tiff'), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
                a_rgb = cv2.cvtColor(cv2.imread(str(base_dir / sample['adversarial_image_path']).replace('.png', '.tiff'), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
                
                t_c = torch.tensor(np.expand_dims(np.transpose(c_rgb, (2, 0, 1)), 0)).float().to(device)
                t_a = torch.tensor(np.expand_dims(np.transpose(a_rgb, (2, 0, 1)), 0)).float().to(device)
                
                with torch.no_grad():
                    rec_c = autoencoder(t_c)[0].cpu().numpy()
                    rec_a = autoencoder(t_a)[0].cpu().numpy()
                
                c_map = np.mean(np.abs(np.transpose(rec_c, (1, 2, 0)) - c_rgb), axis=2)
                a_map = np.mean(np.abs(np.transpose(rec_a, (1, 2, 0)) - a_rgb), axis=2)
                
                plot_anomaly_showcase(
                    c_rgb, a_rgb, c_map, a_map, sample['clean_mse'], sample['adv_mse'], GLOBAL_THRESHOLD,
                    [sample['eval_clean_pred'], sample['eval_adv_pred']], sample['eval_clean_pred'], sample['target_class'],
                    f"Zero-Trust Gateway: {status.upper()} Attack", out_dir / f"anomaly_showcase_{status}.png"
                )

    # =========================================================================
    # GENERAZIONE GRAFICI GLOBALI
    # =========================================================================
    print("\n-> Generazione Grafici Globali...")
    
    # 1. KDE Plot Gaussiane
    if mse_records:
        df_mse = pd.DataFrame(mse_records)
        plt.figure(figsize=(12, 7))
        sns.kdeplot(data=df_mse, x='MSE', hue='Group', fill=True, common_norm=False, palette='tab10', alpha=0.5, linewidth=2)
        plt.axvline(GLOBAL_THRESHOLD, color='red', linestyle='--', linewidth=2, label=f'Threshold (FPR={FPR_TOLERANCE*100}%)')
        plt.title('MSE Distribution & Detection Threshold', fontsize=16, fontweight='bold')
        plt.xlabel('Mean Squared Error (MSE)', fontsize=14)
        plt.ylabel('Density', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.6)
        sns.move_legend(plt.gca(), "center left", bbox_to_anchor=(1, 0.5), title='Legenda', frameon=True)
        plt.tight_layout()
        plt.savefig(out_dir / "mse_kde_distributions.png", bbox_inches='tight', dpi=300)
        plt.close()

    # 2. Bar Chart Global Impact
    if global_impact_data:
        df_impact = pd.DataFrame(global_impact_data)
        fig, ax = plt.subplots(figsize=(14, 7))
        x = np.arange(len(df_impact))
        width = 0.35
        ax.bar(x - width/2, df_impact['Original Robust Acc'], width, label='Model WITHOUT Gatekeeper', color='firebrick')
        ax.bar(x + width/2, df_impact['System Robust Acc'], width, label=f'System WITH {DEFENSE_NAME}', color='forestgreen')
        ax.set_ylabel('Robust Accuracy (%)', fontsize=12)
        ax.set_title('Global Defense Impact Analysis (Zero-Trust System)', fontsize=16, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(df_impact['Attack'], rotation=45, ha='right')
        ax.set_ylim(0, 105)
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=2)
        plt.tight_layout()
        plt.savefig(out_dir / "global_impact_barchart.png", dpi=300)
        plt.close()

    # 3. Curve Multiple (AUROC, Security, Stacked)
    for atk_label, data in curves_data.items():
        if len(data['epsilons']) <= 1: continue 
        safe_name = atk_label.replace(" ", "_").replace("/", "").replace("(", "").replace(")", "").lower()
        
        # A) AUROC Curve
        auc_data = auroc_data[atk_label]
        plt.figure(figsize=(10, 6))
        plt.plot(auc_data['epsilons'], auc_data['auroc'], marker='o', color='purple', linewidth=2.5)
        plt.title(f"Detector Discriminative Power (AUROC)\n{atk_label}", fontsize=14, fontweight='bold')
        plt.xlabel(r"Perturbation Budget ($L_\infty$ $\epsilon$)", fontsize=12)
        plt.ylabel("AUROC Score (%)", fontsize=12)
        plt.ylim(45, 105)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(out_dir / f"auroc_curve_{safe_name}.png", dpi=300)
        plt.close()

        # B) Security Curve (System Level)
        plt.figure(figsize=(10, 6))
        plt.plot(data['epsilons'], data['orig_acc'], marker='o', color='red', linestyle='-', linewidth=2, label="Original FaceNet Accuracy")
        plt.plot(data['epsilons'], data['sys_acc'], marker='D', color='green', linestyle='--', linewidth=2.5, label=f"System Accuracy (Gatekeeper + FaceNet)")
        plt.title(f"System Security Evaluation Curve\n{atk_label}", fontsize=14, fontweight='bold')
        plt.xlabel(r"Perturbation Budget ($L_\infty$ $\epsilon$)", fontsize=12)
        plt.ylabel("System Robust Accuracy (%)", fontsize=12)
        plt.ylim(-5, 105)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
        plt.tight_layout()
        plt.savefig(out_dir / f"system_curve_{safe_name}.png", dpi=300)
        plt.close()
        
        # C) Nuova Stacked Bar Chart (Rilevato, Resistito, Fatale)
        plt.figure(figsize=(10, 6))
        detected = np.array(data['detected'])
        resisted_net = np.array(data['sys_acc']) - detected # Quelli che hanno passato il gatekeeper ma FaceNet ha retto
        fatal = np.array(data['bypassed_fatal'])
        
        x_pos = np.arange(len(data['epsilons']))
        b1 = plt.bar(x_pos, detected, color='mediumblue', edgecolor='white', label='Detected & Blocked by AE')
        b2 = plt.bar(x_pos, resisted_net, bottom=detected, color='forestgreen', edgecolor='white', label='Bypassed but Resisted by FaceNet')
        b3 = plt.bar(x_pos, fatal, bottom=detected+resisted_net, color='firebrick', edgecolor='white', label='Bypassed & Fatal (Attack Success)')
        
        plt.xticks(x_pos, [f"{e:.3f}" for e in data['epsilons']])
        plt.title(f"Zero-Trust Architecture Outcome\n{atk_label}", fontsize=16, pad=15)
        plt.xlabel(r"Perturbation Budget ($L_\infty$ $\epsilon$)", fontsize=14)
        plt.ylabel("Percentage of Test Set (%)", fontsize=14)
        plt.ylim(0, 105)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12)
        plt.tight_layout()
        plt.savefig(out_dir / f"zerotrust_outcome_{safe_name}.png", dpi=300)
        plt.close()

    print(f"\n[FINE] Valutazione Anomaly Detector completata. Log inviati e grafici salvati in: {out_dir}")

if __name__ == "__main__":
    main()