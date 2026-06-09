import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path
from PIL import Image

import matplotlib.pyplot as plt
import seaborn as sns

# Impostazioni estetiche
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

from facenet_pytorch import InceptionResnetV1, MTCNN
import art.config
# FORZA ART AD USARE FLOAT64
art.config.ART_NUMPY_DTYPE = np.float64
from art.estimators.classification import PyTorchClassifier

# Importiamo la tua classe Clone SOTA e le utility
from util.cw_benchmarks.cw_pytorch import CarliniLInfMethodPyTorch
from util.identity_mapper import IdentityMapper
from util.attack_error_specific_utils import get_one_hot_target

# ==========================================
# WRAPPER (Per Normalizzazione Range e Float64)
# ==========================================
class FacenetWrapper(nn.Module):
    """Usato per la valutazione standard finale."""
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x):
        x_double = x.to(torch.float64)
        return self.model(x_double * 2.0 - 1.0)

class ARTFloat64Wrapper(nn.Module):
    """Intercetta l'input degradato a Float32 da ART e lo riporta a Float64 per ResNet."""
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x):
        return self.model(x.to(torch.float64))

def main():
    print("======================================================")
    print(" SCOUTING HYPER-PARAMS: C&W UNTARGETED (Error-Generic)")
    print("======================================================\n")

    base_dir = Path.cwd()
    csv_path = base_dir / "dataset" / "clean" / "splits" / "manifest.csv"
    meta_csv_path = base_dir / "dataset" / "clean" / "splits" / "identity_meta.csv"
    output_dir = base_dir / "plots" / "3_Adversarial_Examples" / "error_generic" / "cw"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    txt_log_path = output_dir / "cw_untargeted_scouting_report.txt"

    # --- PARAMETRI GRID SEARCH ---
    BUDGET_LINF = 0.10
    SAMPLES_PER_ID = 1  
    BATCH_SIZE = 128 
    
    # Parametri da esplorare (Untargeted converge molto prima)
    learning_rates = [0.005, 0.01, 0.05, 0.1]
    max_iters_list = [5, 10, 25]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"-> Inizializzazione Reti su {device} (Double Precision)...")
    
    mapper = IdentityMapper(meta_csv_path)
    mtcnn = MTCNN(image_size=160, margin=0, keep_all=True, post_process=True, device=device)
    
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    resnet.classify = True 
    resnet.double() # <-- FORZIAMO LA RETE A LAVORARE IN 64-BIT
    
    wrapped_model = FacenetWrapper(resnet).eval()
    art_resnet_shield = ARTFloat64Wrapper(resnet).eval()

    # ART Classifier con 8631 classi
    classifier = PyTorchClassifier(
        model=art_resnet_shield, loss=nn.CrossEntropyLoss(), input_shape=(3, 160, 160), 
        nb_classes=8631, preprocessing=(0.5, 0.5), clip_values=(0.0, 1.0), 
        device_type='gpu' if torch.cuda.is_available() else 'cpu'
    )

    df_clean = pd.read_csv(csv_path)

    # --- FASE 1: PRE-FILTRAGGIO E TARGETING ---
    print(f"\n[FASE 1] Estrazione Dataset ({SAMPLES_PER_ID} img/ID)...")
    valid_x = []
    valid_y_true_raw = []
    valid_y_true_onehot = []
    
    grouped = df_clean.groupby('identity_id')
    
    with torch.no_grad():
        for identity_id, group in tqdm(grouped, desc="Pre-Inferenza"):
            facenet_id = mapper.get_facenet_id_by_class_id(identity_id)
            if facenet_id == -1: continue
            
            samples_taken = 0
            for _, row in group.iterrows():
                if SAMPLES_PER_ID is not None and samples_taken >= SAMPLES_PER_ID:
                    break
                    
                img_path = str(base_dir / row['image_path'])
                try:
                    img_pil = Image.open(img_path).convert('RGB')
                except: continue
                
                faces = mtcnn(img_pil)
                if faces is None: continue
                
                # Attenzione: anche il volto in input deve essere double!
                faces_device = faces.to(device).double()
                preds_all = torch.argmax(resnet(faces_device), dim=1).cpu().numpy()
                
                if facenet_id in preds_all:
                    match_idx = int(np.where(preds_all == facenet_id)[0][0])
                    best_face = faces_device[match_idx] 
                    tensor_img_01 = (best_face + 1.0) / 2.0
                    
                    y_true_onehot = get_one_hot_target(facenet_id, num_classes=8631)
                    
                    valid_x.append(tensor_img_01.cpu().numpy())
                    valid_y_true_raw.append(facenet_id)
                    valid_y_true_onehot.append(y_true_onehot[0])
                    samples_taken += 1

    if not valid_x:
        print("[ERRORE] Nessun campione valido.")
        return

    # Gli array in FLOAT64
    x_clean_np = np.stack(valid_x).astype(np.float64)
    y_true_onehot_np = np.stack(valid_y_true_onehot).astype(np.float64)
    
    # Tensori per la valutazione ultraveloce PyTorch
    x_clean_tensor = torch.tensor(x_clean_np).to(device)
    y_true_tensor = torch.tensor(valid_y_true_raw, dtype=torch.long, device=device)
    
    total_samples = len(valid_x)
    print(f"-> Immagini valide raccolte: {total_samples}")

    # --- FASE 2: GRID SEARCH AVVERSARIA UNTARGETED ---
    print("\n[FASE 2] Avvio C&W Untargeted Grid Search...\n")
    
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    
    with open(txt_log_path, 'w') as f:
        f.write(f"REPORT SCOUTING CARLINI & WAGNER UNTARGETED (CLONE 64-BIT)\n")
        f.write(f"Campioni testati: {total_samples}\n")
        f.write(f"Budget Massimo L_inf: {BUDGET_LINF}\n\n")

    plot_data = {lr: {'iters': [], 'acc': [], 'linf': [], 'early_stopped': False} for lr in learning_rates}

    for lr in learning_rates:
        print(f"\n{'='*50}")
        print(f"Inizio test per LEARNING RATE = {lr}")
        print(f"{'='*50}")
        
        with open(txt_log_path, 'a') as f:
            f.write(f"\n{'='*50}\nInizio test per LEARNING RATE = {lr}\n{'='*50}\n")
            
        for steps in max_iters_list:
            log_str = f"\nGenerazione C&W con max_iter={steps}, lr={lr}..."
            print(log_str)
            
            # 1. Istanziamo la tua classe Ultra-veloce in modalità Untargeted
            attack = CarliniLInfMethodPyTorch(
                classifier=classifier, 
                confidence=0.10,
                targeted=False,          # <--- ATTENZIONE QUI
                max_iter=steps,         
                learning_rate=lr,
                batch_size=BATCH_SIZE,
                verbose=False
            )
            
            start_time = time.time()
            
            # 2. Generazione Batched
            x_adv_np = attack.generate(x=x_clean_np, y=y_true_onehot_np)
            gen_time = time.time() - start_time
            
            x_adv_tensor = torch.tensor(x_adv_np).to(device)
            
            # 3. Valutazione Matematica
            with torch.no_grad():
                adv_logits = wrapped_model(x_adv_tensor)
                adv_preds = torch.argmax(adv_logits, dim=1)
                
                diffs = torch.abs(x_adv_tensor - x_clean_tensor)
                l_infs = torch.amax(diffs, dim=(1, 2, 3)).cpu().numpy()
                
                # SUCCESSO = L'immagine predetta è DIVERSA dall'originale
                success_np = (adv_preds != y_true_tensor).cpu().numpy()
            
            l_min = l_infs.min()
            l_mean = l_infs.mean()
            l_median = np.median(l_infs)
            l_max = l_infs.max()
            
            within_budget_mask = l_infs <= BUDGET_LINF
            num_within_budget = within_budget_mask.sum()
            
            successful_and_legal_mask = within_budget_mask & success_np
            num_success_legal = successful_and_legal_mask.sum()
            
            untargeted_asr = num_success_legal / total_samples
            
            plot_data[lr]['iters'].append(steps)
            plot_data[lr]['acc'].append(untargeted_asr * 100) # Salviamo l'Untargeted ASR (%)
            plot_data[lr]['linf'].append(l_max)
            
            stats_str = (
                f"   Linf stats: min={l_min:.4f}, mean={l_mean:.4f}, median={l_median:.4f}, max={l_max:.4f}\n"
                f"   Within budget (<= {BUDGET_LINF}): {num_within_budget}/{total_samples} ({(num_within_budget/total_samples)*100:.2f}%)\n"
                f"   Untargeted successes within budget: {num_success_legal}/{total_samples} ({(num_success_legal/total_samples)*100:.2f}%)\n"
                f"-> Risultato: Untargeted ASR (per eps <= {BUDGET_LINF}) = {untargeted_asr*100:.2f}%\n"
                f"-> Tempo impiegato: {gen_time:.2f} secondi\n"
            )
            print(stats_str, end="")
            
            with open(txt_log_path, 'a') as f:
                f.write(log_str + "\n")
                f.write(stats_str)
                
            # EARLY STOPPING
            if untargeted_asr == 1.0:
                msg = f"   [!] Untargeted ASR arrivato al 100%. Salto max_iter successivi per questo LR per risparmiare tempo.\n"
                print(msg)
                with open(txt_log_path, 'a') as f:
                    f.write(msg)
                    
                # Ci fermiamo! Senza finti riempimenti della lista
                plot_data[lr]['early_stopped'] = True
                break 

    # --- FASE 3: GENERAZIONE GRAFICO COMPARATIVO ---
    print("\n[FASE 3] Generazione Grafico di Scouting...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.5))
    
    unique_lrs = sorted(list(set(learning_rates)))
    colors = sns.color_palette("tab10", n_colors=len(unique_lrs))
    markers = ['o', 's', '^', 'D', 'v', 'p']
    
    global_handles = []
    global_labels = []
    
    for i, lr in enumerate(unique_lrs):
        iters = plot_data[lr]['iters']
        if not iters: continue 
        
        accs = plot_data[lr]['acc']
        linfs = plot_data[lr]['linf']
        is_stopped = plot_data[lr]['early_stopped']
        
        color = colors[i]
        marker = markers[i % len(markers)]
        
        line1, = ax1.plot(iters, accs, marker=marker, linewidth=2.5, markersize=8, color=color)
        line2, = ax2.plot(iters, linfs, marker=marker, linewidth=2.5, markersize=8, color=color)
        
        global_handles.append(line1)
        global_labels.append(f'LR = {lr}')
        
        if is_stopped:
            last_iter = iters[-1]
            last_acc = accs[-1]
            last_linf = linfs[-1]
            
            # DISEGNO LA "X" PER MODELLO DISTRUTTO
            ax1.scatter(last_iter, last_acc, marker='X', s=350, color=color, edgecolor='black', linewidth=1.5, zorder=5)
            ax2.scatter(last_iter, last_linf, marker='X', s=350, color=color, edgecolor='black', linewidth=1.5, zorder=5)
            ax1.annotate("100% (Destroyed)", (last_iter, last_acc), textcoords="offset points", xytext=(0, -20), ha='center', color=color, fontweight='bold', fontsize=10)

    # Estetica Ax1 (ASR)
    ax1.set_title("Untargeted ASR (Error-Generic) vs. Max Iterations", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Max Iterations (steps)", fontsize=12)
    ax1.set_ylabel("Untargeted ASR (%)", fontsize=12)
    ax1.set_xticks(max_iters_list)
    ax1.set_xticklabels(max_iters_list, rotation=45) 
    ax1.set_ylim(-5, 105)
    
    # Estetica Ax2 (L_inf)
    ax2.set_title(r"Max $L_\infty$ Perturbation vs. Max Iterations", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Max Iterations (steps)", fontsize=12)
    ax2.set_ylabel(r"Max $L_\infty$ Norm", fontsize=12)
    ax2.set_xticks(max_iters_list)
    ax2.set_xticklabels(max_iters_list, rotation=45) 
    
    line_budget = ax2.axhline(y=BUDGET_LINF, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    global_handles.append(line_budget)
    global_labels.append(f"Max Budget ({BUDGET_LINF})")
    
    num_cols = min(6, len(global_labels))
    fig.legend(global_handles, global_labels, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=num_cols, fontsize=11, frameon=True)
    
    plt.suptitle("Carlini & Wagner Untargeted: Hyperparameter Tuning Conf=0.10", fontsize=18, y=1.02)
    plt.tight_layout() 
    
    plot_path = output_dir / "cw_untargeted_tuning_analysis.png"
    plt.savefig(plot_path, bbox_inches='tight', dpi=300)
    print(f"-> Grafico salvato in: {plot_path}")

if __name__ == "__main__":
    main()