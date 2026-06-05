import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from PIL import Image

from facenet_pytorch import InceptionResnetV1, MTCNN

# Importiamo le tue utility
from util.identity_mapper import IdentityMapper
from util.cw_custom import PyTorchCarliniLInf_EarlyStop

# ==========================================
# WRAPPERS PER LA VRAM
# ==========================================
class TopKFacenetWrapper(nn.Module):
    def __init__(self, model, k=10):
        super().__init__()
        self.model = model
        self.k = k
        self.active_indices = None
        
    def freeze_target_classes(self, x):
        with torch.no_grad():
            x_scaled = (x * 2.0) - 1.0
            logits = self.model(x_scaled)
            _, self.active_indices = torch.topk(logits, self.k, dim=1)

    def forward(self, x):
        x_scaled = (x * 2.0) - 1.0
        logits = self.model(x_scaled)
        gathered = torch.gather(logits, 1, self.active_indices)
        return gathered

class FacenetWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x):
        x_scaled = (x * 2.0) - 1.0
        return self.model(x_scaled)

def main():
    print("======================================================")
    print(" SCOUTING HYPER-PARAMS: C&W TARGETED (Pure-GPU)       ")
    print("======================================================\n")

    base_dir = Path.cwd()
    csv_path = base_dir / "dataset" / "clean" / "splits" / "manifest.csv"
    meta_csv_path = base_dir / "dataset" / "clean" / "splits" / "identity_meta.csv"
    output_dir = base_dir / "plots" / "3_Adversarial_Examples" / "error_specific" / "cw"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    txt_log_path = output_dir / "cw_scouting_report.txt"

    BATCH_SIZE = 64
    BUDGET_LINF = 0.10
    SAMPLES_PER_ID = 1  
    
    # Parametri C&W da esplorare
    # Cerchiamo il bilanciamento tra learning_rate (troppo alto distrugge L_inf, troppo basso non converge)
    # e max_iter (troppe iterazioni = troppo lento)
    learning_rates = [0.001, 0.005, 0.01, 0.05, 0.1]
    max_iters_list = [10, 50, 100, 200]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"-> Inizializzazione Reti su {device}...")
    
    mapper = IdentityMapper(meta_csv_path)
    mtcnn = MTCNN(image_size=160, margin=0, keep_all=True, post_process=True, device=device)
    
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    resnet.classify = True 
    
    wrapped_model_full = FacenetWrapper(resnet).eval()
    
    # K=10. Il nostro bersaglio sarà la classe meno probabile *all'interno delle Top-10*.
    # (È un'ottima approssimazione del least-likely per non fondere la VRAM calcolandole tutte e 8631).
    wrapped_model_spliced = TopKFacenetWrapper(resnet, k=8631).eval() 

    df_clean = pd.read_csv(csv_path)

    # --- 1. PRE-FILTRAGGIO E CAMPIONAMENTO ---
    print(f"\n[FASE 1] Estrazione chirurgica e Campionamento ({SAMPLES_PER_ID} img/ID)...")
    valid_x = []
    valid_y = []
    
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
                
                faces = faces.to(device)
                preds_all = torch.argmax(resnet(faces), dim=1).cpu().numpy()
                
                if facenet_id in preds_all:
                    match_idx = int(torch.where(torch.tensor(preds_all) == facenet_id)[0][0])
                    best_face = faces[match_idx] # Range [-1, 1]
                    tensor_img_01 = (best_face + 1.0) / 2.0
                    
                    valid_x.append(tensor_img_01)
                    valid_y.append(torch.tensor([facenet_id], device=device))
                    samples_taken += 1

    if not valid_x:
        print("[ERRORE] Nessun campione valido.")
        return

    x_clean_tensor = torch.stack(valid_x)       # (N, 3, 160, 160)
    y_clean_tensor = torch.cat(valid_y)         # (N,)
    total_samples = len(valid_x)
    print(f"-> Immagini valide raccolte: {total_samples}")

    # ==========================================
    # 2. GRID SEARCH AVVERSARIA A BATCH
    # ==========================================
    print("\n[FASE 2] Avvio C&W Grid Search (Target: Least-Likely Top-10)...\n")
    
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    
    with open(txt_log_path, 'w') as f:
        f.write(f"REPORT SCOUTING CARLINI & WAGNER L_INF\n")
        f.write(f"Campioni testati: {total_samples}\n")
        f.write(f"Budget Massimo L_inf: {BUDGET_LINF}\n\n")

    for lr in learning_rates:
        print(f"\n{'='*50}")
        print(f"Inizio test per LEARNING RATE = {lr}")
        print(f"{'='*50}")
        
        with open(txt_log_path, 'a') as f:
            f.write(f"\n{'='*50}\nInizio test per LEARNING RATE = {lr}\n{'='*50}\n")
            
        for steps in max_iters_list:
            log_str = f"\nGenerazione C&W con max_iter={steps}, lr={lr}..."
            print(log_str)
            
            all_l_infs = []
            all_successes = []
            
            # --- ELABORAZIONE A BATCH ---
            for i in tqdm(range(0, total_samples, BATCH_SIZE), desc=f"Generazione (steps={steps})"):
                x_batch = x_clean_tensor[i : i + BATCH_SIZE]
                y_batch_true = y_clean_tensor[i : i + BATCH_SIZE]
                
                # 1. Congeliamo l'universo delle classi per questo batch (Mantiene le Top-10)
                wrapped_model_spliced.freeze_target_classes(x_batch)
                
                # 2. Selezioniamo il bersaglio più difficile (Least-Likely)
                # Il nostro TopKFacenetWrapper ha shape (Batch, 10).
                # L'indice 0 è la classe vera. L'indice 9 è la decima classe più probabile (least-likely).
                # Creiamo un tensore pieno di "9" lungo quanto il batch.
                local_y_batch = torch.full((x_batch.size(0),), 9, dtype=torch.long, device=device)
                
                # Estraiamo anche l'ID globale del bersaglio (ci serve per verificare il successo sull'intera rete)
                # active_indices ha shape [Batch, 10]. Prendiamo la colonna 9.
                target_global_ids = wrapped_model_spliced.active_indices[:, 9]
                
                # 3. Istanziamo l'attacco
                attack = PyTorchCarliniLInf_EarlyStop(
                    model=wrapped_model_spliced, 
                    targeted=True, 
                    max_iter=steps,         
                    learning_rate=lr,
                    initial_const=1e-3,  
                    largest_const=20.0,
                    early_stop_epsilon=BUDGET_LINF,
                    verbose=False # Mettilo a True se vuoi vedere i log interni "img concluse allo step X"
                )
                
                # 4. Generazione Avversaria in Parallelo
                x_adv_batch = attack.forward(image=x_batch, label=local_y_batch)
                
                # 5. Valutazione Avversaria sulla Rete Intera (Non Wrappata)
                with torch.no_grad():
                    adv_logits = wrapped_model_full(x_adv_batch)
                    adv_preds = torch.argmax(adv_logits, dim=1)
                
                # Calcolo L_inf img per img (vettorizzato)
                diffs = torch.abs(x_adv_batch - x_batch)
                l_infs_batch = torch.amax(diffs, dim=(1, 2, 3))
                
                # Verifichiamo se ha ingannato la rete centrando esattamente il bersaglio globale
                successes_batch = (adv_preds == target_global_ids)
                
                all_l_infs.append(l_infs_batch)
                all_successes.append(successes_batch)
            
            # --- AGGREGAZIONE RISULTATI DELLO STEP ---
            l_infs_tensor = torch.cat(all_l_infs)
            success_tensor = torch.cat(all_successes)
            
            l_infs_np = l_infs_tensor.cpu().numpy()
            success_np = success_tensor.cpu().numpy()
            
            l_min = l_infs_np.min()
            l_mean = l_infs_np.mean()
            l_median = np.median(l_infs_np)
            l_p95 = np.percentile(l_infs_np, 95)
            l_max = l_infs_np.max()
            
            within_budget_mask = l_infs_np <= BUDGET_LINF
            num_within_budget = within_budget_mask.sum()
            
            successful_and_legal_mask = within_budget_mask & success_np
            num_success_legal = successful_and_legal_mask.sum()
            
            targeted_asr = num_success_legal / total_samples
            
            stats_str = (
                f"   Linf stats: min={l_min:.4f}, mean={l_mean:.4f}, median={l_median:.4f}, p95={l_p95:.4f}, max={l_max:.4f}\n"
                f"   Within budget (<= {BUDGET_LINF}): {num_within_budget}/{total_samples} ({(num_within_budget/total_samples)*100:.2f}%)\n"
                f"   Successful attacks within budget: {num_success_legal}/{total_samples} ({(num_success_legal/total_samples)*100:.2f}%)\n"
                f"-> Risultato: Targeted ASR (per eps <= {BUDGET_LINF}) = {targeted_asr*100:.2f}%\n"
            )
            print(stats_str, end="")
            
            with open(txt_log_path, 'a') as f:
                f.write(log_str + "\n")
                f.write(stats_str)
                
            if targeted_asr == 1.0:
                msg = f"   [!] Targeted ASR raggiunto il 100%. Salto max_iter successivi per questo LR per risparmiare tempo.\n"
                print(msg)
                with open(txt_log_path, 'a') as f:
                    f.write(msg)
                break

    print(f"\n[OK] Scouting completato. Report salvato in: {txt_log_path}")

if __name__ == "__main__":
    main()