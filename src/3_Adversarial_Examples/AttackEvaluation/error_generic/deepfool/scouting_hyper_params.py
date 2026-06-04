import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path
from PIL import Image

from facenet_pytorch import InceptionResnetV1, MTCNN
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import DeepFool

from util.identity_mapper import IdentityMapper

def main():
    print("======================================================")
    print(" SCOUTING HYPER-PARAMS: DEEPFOOL (Error-Generic)      ")
    print("======================================================\n")

    # ==========================================
    # 1. SETUP PATH E PARAMETRI
    # ==========================================
    base_dir = Path.cwd()
    csv_path = base_dir / "dataset" / "clean" / "splits" / "manifest.csv"
    meta_csv_path = base_dir / "dataset" / "clean" / "splits" / "identity_meta.csv"
    output_dir = base_dir / "plots" / "3_Adversarial_Examples" / "error_generic" / "deepfool"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    txt_log_path = output_dir / "deepfool_scouting_report.txt"

    # Parametri della Grid Search
    BUDGET_LINF = 0.10
    SAMPLES_PER_ID = 1  # Quante immagini per identità valutare (None per farle tutte)
    
    # Parametri DeepFool da esplorare
    overshoots = [0.01, 0.1, 1.0, 2.5, 3.0] # L'epsilon di DeepFool (la spinta oltre il confine)
    nb_grads_list = [3, 5]          # Quante classi valutare per trovare il confine più vicino
    max_iters = [100, 150, 200, 350]             # Timeout di sicurezza (come concordato guardando BIM)
    
    # ==========================================
    # 2. INIZIALIZZAZIONE RETI
    # ==========================================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"-> Inizializzazione Reti su {device}...")
    
    mapper = IdentityMapper(meta_csv_path)
    mtcnn = MTCNN(image_size=160, margin=0, keep_all=True, post_process=True, device=device)
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    resnet.classify = True 

    classifier = PyTorchClassifier(
        model=resnet, clip_values=(0.0, 1.0), loss=nn.CrossEntropyLoss(), optimizer=None,
        input_shape=(3, 160, 160), nb_classes=8631, preprocessing=(0.5, 0.5), 
        device_type='gpu' if torch.cuda.is_available() else 'cpu'
    )

    df_clean = pd.read_csv(csv_path)

    # ==========================================
    # 3. PRE-FILTRAGGIO E CAMPIONAMENTO
    # ==========================================
    print(f"\n[FASE 1] Estrazione chirurgica e Campionamento ({SAMPLES_PER_ID} img/ID)...")
    valid_x = []
    valid_y = []
    
    # Raggruppiamo per ID per fare il campionamento controllato
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
                    match_idx = np.where(preds_all == facenet_id)[0][0]
                    best_face = faces[match_idx]
                    
                    np_img_01 = (best_face.cpu().numpy() + 1.0) / 2.0
                    valid_x.append(np_img_01)
                    valid_y.append(facenet_id)
                    samples_taken += 1

    x_clean_arr = np.array(valid_x) # Shape: (N, 3, 160, 160)
    y_clean_arr = np.array(valid_y)
    total_samples = len(valid_x)
    print(f"-> Immagini valide raccolte: {total_samples}")

    if total_samples == 0:
        print("[ERRORE] Nessun campione valido.")
        return

    # ==========================================
    # 4. GRID SEARCH AVVERSARIA
    # ==========================================
    print("\n[FASE 2] Avvio DeepFool Grid Search...\n")
    
    # Inizializziamo il file di log pulito
    with open(txt_log_path, 'w') as f:
        f.write(f"REPORT SCOUTING DEEPFOOL\n")
        f.write(f"Campioni testati: {total_samples}\n")
        f.write(f"Budget Massimo L_inf: {BUDGET_LINF}\n\n")

    # Abbassiamo il batch size: DeepFool richiede molta memoria computazionale
    GEN_BATCH_SIZE = 32 
    for max_iter in max_iters:
        print(f"Inizio test per max_iter = {max_iter}")
        for ov_shoot in overshoots:
            print(f"\n{'='*50}")
            print(f"Inizio test per OVERSHOOT (epsilon) = {ov_shoot}")
            print(f"{'='*50}")
            
                
            for nb_g in nb_grads_list:
                with open(txt_log_path, 'a') as f:
                    f.write(f"\n{'='*50}\nInizio test con max_iter={max_iter}, overshoot={ov_shoot}, nb_grads={nb_g}\n{'='*50}\n")

                log_str = f"\nGenerazione DeepFool con max_iter={max_iter}, overshoot={ov_shoot}, nb_grads={nb_g}..."
                print(log_str)
                
                attack = DeepFool(
                    classifier=classifier, 
                    max_iter=max_iter, 
                    epsilon=ov_shoot, 
                    nb_grads=nb_g, 
                    batch_size=GEN_BATCH_SIZE
                )
                
                # --- MODIFICA CHIAVE: Batching Manuale con TQDM ---
                x_adv_list = []
                
                # Iteriamo sull'array pulito a blocchi per salvare la RAM e vedere la progress bar
                for i in tqdm(range(0, len(x_clean_arr), GEN_BATCH_SIZE), desc=f"Generazione (nb_grads={nb_g})"):
                    x_batch = x_clean_arr[i : i + GEN_BATCH_SIZE]
                    x_adv_batch = attack.generate(x=x_batch)
                    x_adv_list.append(x_adv_batch)
                    
                x_adv_arr = np.concatenate(x_adv_list, axis=0)
                # ---------------------------------------------------
                
                # Da qui in poi rimane identico
                adv_preds_raw = classifier.predict(x_adv_arr)
                adv_preds = np.argmax(adv_preds_raw, axis=1)
                
                # Calcolo L_inf img per img
                # Assi: (N, C, H, W). Calcoliamo la max differenza per ogni N
                diffs = np.abs(x_adv_arr - x_clean_arr)
                l_infs = np.max(diffs, axis=(1, 2, 3))
                
                # Statistiche L_inf
                l_min = np.min(l_infs)
                l_mean = np.mean(l_infs)
                l_median = np.median(l_infs)
                l_p95 = np.percentile(l_infs, 95)
                l_max = np.max(l_infs)
                
                # Quanti rispettano il budget?
                within_budget_mask = l_infs <= BUDGET_LINF
                num_within_budget = np.sum(within_budget_mask)
                
                # Tra quelli che rispettano il budget, quanti hanno ingannato la rete? (Untargeted: adv_pred != clean_pred)
                successful_and_legal_mask = within_budget_mask & (adv_preds != y_clean_arr)
                num_success_legal = np.sum(successful_and_legal_mask)
                
                # La Robust Accuracy è: Quante immagini HANNO RESISTITO oppure L'ATTACCO È ILLEGALE
                # Quindi: total - (successi legali)
                robust_accuracy = (total_samples - num_success_legal) / total_samples
                
                # Formattazione Output
                stats_str = (
                    f"   Linf stats: min={l_min:.4f}, mean={l_mean:.4f}, median={l_median:.4f}, p95={l_p95:.4f}, max={l_max:.4f}\n"
                    f"   Within budget (<= {BUDGET_LINF}): {num_within_budget}/{total_samples} ({(num_within_budget/total_samples)*100:.2f}%)\n"
                    f"   Successful attacks within budget: {num_success_legal}/{total_samples} ({(num_success_legal/total_samples)*100:.2f}%)\n"
                    f"-> Risultato: Robust Accuracy (per eps <= {BUDGET_LINF}) = {robust_accuracy*100:.2f}%\n"
                )
                print(stats_str, end="")
                
                with open(txt_log_path, 'a') as f:
                    f.write(log_str + "\n")
                    f.write(stats_str)
                    
                # --- EARLY STOPPING LOGIC ---
                if robust_accuracy == 0.0:
                    msg = f"   [!] Accuracy crollata a 0.00%. Salto nb_grads successivi per questo overshoot.\n"
                    print(msg)
                    with open(txt_log_path, 'a') as f:
                        f.write(msg)
                    break # Interrompe il loop sui nb_grads, passa al prossimo overshoot

    print(f"\n[OK] Scouting completato. Report salvato in: {txt_log_path}")

if __name__ == "__main__":
    main()