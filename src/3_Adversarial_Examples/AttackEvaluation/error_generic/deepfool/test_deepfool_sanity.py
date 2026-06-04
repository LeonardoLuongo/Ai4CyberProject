import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

import torchattacks
from facenet_pytorch import InceptionResnetV1, MTCNN
from util.identity_mapper import IdentityMapper

# Aggiungiamo l'import del tuo plot per l'ispezione visiva
from util.plot.utils_plot_shared import plot_adversarial_showcase

# ==========================================
# I WRAPPER ESATTI DEL TUO SCRIPT
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
    print(" SANITY CHECK: DEEPFOOL + TOP-K WRAPPER               ")
    print("======================================================\n")

    base_dir = Path.cwd()
    csv_path = base_dir / "dataset" / "clean" / "splits" / "manifest.csv"
    meta_csv_path = base_dir / "dataset" / "clean" / "splits" / "identity_meta.csv"
    
    out_dir = base_dir / "plots" / "debug"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"-> Inizializzazione Reti su {device}...")
    
    mapper = IdentityMapper(meta_csv_path)
    mtcnn = MTCNN(image_size=160, margin=0, keep_all=True, post_process=True, device=device)
    
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    resnet.classify = True 
    
    wrapped_model_full = FacenetWrapper(resnet).eval()
    wrapped_model_spliced = TopKFacenetWrapper(resnet, k=10).eval()

    df_clean = pd.read_csv(csv_path)

    # --- 1. TROVIAMO LA PRIMA IMMAGINE VALIDA ---
    print("\n-> Ricerca di un'immagine campione valida...")
    sample_img_01 = None
    true_facenet_id = None
    identity_name = ""

    with torch.no_grad():
        for _, row in df_clean.iterrows():
            facenet_id = mapper.get_facenet_id_by_class_id(str(row['identity_id']))
            if facenet_id == -1: continue
            
            img_pil = Image.open(str(base_dir / row['image_path'])).convert('RGB')
            faces = mtcnn(img_pil)
            if faces is None: continue
            
            faces = faces.to(device)
            preds_all = torch.argmax(resnet(faces), dim=1).cpu().numpy()
            
            if facenet_id in preds_all:
                match_idx = np.where(preds_all == facenet_id)[0][0]
                best_face = faces[match_idx] # Range [-1, 1]
                # Convertiamo in [0, 1]
                sample_img_01 = ((best_face + 1.0) / 2.0).unsqueeze(0)
                true_facenet_id = facenet_id
                identity_name = row['identity_name']
                break

    if sample_img_01 is None:
        print("Errore: Nessuna immagine valida trovata.")
        return
        
    print(f"   [OK] Trovato: {identity_name} (FaceNet ID: {true_facenet_id})")

    # --- 2. VALUTAZIONE CLEAN (Controllo Confidenza) ---
    with torch.no_grad():
        clean_logits = wrapped_model_full(sample_img_01)
        clean_probs = F.softmax(clean_logits, dim=1)
        clean_conf = float(clean_probs[0, true_facenet_id].cpu())
        print(f"\n-> Confidenza Rete su Immagine Pulita: {clean_conf*100:.2f}%")

    # --- 3. ATTACCO DEEPFOOL ---
    print("\n-> Avvio DeepFool (steps=50, overshoot=0.02)...")
    
    # Prepariamo i wrapper per l'attacco
    wrapped_model_spliced.freeze_target_classes(sample_img_01)
    local_y = torch.zeros(1, dtype=torch.long, device=device) # Indice 0 = top1
    
    attack = torchattacks.DeepFool(wrapped_model_spliced, steps=50, overshoot=0.02)
    
    # Generazione
    adv_img_01 = attack(sample_img_01, local_y)

    # --- 4. VALUTAZIONE AVVERSARIA ---
    with torch.no_grad():
        adv_logits = wrapped_model_full(adv_img_01)
        adv_pred = int(torch.argmax(adv_logits, dim=1).cpu()[0])
        adv_probs = F.softmax(adv_logits, dim=1)
        
        # Qual è la nuova identità?
        adv_identity_info = mapper.get_info_by_facenet_id(adv_pred)
        adv_name = adv_identity_info['Name'] if adv_identity_info else f"Unknown Class {adv_pred}"
        
        adv_conf_on_target = float(adv_probs[0, adv_pred].cpu())
        adv_conf_on_original = float(adv_probs[0, true_facenet_id].cpu())

    # Calcolo L_inf e MSE
    diff = torch.abs(adv_img_01 - sample_img_01)
    l_inf = float(torch.amax(diff).cpu())
    
    print("\n======================================")
    print(" RISULTATI DEL SANITY CHECK           ")
    print("======================================")
    print(f" L_inf Epsilon reale   : {l_inf:.4f} (Deve essere < 0.10 per la traccia)")
    if adv_pred != true_facenet_id:
        print(f" Esito Attacco         : SUCCESSO!")
        print(f" Nuova Identità        : {adv_name} (Confidenza: {adv_conf_on_target*100:.2f}%)")
        print(f" Conf. su Identità Vera: {adv_conf_on_original*100:.2f}% (Decaduta)")
    else:
        print(f" Esito Attacco         : FALLITO.")
        print(f" La rete crede ancora che sia {identity_name}.")

    # --- 5. VISUALIZZAZIONE ---
    print("\n-> Generazione Showcase Visivo...")
    
    c_img_plot = np.transpose(sample_img_01[0].cpu().numpy(), (1, 2, 0))
    a_img_plot = np.transpose(adv_img_01[0].cpu().numpy(), (1, 2, 0))
    
    plot_path = str(out_dir / "deepfool_sanity_check.png")
    
    plot_adversarial_showcase(
        clean_img=c_img_plot, 
        adv_img=a_img_plot, 
        true_label_name=f"ID {true_facenet_id}", 
        adv_label_name=f"ID {adv_pred} ({'HIT' if adv_pred != true_facenet_id else 'MISS'})", 
        save_flag=True, 
        save_path=plot_path
    )
    
    print(f"Controlla l'immagine in: {plot_path}")

if __name__ == "__main__":
    main()