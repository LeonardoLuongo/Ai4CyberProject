import os
import cv2
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

from facenet_pytorch import InceptionResnetV1, MTCNN
from identity_mapper import IdentityMapper
from plot.utils_plot_shared import plot_adversarial_showcase

# ==========================================
# 1. CLASSE C&W PURA CON EARLY STOPPING
# ==========================================
class PyTorchCarliniLInf_EarlyStop:
    """
    Traduzione nativa in PyTorch dell'algoritmo Carlini & Wagner L_infinity.
    Include l'Early Stopping Brutale se l'attacco ha successo e L_inf <= budget.
    """
    def __init__(self, model, targeted=False, confidence=0.0,
                 learning_rate=0.01, max_iter=50,
                 decrease_factor=0.9, initial_const=1e-3,
                 largest_const=20.0, const_factor=2.0, 
                 early_stop_epsilon=0.10): # <-- IL NOSTRO LIMITE
        self.model = model
        self.targeted = targeted
        self.confidence = confidence
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.decrease_factor = decrease_factor
        self.initial_const = initial_const
        self.largest_const = largest_const
        self.const_factor = const_factor
        self.early_stop_epsilon = early_stop_epsilon
        self.device = next(model.parameters()).device

    def atanh(self, x):
        return 0.5 * torch.log((1 + x) / (1 - x))

    def forward(self, image, label):
        image = image.to(self.device)
        label = label.to(self.device)

        best_adv_image = image.clone().detach()
        best_Linf = float('inf')

        tau = 1.0 
        sample_done = False

        while tau > 1.0 / 256.0 and not sample_done:
            sample_done = True
            const = self.initial_const

            while const < self.largest_const:
                x_clamp = torch.clamp(image, 1e-4, 1 - 1e-4)
                w = self.atanh(x_clamp * 2 - 1).clone().detach()
                w.requires_grad = True

                optimizer = optim.Adam([w], lr=self.learning_rate)

                for step in range(self.max_iter):
                    adv_image = 0.5 * (torch.tanh(w) + 1)
                    logits = self.model(adv_image)

                    one_hot = torch.eye(logits.shape[1], device=self.device)[label]
                    
                    real = torch.max(one_hot * logits, dim=1)[0]
                    other = torch.max((1 - one_hot) * logits - one_hot * 10000, dim=1)[0]

                    if self.targeted:
                        loss_1 = torch.clamp(other - real + self.confidence, min=0.0)
                    else:
                        loss_1 = torch.clamp(real - other + self.confidence, min=0.0)

                    diff = torch.abs(adv_image - image)
                    loss_2 = torch.sum(torch.clamp(diff - tau, min=0.0))

                    loss = const * loss_1 + loss_2

                    optimizer.zero_grad()
                    loss.backward()  
                    optimizer.step()
                    
                    # --- CONTROLLO EARLY STOPPING IN TEMPO REALE ---
                    # Per non fare inferenze inutili ogni step, controlliamo ogni 5 step
                    if step % 5 == 0:
                        with torch.no_grad():
                            eval_img = 0.5 * (torch.tanh(w) + 1)
                            eval_logits = self.model(eval_img)
                            eval_pred = torch.argmax(eval_logits, dim=1)
                            eval_tau = torch.max(torch.abs(eval_img - image)).item()
                            eval_success = (eval_pred == label) if self.targeted else (eval_pred != label)
                            
                            # Se l'attacco ha successo E siamo sotto il budget: ci fermiamo SUBITO!
                            if eval_success.item() and eval_tau <= self.early_stop_epsilon:
                                return eval_img.detach()

                with torch.no_grad():
                    adv_image_eval = 0.5 * (torch.tanh(w) + 1)
                    final_logits = self.model(adv_image_eval)
                    pred = torch.argmax(final_logits, dim=1)
                    
                    actual_tau = torch.max(torch.abs(adv_image_eval - image)).item()
                    is_success = (pred == label) if self.targeted else (pred != label)

                    if is_success.item() and actual_tau < best_Linf:
                        best_adv_image = adv_image_eval.clone()
                        best_Linf = actual_tau
                        sample_done = False 
                        
                        # Altro controllo Early Stopping a fine ciclo
                        if best_Linf <= self.early_stop_epsilon:
                            return best_adv_image.detach()

                const *= self.const_factor

            if best_Linf < tau:
                tau = best_Linf
            tau *= self.decrease_factor

        return best_adv_image

# ==========================================
# 2. WRAPPER TOP-K
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
    print(" SANITY CHECK: CARLINI & WAGNER PURO + EARLY STOPPING ")
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

    # --- 1. TROVIAMO UN'IMMAGINE ---
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
                best_face = faces[match_idx]
                sample_img_01 = ((best_face + 1.0) / 2.0).unsqueeze(0)
                true_facenet_id = facenet_id
                identity_name = row['identity_name']
                break

    if sample_img_01 is None:
        print("Errore: Nessuna immagine trovata.")
        return
        
    print(f"   [OK] Trovato: {identity_name} (FaceNet ID: {true_facenet_id})")

    # --- 2. SELEZIONE TARGET (Next-Best) ---
    with torch.no_grad():
        clean_logits = wrapped_model_full(sample_img_01)
        sorted_indices = torch.argsort(clean_logits, dim=1, descending=True)[0]
        
        target_id_global = int(sorted_indices[1]) if int(sorted_indices[0]) == true_facenet_id else int(sorted_indices[0])
        print(f"-> Bersaglio Scelto (Next-Best Global): ID {target_id_global}")

    # --- 3. ATTACCO C&W PURO ---
    print("\n-> Avvio C&W Puro con Early Stopping (eps=0.10)...")
    
    wrapped_model_spliced.freeze_target_classes(sample_img_01)
    
    target_idx_local_tensor = torch.where(wrapped_model_spliced.active_indices[0] == target_id_global)[0]
    if len(target_idx_local_tensor) == 0:
        print("Errore critico: Il target non è tra le top-10 classi!")
        return
    local_y = torch.tensor([target_idx_local_tensor.item()], dtype=torch.long, device=device)
    
    start_time = time.time()
    
    attack = PyTorchCarliniLInf_EarlyStop(
        model=wrapped_model_spliced, 
        targeted=True, 
        max_iter=50,         
        learning_rate=0.01,
        initial_const=1e-3,  
        largest_const=20.0,
        early_stop_epsilon=0.20 # Il nostro salvavita
    )
    
    adv_img_01_tensor = attack.forward(image=sample_img_01, label=local_y)
    
    end_time = time.time()

    # --- 4. VALUTAZIONE AVVERSARIA ---
    with torch.no_grad():
        adv_logits = wrapped_model_full(adv_img_01_tensor)
        adv_pred = int(torch.argmax(adv_logits, dim=1).cpu()[0])
        adv_probs = F.softmax(adv_logits, dim=1)
        
        adv_conf_on_target = float(adv_probs[0, target_id_global].cpu())

    diff = torch.abs(adv_img_01_tensor - sample_img_01)
    l_inf = float(torch.amax(diff).cpu())
    
    print("\n======================================")
    print(" RISULTATI DEL SANITY CHECK C&W PURO  ")
    print("======================================")
    print(f" Tempo Impiegato       : {end_time - start_time:.2f} secondi")
    print(f" L_inf Epsilon reale   : {l_inf:.4f} (Budget: 0.10)")
    
    if adv_pred == target_id_global:
        print(f" Esito Attacco         : SUCCESSO! \u2705")
        print(f" Conf. su Target       : {adv_conf_on_target*100:.2f}%")
    else:
        print(f" Esito Attacco         : FALLITO. \u274C")

    # --- 5. VISUALIZZAZIONE ---
    print("\n-> Generazione Showcase Visivo...")
    c_img_plot = np.transpose(sample_img_01[0].cpu().numpy(), (1, 2, 0))
    a_img_plot = np.transpose(adv_img_01_tensor[0].cpu().numpy(), (1, 2, 0))
    
    plot_path = str(out_dir / "cw_pure_early_stop.png")
    plot_adversarial_showcase(
        clean_img=c_img_plot, adv_img=a_img_plot, 
        true_label_name=f"ID {true_facenet_id}", 
        adv_label_name=f"ID {adv_pred} ({'HIT' if adv_pred == target_id_global else 'MISS'})", 
        save_flag=True, save_path=plot_path
    )
    print(f"Controlla l'immagine in: {plot_path}")

if __name__ == "__main__":
    main()