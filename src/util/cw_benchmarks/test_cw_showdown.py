import os
import cv2
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from pathlib import Path
from PIL import Image

from facenet_pytorch import InceptionResnetV1, MTCNN
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import CarliniLInfMethod

from util.identity_mapper import IdentityMapper
from util.attack_error_specific_utils import get_one_hot_target

# =====================================================================
# LA TUA CLASSE OTTIMIZZATA
# =====================================================================
class PyTorchCarliniLInf_BinarySteps:
    def __init__(self, model, targeted=False, confidence=0.0,
                 learning_rate=0.01, max_iter=50, search_steps=9, 
                 initial_const=1e-3, largest_const=20.0, loss_converged=0.001): 
        self.model = model
        self.targeted = targeted
        self.confidence = confidence
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.search_steps = search_steps
        self.initial_const = initial_const
        self.largest_const = largest_const
        self.loss_converged = loss_converged
        self.device = next(model.parameters()).device

    def atanh(self, x):
        return 0.5 * torch.log((1 + x) / (1 - x))

    def forward(self, image, label):
        batch_size = image.size(0)
        image = image.to(self.device)
        label = label.to(self.device)

        c = torch.full((batch_size,), self.initial_const, device=self.device)
        lower_bound = torch.zeros((batch_size,), device=self.device)
        upper_bound = torch.full((batch_size,), self.largest_const, device=self.device)
        
        tau = torch.ones((batch_size,), device=self.device)

        best_adv_image = image.clone().detach()
        best_Linf = torch.full((batch_size,), float('inf'), device=self.device)

        use_fused_adam = self.device.type == 'cuda' and hasattr(torch.optim.Adam, 'fused')

        for search in range(self.search_steps):
            x_clamp = torch.clamp(image, 1e-4, 1 - 1e-4)
            w = self.atanh(x_clamp * 2 - 1).clone().detach()
            w.requires_grad = True

            if use_fused_adam:
                optimizer = optim.Adam([w], lr=self.learning_rate, fused=True)
            else:
                optimizer = optim.Adam([w], lr=self.learning_rate)

            prev_loss = float('inf')

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
                loss_2 = torch.sum(torch.clamp(diff - tau.view(-1, 1, 1, 1), min=0.0), dim=(1, 2, 3))

                loss = torch.sum(c * loss_1 + loss_2)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()  
                optimizer.step()

                current_loss = loss.item()
                if abs(prev_loss - current_loss) < self.loss_converged:
                    break
                prev_loss = current_loss

            with torch.no_grad():
                eval_img = 0.5 * (torch.tanh(w) + 1)
                eval_logits = self.model(eval_img)
                eval_pred = torch.argmax(eval_logits, dim=1)
                
                eval_tau = torch.amax(torch.abs(eval_img - image), dim=(1, 2, 3))
                eval_success = (eval_pred == label) if self.targeted else (eval_pred != label)
                
                better_mask = eval_success & (eval_tau < best_Linf)
                if better_mask.any():
                    best_adv_image[better_mask] = eval_img[better_mask].detach()
                    best_Linf[better_mask] = eval_tau[better_mask]

                upper_bound = torch.where(eval_success, c, upper_bound)
                lower_bound = torch.where(~eval_success, c, lower_bound)
                
                tau = torch.where(eval_success, eval_tau * 0.9, tau)
                
                c_next_binary = (lower_bound + upper_bound) / 2.0
                c_next_exponential = c * 2.0
                c = torch.where(upper_bound < self.largest_const, c_next_binary, c_next_exponential)

        return best_adv_image

# =====================================================================
# WRAPPER FACENET
# =====================================================================
class FacenetWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x):
        return self.model(x * 2.0 - 1.0)

# =====================================================================
# FUNZIONI DI SUPPORTO PER IL TEST
# =====================================================================
def evaluate_batch(model, x_clean, x_adv, y_true, y_target=None):
    """Valuta il batch e ritorna Accuracy/ASR e L_inf."""
    with torch.no_grad():
        preds = torch.argmax(model(x_adv), dim=1)
        diff = torch.abs(x_adv - x_clean)
        linfs = torch.amax(diff, dim=(1, 2, 3))
        
        # Le immagini che hanno sforato il budget di 0.10 sono considerate FALLIMENTI
        valid_mask = linfs <= 0.10
        
        if y_target is None: # Untargeted
            success = (preds != y_true) & valid_mask
        else: # Targeted
            success = (preds == y_target) & valid_mask
            
        success_rate = success.float().mean().item() * 100
        mean_linf = linfs.mean().item()
        max_linf = linfs.max().item()
        
    return success_rate, mean_linf, max_linf

def main():
    print("======================================================")
    print(" SHOWDOWN: ART C&W vs CUSTOM BINARY-STEPS C&W         ")
    print("======================================================\n")

    base_dir = Path.cwd()
    csv_path = base_dir / "dataset" / "clean" / "splits" / "manifest.csv"
    meta_path = base_dir / "dataset" / "clean" / "splits" / "identity_meta.csv"
    
    # CARTELLA DI CACHE PER ART
    cache_dir = base_dir / "plots" / "debug" / "cw_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mapper = IdentityMapper(meta_path)
    mtcnn = MTCNN(image_size=160, margin=0, keep_all=True, post_process=True, device=device)
    
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    resnet.classify = True 
    wrapped_model = FacenetWrapper(resnet).eval()

    # --- PREPARAZIONE DEL BATCH DI TEST (10 Immagini) ---
    print("-> Preparazione batch di 10 immagini...")
    df = pd.read_csv(csv_path)
    
    x_clean_list, y_true_list = [], []
    
    with torch.no_grad():
        for _, row in df.iterrows():
            if len(x_clean_list) >= 10: break
            
            fid = mapper.get_facenet_id_by_class_id(str(row['identity_id']))
            if fid == -1: continue
            
            try:
                img_pil = Image.open(str(base_dir / row['image_path'])).convert('RGB')
            except: continue
            
            faces = mtcnn(img_pil)
            if faces is None: continue
            
            preds = torch.argmax(resnet(faces.to(device)), dim=1).cpu().numpy()
            if fid in preds:
                idx = np.where(preds == fid)[0][0]
                img_01 = (faces[idx] + 1.0) / 2.0
                x_clean_list.append(img_01)
                y_true_list.append(fid)

    x_clean = torch.stack(x_clean_list).to(device)
    y_true = torch.tensor(y_true_list, dtype=torch.long, device=device)
    x_clean_np = x_clean.cpu().numpy()
    
    # Calcolo del target "Least-Likely Class" per il test mirato
    with torch.no_grad():
        logits = wrapped_model(x_clean)
        y_target_llc = torch.argmin(logits, dim=1) # Il neurone con probabilità più bassa!

    print(f"   Batch preparato. Shape: {x_clean.shape}")

    # =================================================================
    # FASE 1: UNTARGETED ATTACK (Error Generic)
    # =================================================================
    print("\n" + "="*40)
    print(" TEST 1: ERROR GENERIC (Untargeted)")
    print("="*40)

    # 1A. ART C&W (Con Caching)
    art_untargeted_cache = cache_dir / "art_untargeted.npz"
    if art_untargeted_cache.exists():
        print("-> [ART] Trovata cache! Caricamento da disco...")
        data = np.load(art_untargeted_cache)
        x_adv_art_np = data['adv']
        art_untarg_time = float(data['time'])
    else:
        print("-> [ART] Esecuzione attacco C&W L_inf (Preparati ad aspettare)...")
        classifier = PyTorchClassifier(model=resnet, loss=nn.CrossEntropyLoss(), input_shape=(3, 160, 160), nb_classes=8631, preprocessing=(0.5, 0.5), clip_values=(0.0, 1.0), device_type='gpu')
        
        art_cw = CarliniLInfMethod(classifier=classifier, targeted=False, max_iter=10, learning_rate=0.01, batch_size=10, verbose=False)
        
        start = time.time()
        x_adv_art_np = art_cw.generate(x=x_clean_np)
        art_untarg_time = time.time() - start
        
        np.savez(art_untargeted_cache, adv=x_adv_art_np, time=art_untarg_time)
        print(f"   Cache salvata in {art_untargeted_cache}")

    art_untarg_succ, art_untarg_mean, art_untarg_max = evaluate_batch(wrapped_model, x_clean, torch.tensor(x_adv_art_np).to(device), y_true)

    # 1B. CUSTOM C&W (Binary Steps)
    print("-> [CUSTOM] Esecuzione attacco BinarySteps...")
    custom_cw_untarg = PyTorchCarliniLInf_BinarySteps(model=wrapped_model, targeted=False, max_iter=10, search_steps=9, learning_rate=0.01)
    
    start = time.time()
    x_adv_cust_untarg = custom_cw_untarg.forward(x_clean, y_true)
    cust_untarg_time = time.time() - start
    
    cust_untarg_succ, cust_untarg_mean, cust_untarg_max = evaluate_batch(wrapped_model, x_clean, x_adv_cust_untarg, y_true)

    print(f"\n[RISULTATI UNTARGETED]")
    print(f"| Metrica         | ART Originale | Custom BinarySteps |")
    print(f"|-----------------|---------------|--------------------|")
    print(f"| Tempo (sec)     | {art_untarg_time:13.2f} | {cust_untarg_time:18.2f} |")
    print(f"| Success Rate    | {art_untarg_succ:12.1f}% | {cust_untarg_succ:17.1f}% |")
    print(f"| L_inf Max       | {art_untarg_max:13.4f} | {cust_untarg_max:18.4f} |")
    print(f"| L_inf Mean      | {art_untarg_mean:13.4f} | {cust_untarg_mean:18.4f} |")

    # =================================================================
    # FASE 2: TARGETED ATTACK (Error Specific su Least-Likely Class)
    # =================================================================
    print("\n" + "="*40)
    print(" TEST 2: ERROR SPECIFIC (Least-Likely Target)")
    print("="*40)

    # 2A. ART C&W (Con Caching)
    art_targeted_cache = cache_dir / "art_targeted.npz"
    if art_targeted_cache.exists():
        print("-> [ART] Trovata cache! Caricamento da disco...")
        data = np.load(art_targeted_cache)
        x_adv_art_t_np = data['adv']
        art_targ_time = float(data['time'])
    else:
        print("-> [ART] Esecuzione attacco C&W L_inf (Preparati ad aspettare di nuovo)...")
        classifier = PyTorchClassifier(model=resnet, loss=nn.CrossEntropyLoss(), input_shape=(3, 160, 160), nb_classes=8631, preprocessing=(0.5, 0.5), clip_values=(0.0, 1.0), device_type='gpu')
        
        art_cw_t = CarliniLInfMethod(classifier=classifier, targeted=True, max_iter=100, learning_rate=0.01, batch_size=10, verbose=True)
        
        y_targets_np = y_target_llc.cpu().numpy()
        y_tgt_onehot = np.zeros((len(y_targets_np), 8631), dtype=np.float32)
        y_tgt_onehot[np.arange(len(y_targets_np)), y_targets_np] = 1.0
        
        start = time.time()
        x_adv_art_t_np = art_cw_t.generate(x=x_clean_np, y=y_tgt_onehot)
        art_targ_time = time.time() - start
        
        np.savez(art_targeted_cache, adv=x_adv_art_t_np, time=art_targ_time)

    art_targ_succ, art_targ_mean, art_targ_max = evaluate_batch(wrapped_model, x_clean, torch.tensor(x_adv_art_t_np).to(device), y_true, y_target_llc)

    # 2B. CUSTOM C&W (Binary Steps)
    print("-> [CUSTOM] Esecuzione attacco BinarySteps...")
    custom_cw_targ = PyTorchCarliniLInf_BinarySteps(model=wrapped_model, targeted=True, max_iter=100, search_steps=9, learning_rate=0.01)
    
    start = time.time()
    x_adv_cust_targ = custom_cw_targ.forward(x_clean, y_target_llc) # PyTorch nativo prende direttamente gli indici!
    cust_targ_time = time.time() - start
    
    cust_targ_succ, cust_targ_mean, cust_targ_max = evaluate_batch(wrapped_model, x_clean, x_adv_cust_targ, y_true, y_target_llc)

    print(f"\n[RISULTATI TARGETED (Least-Likely)]")
    print(f"| Metrica         | ART Originale | Custom BinarySteps |")
    print(f"|-----------------|---------------|--------------------|")
    print(f"| Tempo (sec)     | {art_targ_time:13.2f} | {cust_targ_time:18.2f} |")
    print(f"| Success Rate    | {art_targ_succ:12.1f}% | {cust_targ_succ:17.1f}% |")
    print(f"| L_inf Max       | {art_targ_max:13.4f} | {cust_targ_max:18.4f} |")
    print(f"| L_inf Mean      | {art_targ_mean:13.4f} | {cust_targ_mean:18.4f} |")

if __name__ == "__main__":
    main()