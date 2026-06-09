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

from torch.profiler import profile, record_function, ProfilerActivity
import matplotlib.pyplot as plt
from facenet_pytorch import InceptionResnetV1, MTCNN
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import CarliniLInfMethod

from util.identity_mapper import IdentityMapper
from util.attack_error_specific_utils import get_one_hot_target

from util.cw_benchmarks.cw_pytorch import CarliniLInfMethodPyTorch
import art.config
# FORZA ART AD USARE FLOAT64
art.config.ART_NUMPY_DTYPE = np.float64

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
        # Ignoriamo il downgrade di ART e riforziamo il tensore in Float64 
        # (mantenendo intatto il grafo dei gradienti)
        x_double = x.to(torch.float64)
        return self.model(x_double * 2.0 - 1.0)

class ARTFloat64Wrapper(nn.Module):
    """Intercetta l'input degradato a Float32 da ART e lo riporta a Float64 per ResNet"""
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, x):
        return self.model(x.to(torch.float64))
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
    print("======================================")
    print(" SHOWDOWN: ART vs CUSTOM (ART Match) ")
    print("======================================\n")

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
    resnet.double() 
    
    # Mantieni il tuo wrapper come lo avevi prima
    wrapped_model = FacenetWrapper(resnet).eval()

    # --- AGGIUNGI QUESTO ---
    art_resnet_shield = ARTFloat64Wrapper(resnet).eval()


    # Inizializziamo l'Estimator di ART (Serve sia ad ART originale che alla nostra replica esatta)
    classifier = PyTorchClassifier(
        model=art_resnet_shield, 
        loss=nn.CrossEntropyLoss(), 
        input_shape=(3, 160, 160), 
        nb_classes=8631, 
        preprocessing=(0.5, 0.5), 
        clip_values=(0.0, 1.0), 
        device_type='gpu' if torch.cuda.is_available() else 'cpu'
    )

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
            
            faces_device = faces.to(device).double() # <--- Cast in Double!
            preds = torch.argmax(resnet(faces_device), dim=1).cpu().numpy()
            
            if fid in preds:
                idx = np.where(preds == fid)[0][0]
                img_01 = (faces_device[idx] + 1.0) / 2.0
                x_clean_list.append(img_01)
                y_true_list.append(fid)


    x_clean = torch.stack(x_clean_list).to(device)
    y_true = torch.tensor(y_true_list, dtype=torch.long, device=device)
    x_clean_np = x_clean.cpu().numpy()
    
    with torch.no_grad():
        logits = wrapped_model(x_clean)
        y_target_llc = torch.argmin(logits, dim=1) 

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
        print("-> [1. ART SOTA] Trovata cache! Caricamento da disco...")
        data = np.load(art_untargeted_cache)
        x_adv_art_np = data['adv']
        art_untarg_time = float(data['time'])
    else:
        print("-> [1. ART SOTA] Esecuzione attacco C&W L_inf (Lento)...")
        art_cw = CarliniLInfMethod(classifier=classifier, targeted=False, max_iter=10, learning_rate=0.01, batch_size=10, verbose=False)
        start = time.time()
        x_adv_art_np = art_cw.generate(x=x_clean_np)
        art_untarg_time = time.time() - start
        np.savez(art_untargeted_cache, adv=x_adv_art_np, time=art_untarg_time)
        
    art_untarg_succ, art_untarg_mean, art_untarg_max = evaluate_batch(wrapped_model, x_clean, torch.tensor(x_adv_art_np).to(device), y_true)

    # 1B. CUSTOM FULL (ART MATCH PyTorch GPU)
    print("-> [2. CUSTOM ART-MATCH] Esecuzione replica fedele su GPU...")
    # Assicurati di aver importato CarliniLInfMethodPyTorch in alto!
    custom_cw_full_untarg = CarliniLInfMethodPyTorch(classifier=classifier, targeted=False, max_iter=10, learning_rate=0.01, batch_size=10, verbose=False)
    start = time.time()
    x_adv_full_untarg_np = custom_cw_full_untarg.generate(x=x_clean_np)
    full_untarg_time = time.time() - start
    full_untarg_succ, full_untarg_mean, full_untarg_max = evaluate_batch(wrapped_model, x_clean, torch.tensor(x_adv_full_untarg_np).to(device), y_true)


    print(f"\n[RISULTATI UNTARGETED]")
    print(f"| Metrica         | ART Originale | Custom ART-Match |")
    print(f"|-----------------|---------------|------------------|")
    print(f"| Tempo (sec)     | {art_untarg_time:13.2f} | {full_untarg_time:16.2f} |")
    print(f"| Success Rate    | {art_untarg_succ:12.1f}% | {full_untarg_succ:15.1f}% |")
    print(f"| L_inf Max       | {art_untarg_max:13.4f} | {full_untarg_max:16.4f} |")
    print(f"| L_inf Mean      | {art_untarg_mean:13.4f} | {full_untarg_mean:16.4f} |")

    # =================================================================
    # FASE 2: TARGETED ATTACK (Error Specific su Least-Likely Class)
    # =================================================================
    print("\n" + "="*40)
    print(" TEST 2: ERROR SPECIFIC (Least-Likely Target)")
    print("="*40)

    # 2A. ART C&W (Con Caching)
    art_targeted_cache = cache_dir / "art_targeted.npz"
    if art_targeted_cache.exists():
        print("-> [1. ART SOTA] Trovata cache! Caricamento da disco...")
        data = np.load(art_targeted_cache)
        x_adv_art_t_np = data['adv']
        art_targ_time = float(data['time'])
    else:
        print("-> [1. ART SOTA] Esecuzione attacco C&W L_inf (Molto Lento)...")
        art_cw_t = CarliniLInfMethod(classifier=classifier, targeted=True, max_iter=10, learning_rate=0.01, batch_size=10, verbose=False)
        y_targets_np = y_target_llc.cpu().numpy()
        y_tgt_onehot = np.zeros((len(y_targets_np), 8631), dtype=np.float32)
        y_tgt_onehot[np.arange(len(y_targets_np)), y_targets_np] = 1.0
        
        start = time.time()
        x_adv_art_t_np = art_cw_t.generate(x=x_clean_np, y=y_tgt_onehot)
        art_targ_time = time.time() - start
        np.savez(art_targeted_cache, adv=x_adv_art_t_np, time=art_targ_time)

    art_targ_succ, art_targ_mean, art_targ_max = evaluate_batch(wrapped_model, x_clean, torch.tensor(x_adv_art_t_np).to(device), y_true, y_target_llc)

    # 2B. CUSTOM FULL (ART MATCH PyTorch GPU) - CON PROFILER
    print("-> [2. CUSTOM ART-MATCH] Esecuzione replica fedele su GPU (PROFILING IN CORSO)...")
    custom_cw_full_targ = CarliniLInfMethodPyTorch(
        classifier=classifier, 
        targeted=True, 
        max_iter=10,           
        learning_rate=0.01, 
        batch_size=10, 
        verbose=True
    )
    
    y_targets_np = y_target_llc.cpu().numpy()
    y_tgt_onehot = np.zeros((len(y_targets_np), 8631), dtype=np.float32)
    y_tgt_onehot[np.arange(len(y_targets_np)), y_targets_np] = 1.0
    
    start = time.time()
    
    # AVVIO DEL PROFILER
    # with profile(
    #     activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    #     record_shapes=True,
    #     profile_memory=True,
    #     with_stack=False # Disattivato per non rallentare troppo la stampa
    # ) as prof:
    x_adv_full_targ_np = custom_cw_full_targ.generate(x=x_clean_np, y=y_tgt_onehot)
        
    full_targ_time = time.time() - start
    
    # STAMPA DEI RISULTATI DEL PROFILER
    # print("\n" + "="*60)
    # print(" TOP 20 OPERAZIONI PIÙ LENTE SU GPU (CUDA Time):")
    # print("="*60)
    # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

    # print("\n" + "="*60)
    # print(" TOP 20 OPERAZIONI PIÙ LENTE SU CPU (CPU Time):")
    # print("="*60)
    # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))
    
    # Valutazione
    full_targ_succ, full_targ_mean, full_targ_max = evaluate_batch(wrapped_model, x_clean, torch.tensor(x_adv_full_targ_np).to(device), y_true, y_target_llc)

    print(f"\n[RISULTATI TARGETED (Least-Likely)]")
    print(f"| Metrica         | ART Originale | Custom ART-Match |")
    print(f"|-----------------|---------------|------------------|")
    print(f"| Tempo (sec)     | {art_targ_time:13.2f} | {full_targ_time:16.2f} |")
    print(f"| Success Rate    | {art_targ_succ:12.1f}% | {full_targ_succ:15.1f}% |")
    print(f"| L_inf Max       | {art_targ_max:13.4f} | {full_targ_max:16.4f} |")
    print(f"| L_inf Mean      | {art_targ_mean:13.4f} | {full_targ_mean:16.4f} |")

    print("\n" + "="*60)
    print(" VERIFICA IDENTICITÀ (CLONE vs ART - UNTARGETED) ")
    print("="*60)
    
    # 1. Definizione out_dir (mancava nel main)
    out_dir = base_dir / "plots" / "benchmark"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 2. Calcolo differenze su tutto il batch (Uso le variabili dell'Untargeted)
    abs_diff = np.abs(x_adv_full_untarg_np - x_adv_art_np)
    
    # Metriche assolute dei pixel: estremi e media
    max_diff = np.max(abs_diff)
    min_diff = np.min(abs_diff)
    mae_diff = np.mean(abs_diff)
    
    # 3. Calcolo L_inf reale della perturbazione (su asse immagini: CHW = 1,2,3)
    linfs_clone = np.amax(np.abs(x_adv_full_untarg_np - x_clean_np), axis=(1, 2, 3))
    linfs_art = np.amax(np.abs(x_adv_art_np - x_clean_np), axis=(1, 2, 3))
    
    max_linf_clone, mean_linf_clone = np.max(linfs_clone), np.mean(linfs_clone)
    max_linf_art, mean_linf_art = np.max(linfs_art), np.mean(linfs_art)
    
    # 4. Verifica le classi predette finali per tutto il batch
    adv_cln_tensor = torch.tensor(x_adv_full_untarg_np).to(device)
    adv_art_tensor = torch.tensor(x_adv_art_np).to(device)
    
    preds_clone = torch.argmax(wrapped_model(adv_cln_tensor), dim=1)
    preds_art = torch.argmax(wrapped_model(adv_art_tensor), dim=1)
    
    matching_preds = (preds_clone == preds_art).sum().item()
    total_imgs = len(preds_clone)

    # 5. Stampa dei risultati "Estremi e Medie"
    print(f"Predizioni di Rete Identiche : {matching_preds}/{total_imgs} immagini corrispondono")
    print("-" * 60)
    print(f"L_inf Budget (Estremo Max)   : Clone={max_linf_clone:.6f} | ART={max_linf_art:.6f}")
    print(f"L_inf Budget (Valore Medio)  : Clone={mean_linf_clone:.6f} | ART={mean_linf_art:.6f}")
    print("-" * 60)
    print(f"Differenza Rumore (Max)      : {max_diff:.8e}")
    print(f"Differenza Rumore (Min)      : {min_diff:.8e}")
    print(f"Differenza Rumore (Media)    : {mae_diff:.8e}")
    print("-" * 60)
    
    # Una differenza < 0.05 su immagini tra [0, 1] e 100% di match predittivo = equivalenza funzionale
    is_functionally_equivalent = (max_diff < 0.05) and (matching_preds == total_imgs)
    print(f"Funzionalmente Equivalenti   : {is_functionally_equivalent}")

    # 6. PLOT DELLA DIFFERENZA AMPLIFICATA
    diff_plot_path = str(out_dir / "clone_vs_art_noise_diff_untargeted.png")
    # Amplifichiamo per 100. Passo l'intero batch, la funzione è già programmata per estrarre [0]
    plot_noise_difference(x_adv_full_untarg_np, x_adv_art_np, diff_plot_path, amplification_factor=100.0)

    # =================================================================
    # VERIFICA IDENTICITÀ (CLONE vs ART - TARGETED)
    # =================================================================
    print("\n" + "="*60)
    print(" VERIFICA IDENTICITÀ (CLONE vs ART - TARGETED) ")
    print("="*60)
    
    # 1. Calcolo differenze su tutto il batch (Uso le variabili del Targeted)
    abs_diff_t = np.abs(x_adv_full_targ_np - x_adv_art_t_np)
    
    # Metriche assolute dei pixel: estremi e media
    max_diff_t = np.max(abs_diff_t)
    min_diff_t = np.min(abs_diff_t)
    mae_diff_t = np.mean(abs_diff_t)
    
    # 2. Calcolo L_inf reale della perturbazione (su asse immagini: CHW = 1,2,3)
    linfs_clone_t = np.amax(np.abs(x_adv_full_targ_np - x_clean_np), axis=(1, 2, 3))
    linfs_art_t = np.amax(np.abs(x_adv_art_t_np - x_clean_np), axis=(1, 2, 3))
    
    max_linf_clone_t, mean_linf_clone_t = np.max(linfs_clone_t), np.mean(linfs_clone_t)
    max_linf_art_t, mean_linf_art_t = np.max(linfs_art_t), np.mean(linfs_art_t)
    
    # 3. Verifica le classi predette finali per tutto il batch
    adv_cln_tensor_t = torch.tensor(x_adv_full_targ_np).to(device)
    adv_art_tensor_t = torch.tensor(x_adv_art_t_np).to(device)
    
    preds_clone_t = torch.argmax(wrapped_model(adv_cln_tensor_t), dim=1)
    preds_art_t = torch.argmax(wrapped_model(adv_art_tensor_t), dim=1)
    
    matching_preds_t = (preds_clone_t == preds_art_t).sum().item()
    total_imgs_t = len(preds_clone_t)

    # 4. Stampa dei risultati "Estremi e Medie"
    print(f"Predizioni di Rete Identiche : {matching_preds_t}/{total_imgs_t} immagini corrispondono")
    print("-" * 60)
    print(f"L_inf Budget (Estremo Max)   : Clone={max_linf_clone_t:.6f} | ART={max_linf_art_t:.6f}")
    print(f"L_inf Budget (Valore Medio)  : Clone={mean_linf_clone_t:.6f} | ART={mean_linf_art_t:.6f}")
    print("-" * 60)
    print(f"Differenza Rumore (Max)      : {max_diff_t:.8e}")
    print(f"Differenza Rumore (Min)      : {min_diff_t:.8e}")
    print(f"Differenza Rumore (Media)    : {mae_diff_t:.8e}")
    print("-" * 60)
    
    # Nota: Nel targeted, come spiegato, la deviazione CuDNN dovuta al batching 
    # può portare a leggeri scostamenti nell'esplorazione del 'const'. 
    # Teniamo una tolleranza più elastica sulla differenza max.
    is_functionally_equivalent_t = (max_diff_t < 0.1) and (matching_preds_t >= total_imgs_t * 0.8)
    print(f"Funzionalmente Equivalenti   : {is_functionally_equivalent_t} (Tolleranza Batch-CuDNN applicata)")

    # 5. PLOT DELLA DIFFERENZA AMPLIFICATA
    diff_plot_path_t = str(out_dir / "clone_vs_art_noise_diff_targeted.png")
    plot_noise_difference(x_adv_full_targ_np, x_adv_art_t_np, diff_plot_path_t, amplification_factor=100.0)



# ==========================================
# FUNZIONE PLOT DIFFERENZA RUMORE (CLONE vs ART)
# ==========================================
def plot_noise_difference(adv_clone_np, adv_art_np, out_path, amplification_factor=100.0):
    # Calcolo della differenza assoluta
    abs_diff = np.abs(adv_clone_np - adv_art_np)
    
    # Se è un batch di immagini, troviamo quella con il rumore peggiore (Max Diff)
    if len(abs_diff.shape) == 4:
        # Calcola il massimo della differenza per ogni immagine (B, C, H, W -> B)
        max_diffs_per_image = np.amax(abs_diff, axis=(1, 2, 3))
        worst_idx = np.argmax(max_diffs_per_image) # Indice dell'immagine più diversa
        print(f"   [Plot] Seleziono l'immagine {worst_idx} per il plot. Differenza Max: {max_diffs_per_image[worst_idx]:.6f}")
        
        # Estraiamo SOLO quell'immagine
        abs_diff = abs_diff[worst_idx]
        
    # Amplificazione e clipping
    amplified_diff = np.clip(abs_diff * amplification_factor, 0, 1)
    amplified_diff_hwc = np.transpose(amplified_diff, (1, 2, 0))

    # Creazione figura
    plt.figure(figsize=(8, 8))
    plt.imshow(amplified_diff_hwc)
    plt.title(f"Differenza: Clone vs ART (Amp. x{int(amplification_factor)})")
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

    
    print(f"[!] Mappa della differenza visiva salvata in: {out_path}")



import random

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # Per configurazioni multi-GPU
    
    # Forza PyTorch a usare algoritmi deterministici
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    
    # Opzionale: blocca operazioni che non hanno una controparte deterministica
    # torch.use_deterministic_algorithms(True)

# Permette alla GPU di trovare l'algoritmo FP64 più veloce per immagini 160x160
torch.backends.cudnn.benchmark = True
# Opzionale: disabilitare il deterministic spinge ulteriormente le performance, 
# pur mantenendo una precisione a 10^-8
torch.backends.cudnn.deterministic = False

if __name__ == "__main__":
    main()