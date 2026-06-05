import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

from facenet_pytorch import InceptionResnetV1, MTCNN
from util.identity_mapper import IdentityMapper

from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import CarliniLInfMethod

# Importiamo la nuova classe automatica
from util.cw_custom import PyTorchCarliniLInf_EarlyStop, PyTorchCarliniLInf_BinarySteps, PyTorchCarliniLInf_ARTMatch

# ==========================================
# WRAPPERS
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
            _, self.active_indices = torch.topk(self.model(x_scaled), self.k, dim=1)
    def forward(self, x):
        return torch.gather(self.model((x * 2.0) - 1.0), 1, self.active_indices)

class FacenetWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x):
        return self.model((x * 2.0) - 1.0)

# ==========================================
# FUNZIONE DI PLOT COMPARATIVO A 5 COLONNE
# ==========================================
def plot_benchmark_v4(clean_np, adv_es_np, adv_full_np, adv_auto_np, adv_art_np, out_path):
    noise_es = np.clip(np.abs(adv_es_np - clean_np) * 10.0, 0, 1)
    noise_full = np.clip(np.abs(adv_full_np - clean_np) * 10.0, 0, 1)
    noise_auto = np.clip(np.abs(adv_auto_np - clean_np) * 10.0, 0, 1)
    noise_art = np.clip(np.abs(adv_art_np - clean_np) * 10.0, 0, 1)

    fig, axes = plt.subplots(2, 5, figsize=(25, 10))
    fig.suptitle('Global Benchmark C&W (UNTARGETED): Custom vs AutoTemp vs ART', fontsize=16)

    axes[0, 0].imshow(np.transpose(clean_np, (1, 2, 0)))
    axes[0, 0].set_title("1. Immagine Pulita")
    axes[0, 0].axis('off')

    axes[0, 1].imshow(np.transpose(adv_es_np, (1, 2, 0)))
    axes[0, 1].set_title("2. Custom ES")
    axes[0, 1].axis('off')

    axes[0, 2].imshow(np.transpose(adv_full_np, (1, 2, 0)))
    axes[0, 2].set_title("3. Custom Full Opt")
    axes[0, 2].axis('off')
    
    axes[0, 3].imshow(np.transpose(adv_auto_np, (1, 2, 0)))
    axes[0, 3].set_title("4. Custom Auto-Temp")
    axes[0, 3].axis('off')

    axes[0, 4].imshow(np.transpose(adv_art_np, (1, 2, 0)))
    axes[0, 4].set_title("5. Libreria ART")
    axes[0, 4].axis('off')

    axes[1, 0].axis('off') 
    
    axes[1, 1].imshow(np.transpose(noise_es, (1, 2, 0)))
    axes[1, 1].set_title("Rumore ES (x10)")
    axes[1, 1].axis('off')

    axes[1, 2].imshow(np.transpose(noise_full, (1, 2, 0)))
    axes[1, 2].set_title("Rumore Full (x10)")
    axes[1, 2].axis('off')
    
    axes[1, 3].imshow(np.transpose(noise_auto, (1, 2, 0)))
    axes[1, 3].set_title("Rumore AutoTemp (x10)")
    axes[1, 3].axis('off')

    axes[1, 4].imshow(np.transpose(noise_art, (1, 2, 0)))
    axes[1, 4].set_title("Rumore ART (x10)")
    axes[1, 4].axis('off')

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"\n[!] Grafico comparativo salvato in: {out_path}")

# ==========================================
# MAIN
# ==========================================
def main():
    print("======================================================")
    print(" GLOBAL BENCHMARK C&W UNTARGETED (ES vs FULL vs AUTOTEMP vs ART) ")
    print("======================================================\n")

    base_dir = Path.cwd()
    csv_path = base_dir / "dataset" / "clean" / "splits" / "manifest.csv"
    meta_csv_path = base_dir / "dataset" / "clean" / "splits" / "identity_meta.csv"
    out_dir = base_dir / "plots" / "benchmark"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mapper = IdentityMapper(meta_csv_path)
    mtcnn = MTCNN(image_size=160, margin=0, keep_all=True, post_process=True, device=device)
    
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    resnet.classify = True 
    
    wrapped_model_full = FacenetWrapper(resnet).eval()
    wrapped_model_spliced = TopKFacenetWrapper(resnet, k=8631).eval()

    df_clean = pd.read_csv(csv_path)

    sample_img_01 = None
    true_facenet_id = None

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
                break

    if sample_img_01 is None: return

    print(f"[OK] Immagine Trovata (True ID: {true_facenet_id}). L'attacco cercherà di farle cambiare classe.\n")

    # Mappiamo l'ID originale (true_facenet_id) nell'indice locale del wrapper
    wrapped_model_spliced.freeze_target_classes(sample_img_01)
    original_idx_local = int(torch.where(wrapped_model_spliced.active_indices[0] == true_facenet_id)[0].item())
    
    # local_y ora rappresenta l'ID ORIGINALE che vogliamo "sfuggire"
    local_y = torch.tensor([original_idx_local], dtype=torch.long, device=device)

    # --- TEST 1: CUSTOM EARLY STOP (UNTARGETED) ---
    print("-> Esecuzione 1: Custom C&W (Early Stop)...")
    attack_es = PyTorchCarliniLInf_EarlyStop(model=wrapped_model_spliced, targeted=False, max_iter=50)
    t0 = time.time()
    adv_es = attack_es.forward(image=sample_img_01, label=local_y)
    time_es = time.time() - t0
    linf_es = float(torch.amax(torch.abs(adv_es - sample_img_01)).cpu())
    succ_es = (int(torch.argmax(wrapped_model_full(adv_es), dim=1).cpu()[0]) != true_facenet_id)

    # --- TEST 2: CUSTOM FULL OPTIMIZED (UNTARGETED) ---
    print("-> Esecuzione 2: Custom C&W (Binary Search)...")
    attack_opt = PyTorchCarliniLInf_BinarySteps(model=wrapped_model_spliced, targeted=False, max_iter=50)
    t0 = time.time()
    adv_full = attack_opt.forward(image=sample_img_01, label=local_y)
    time_full = time.time() - t0
    linf_full = float(torch.amax(torch.abs(adv_full - sample_img_01)).cpu())
    succ_full = (int(torch.argmax(wrapped_model_full(adv_full), dim=1).cpu()[0]) != true_facenet_id)
    
    # --- TEST 3: CUSTOM AUTO TEMP SCALING (UNTARGETED) ---
    print("-> Esecuzione 3: Custom C&W (ART Match)...")
    attack_auto = PyTorchCarliniLInf_ARTMatch(model=wrapped_model_spliced, targeted=False, max_iter=50)
    t0 = time.time()
    adv_auto = attack_auto.forward(image=sample_img_01, label=local_y)
    time_auto = time.time() - t0
    linf_auto = float(torch.amax(torch.abs(adv_auto - sample_img_01)).cpu())
    succ_auto = (int(torch.argmax(wrapped_model_full(adv_auto), dim=1).cpu()[0]) != true_facenet_id)

    # --- TEST 4: LIBRERIA ART (UNTARGETED) ---
    print("-> Esecuzione 4: Libreria ART (SOTA)...")
    loss_fn = nn.CrossEntropyLoss()
    classifier_art = PyTorchClassifier(model=wrapped_model_full, clip_values=(0.0, 1.0), loss=loss_fn, input_shape=(3, 160, 160), nb_classes=8631, device_type='gpu' if torch.cuda.is_available() else 'cpu')
    
    # targeted=False in ART
    attack_art = CarliniLInfMethod(classifier=classifier_art, targeted=False, max_iter=50, learning_rate=0.01, initial_const=1e-3, largest_const=20.0)
    x_np = sample_img_01.cpu().numpy()
    
    # Quando targeted=False, y rappresenta l'etichetta originaria
    y_true_np = np.array([true_facenet_id])
    
    t0 = time.time()
    adv_art_np = attack_art.generate(x=x_np, y=y_true_np)
    time_art = time.time() - t0
    linf_art = np.max(np.abs(adv_art_np - x_np))
    adv_art_tensor = torch.tensor(adv_art_np).to(device)
    succ_art = (int(torch.argmax(wrapped_model_full(adv_art_tensor), dim=1).cpu()[0]) != true_facenet_id)

    # --- RISULTATI ---
    print("\n======================================================")
    print(" SOMMARIO DEI RISULTATI GLOBALI (UNTARGETED) ")
    print("======================================================")
    print(f"Metodo       | Tempo (s) | L_inf Dist | Successo")
    print(f"Custom (ES)  | {time_es:9.2f} | {linf_es:10.4f} | {succ_es}")
    print(f"Custom (BinS)| {time_full:9.2f} | {linf_full:10.4f} | {succ_full}")
    print(f"Custom (Full)| {time_auto:9.2f} | {linf_auto:10.4f} | {succ_auto}")
    print(f"ART (SOTA)   | {time_art:9.2f} | {linf_art:10.4f} | {succ_art}")
    print("======================================================")

    plot_path = str(out_dir / "global_benchmark_cw_v4_untargeted.png")
    plot_benchmark_v4(x_np[0], adv_es.cpu().numpy()[0], adv_full.cpu().numpy()[0], adv_auto.cpu().numpy()[0], adv_art_np[0], plot_path)

if __name__ == "__main__":
    main()