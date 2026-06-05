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

# Importiamo tutte e 3 le versioni
from util.cw_custom import PyTorchCarliniLInf_EarlyStop, PyTorchCarliniLInf_BinarySteps

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
# FUNZIONE DI PLOT COMPARATIVO
# ==========================================
def plot_benchmark_v3(clean_np, adv_es_np, adv_full_np, adv_art_np, out_path):
    # Rumore amplificato x10
    noise_es = np.clip(np.abs(adv_es_np - clean_np) * 10.0, 0, 1)
    noise_full = np.clip(np.abs(adv_full_np - clean_np) * 10.0, 0, 1)
    noise_art = np.clip(np.abs(adv_art_np - clean_np) * 10.0, 0, 1)

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle('Global Benchmark C&W su Batch (Visualizzata prima img): Custom (ES) vs Custom (Full) vs ART', fontsize=16)

    # Immagini (Riga 1)
    axes[0, 0].imshow(np.transpose(clean_np, (1, 2, 0)))
    axes[0, 0].set_title("1. Immagine Pulita")
    axes[0, 0].axis('off')

    axes[0, 1].imshow(np.transpose(adv_es_np, (1, 2, 0)))
    axes[0, 1].set_title("2. Custom ES")
    axes[0, 1].axis('off')

    axes[0, 2].imshow(np.transpose(adv_full_np, (1, 2, 0)))
    axes[0, 2].set_title("3. Custom Full")
    axes[0, 2].axis('off')

    axes[0, 3].imshow(np.transpose(adv_art_np, (1, 2, 0)))
    axes[0, 3].set_title("4. ART (SOTA)")
    axes[0, 3].axis('off')

    # Rumori (Riga 2)
    axes[1, 0].axis('off') 
    
    axes[1, 1].imshow(np.transpose(noise_es, (1, 2, 0)))
    axes[1, 1].set_title("Rumore ES (x10)")
    axes[1, 1].axis('off')

    axes[1, 2].imshow(np.transpose(noise_full, (1, 2, 0)))
    axes[1, 2].set_title("Rumore Full (x10)")
    axes[1, 2].axis('off')

    axes[1, 3].imshow(np.transpose(noise_art, (1, 2, 0)))
    axes[1, 3].set_title("Rumore ART (x10)")
    axes[1, 3].axis('off')

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"\n[!] Grafico comparativo globale salvato in: {out_path}")

# ==========================================
# MAIN
# ==========================================
def main():
    print("======================================================")
    print(" GLOBAL BENCHMARK C&W (BATCH DA 10 IMMAGINI)          ")
    print("======================================================\n")

    BATCH_SIZE = 10

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
    wrapped_model_spliced = TopKFacenetWrapper(resnet, k=10).eval()

    df_clean = pd.read_csv(csv_path)

    # --- 1. ESTRAZIONE BATCH IMMAGINI ---
    clean_images_list = []
    true_facenet_ids_list = []

    print(f"[*] Ricerca di {BATCH_SIZE} immagini valide...")
    with torch.no_grad():
        for _, row in df_clean.iterrows():
            if len(clean_images_list) >= BATCH_SIZE:
                break

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
                img_tensor = ((best_face + 1.0) / 2.0).unsqueeze(0)
                clean_images_list.append(img_tensor)
                true_facenet_ids_list.append(facenet_id)

    if len(clean_images_list) == 0:
        print("[!] Nessuna immagine trovata. Esco.")
        return

    # Creazione dei tensori Batch
    sample_img_batch = torch.cat(clean_images_list, dim=0) # Shape: (B, 3, 160, 160)
    actual_batch_size = sample_img_batch.size(0)
    
    # Preparazione dei target labels (globali e locali)
    target_ids_global = []
    with torch.no_grad():
        clean_logits = wrapped_model_full(sample_img_batch)
        for i in range(actual_batch_size):
            sorted_indices = torch.argsort(clean_logits[i], descending=True)
            t_id = int(sorted_indices[1]) if int(sorted_indices[0]) == true_facenet_ids_list[i] else int(sorted_indices[0])
            target_ids_global.append(t_id)
    
    print(f"[OK] Trovate {actual_batch_size} immagini. Inizio attacchi...\n")

    wrapped_model_spliced.freeze_target_classes(sample_img_batch)
    target_idx_local = []
    for i in range(actual_batch_size):
        idx = int(torch.where(wrapped_model_spliced.active_indices[i] == target_ids_global[i])[0].item())
        target_idx_local.append(idx)
        
    local_y_batch = torch.tensor(target_idx_local, dtype=torch.long, device=device)

    # --- FUNZIONE DI SUPPORTO PER VALUTARE IL BATCH ---
    def eval_batch(adv_tensor):
        linf = float(torch.amax(torch.abs(adv_tensor - sample_img_batch), dim=(1,2,3)).mean().cpu())
        preds = torch.argmax(wrapped_model_full(adv_tensor), dim=1).cpu().numpy()
        successes = np.sum(preds == np.array(target_ids_global))
        sr = (successes / actual_batch_size) * 100
        return f"{linf:.4f}", f"{successes}/{actual_batch_size} ({sr:.1f}%)"

    # --- TEST 1: CUSTOM EARLY STOP ---
    print("-> Esecuzione 1: Custom C&W (Early Stop)...")
    attack_es = PyTorchCarliniLInf_EarlyStop(model=wrapped_model_spliced, targeted=True, max_iter=50, learning_rate=0.01, early_stop_epsilon=0.10)
    t0 = time.time()
    adv_es = attack_es.forward(image=sample_img_batch, label=local_y_batch)
    time_es = time.time() - t0
    linf_es, succ_es = eval_batch(adv_es)

    # --- TEST 2: CUSTOM FULL OPTIMIZED ---
    print("-> Esecuzione 2: Custom C&W (True SOTA Binary Search)...")
    attack_opt = PyTorchCarliniLInf_BinarySteps(model=wrapped_model_spliced, targeted=True, max_iter=50, search_steps=9, learning_rate=0.01, loss_converged=0.001)
    t0 = time.time()
    adv_full = attack_opt.forward(image=sample_img_batch, label=local_y_batch)
    time_full = time.time() - t0
    linf_full, succ_full = eval_batch(adv_full)

    # --- TEST 3: LIBRERIA ART ---
    print("-> Esecuzione 3: Libreria ART (SOTA)...")
    loss_fn = nn.CrossEntropyLoss()
    classifier_art = PyTorchClassifier(model=wrapped_model_full, clip_values=(0.0, 1.0), loss=loss_fn, input_shape=(3, 160, 160), nb_classes=8631, device_type='gpu' if torch.cuda.is_available() else 'cpu')
    attack_art = CarliniLInfMethod(classifier=classifier_art, targeted=True, max_iter=50, learning_rate=0.01, initial_const=1e-3, largest_const=20.0)
    
    x_np = sample_img_batch.cpu().numpy()
    y_target_np = np.array(target_ids_global)
    
    t0 = time.time()
    adv_art_np = attack_art.generate(x=x_np, y=y_target_np)
    time_art = time.time() - t0
    adv_art_tensor = torch.tensor(adv_art_np).to(device)
    linf_art, succ_art = eval_batch(adv_art_tensor)

    # --- RISULTATI ---
    print("\n=================================================================")
    print(" SOMMARIO DEI RISULTATI GLOBALI SUL BATCH ")
    print("=================================================================")
    print(f"Metodo       | Tempo (s) | Media L_inf | Success Rate (Batch)")
    print(f"Custom (ES)  | {time_es:9.2f} | {linf_es:>11} | {succ_es}")
    print(f"Custom (Full)| {time_full:9.2f} | {linf_full:>11} | {succ_full}")
    print(f"ART (SOTA)   | {time_art:9.2f} | {linf_art:>11} | {succ_art}")
    print("=================================================================")

    # Plot della prima immagine del batch (indice 0)
    plot_path = str(out_dir / "global_benchmark_cw_batch.png")
    plot_benchmark_v3(x_np[0], adv_es[0].cpu().numpy(), adv_full[0].cpu().numpy(), adv_art_np[0], plot_path)

if __name__ == "__main__":
    main()