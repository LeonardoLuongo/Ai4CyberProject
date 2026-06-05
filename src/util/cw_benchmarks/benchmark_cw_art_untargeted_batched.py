import os
import time
import torch
import torch.nn as nn
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
from util.cw_custom import PyTorchCarliniLInf_EarlyStop, PyTorchCarliniLInf_BinarySteps, PyTorchCarliniLInf_AutoTemp

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
            # Funziona nativamente sui batch: active_indices avrà shape (B, K)
            _, self.active_indices = torch.topk(self.model(x_scaled), self.k, dim=1)
            
    def forward(self, x):
        # Gather supporta i batch senza problemi
        return torch.gather(self.model((x * 2.0) - 1.0), 1, self.active_indices)

class FacenetWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x):
        return self.model((x * 2.0) - 1.0)

# ==========================================
# FUNZIONE DI VALUTAZIONE BATCH
# ==========================================
def evaluate_batch(adv_batch, clean_batch, y_true_batch, model_full):
    """
    Calcola la L_inf media e la percentuale di successo (untargeted) per il batch.
    """
    with torch.no_grad():
        preds = torch.argmax(model_full(adv_batch), dim=1)
        # Successo se la predizione è DIVERSA dalla classe originale (untargeted)
        successes = (preds != y_true_batch).sum().item()
        success_rate = (successes / len(y_true_batch)) * 100.0
        
        # Calcolo L-inf media del batch
        diff = torch.abs(adv_batch - clean_batch)
        linfs_per_image = torch.amax(diff, dim=(1, 2, 3)) # L-inf per ogni immagine
        mean_linf = linfs_per_image.mean().item()
        
    return success_rate, mean_linf

# ==========================================
# FUNZIONE DI PLOT COMPARATIVO (Mostra 1 Sample del Batch)
# ==========================================
def plot_benchmark_v4_sample(clean_np, adv_es_np, adv_full_np, adv_auto_np, adv_art_np, out_path):
    noise_es = np.clip(np.abs(adv_es_np - clean_np) * 10.0, 0, 1)
    noise_full = np.clip(np.abs(adv_full_np - clean_np) * 10.0, 0, 1)
    noise_auto = np.clip(np.abs(adv_auto_np - clean_np) * 10.0, 0, 1)
    noise_art = np.clip(np.abs(adv_art_np - clean_np) * 10.0, 0, 1)

    fig, axes = plt.subplots(2, 5, figsize=(25, 10))
    fig.suptitle('Global Benchmark C&W (UNTARGETED) - Esempio Estratto dal Batch (1/10)', fontsize=16)

    axes[0, 0].imshow(np.transpose(clean_np, (1, 2, 0)))
    axes[0, 0].set_title("1. Pulita")
    axes[0, 0].axis('off')

    axes[0, 1].imshow(np.transpose(adv_es_np, (1, 2, 0)))
    axes[0, 1].set_title("2. Custom ES")
    axes[0, 1].axis('off')

    axes[0, 2].imshow(np.transpose(adv_full_np, (1, 2, 0)))
    axes[0, 2].set_title("3. Custom Full")
    axes[0, 2].axis('off')
    
    axes[0, 3].imshow(np.transpose(adv_auto_np, (1, 2, 0)))
    axes[0, 3].set_title("4. Custom Auto-Temp")
    axes[0, 3].axis('off')

    axes[0, 4].imshow(np.transpose(adv_art_np, (1, 2, 0)))
    axes[0, 4].set_title("5. ART")
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
    print(f"\n[!] Grafico comparativo del campione salvato in: {out_path}")

# ==========================================
# MAIN
# ==========================================
def main():
    print("======================================================")
    print(" GLOBAL BENCHMARK C&W UNTARGETED - BATCH 10 IMMAGINI  ")
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
    wrapped_model_spliced = TopKFacenetWrapper(resnet, k=8631).eval()

    df_clean = pd.read_csv(csv_path)

    sample_imgs_list = []
    true_facenet_ids_list = []

    print(f"-> Cerco {BATCH_SIZE} immagini valide nel dataset...")
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
                sample_imgs_list.append(((best_face + 1.0) / 2.0)) # Shape (3, 160, 160)
                true_facenet_ids_list.append(facenet_id)
            
            if len(sample_imgs_list) == BATCH_SIZE:
                break

    if len(sample_imgs_list) < BATCH_SIZE: 
        print(f"[!] Errore: Trovate solo {len(sample_imgs_list)} immagini valide su {BATCH_SIZE} richieste.")
        return

    # Creazione dei tensori Batch
    x_batch = torch.stack(sample_imgs_list).to(device) # Shape: (10, 3, 160, 160)
    y_true_batch = torch.tensor(true_facenet_ids_list, dtype=torch.long, device=device) # Shape: (10,)

    print(f"[OK] Batch di {BATCH_SIZE} immagini creato. Esecuzione attacchi untargeted...\n")

    # Mappiamo gli ID originali negli indici locali per il TopK Wrapper (riga per riga del batch)
    wrapped_model_spliced.freeze_target_classes(x_batch)
    
    local_y_list = []
    for i in range(BATCH_SIZE):
        orig_id = true_facenet_ids_list[i]
        # Cerchiamo l'indice locale per l'i-esima immagine del batch
        local_idx = int(torch.where(wrapped_model_spliced.active_indices[i] == orig_id)[0].item())
        local_y_list.append(local_idx)
        
    local_y_batch = torch.tensor(local_y_list, dtype=torch.long, device=device)

    # --- TEST 1: CUSTOM EARLY STOP (UNTARGETED) ---
    print("-> Esecuzione 1: Custom C&W (Early Stop) sul Batch...")
    attack_es = PyTorchCarliniLInf_EarlyStop(model=wrapped_model_spliced, targeted=False, max_iter=50)
    t0 = time.time()
    adv_es_batch = attack_es.forward(image=x_batch, label=local_y_batch)
    time_es = time.time() - t0
    succ_es, linf_es = evaluate_batch(adv_es_batch, x_batch, y_true_batch, wrapped_model_full)

    # --- TEST 2: CUSTOM FULL OPTIMIZED (UNTARGETED) ---
    print("-> Esecuzione 2: Custom C&W (True SOTA Binary Search) sul Batch...")
    attack_opt = PyTorchCarliniLInf_BinarySteps(model=wrapped_model_spliced, targeted=False, max_iter=50)
    t0 = time.time()
    adv_full_batch = attack_opt.forward(image=x_batch, label=local_y_batch)
    time_full = time.time() - t0
    succ_full, linf_full = evaluate_batch(adv_full_batch, x_batch, y_true_batch, wrapped_model_full)
    
    # --- TEST 3: CUSTOM AUTO TEMP SCALING (UNTARGETED) ---
    print("-> Esecuzione 3: Custom C&W (Auto Temp Scaling + Binary Search) sul Batch...")
    attack_auto = PyTorchCarliniLInf_AutoTemp(model=wrapped_model_spliced, targeted=False, max_iter=50)
    t0 = time.time()
    adv_auto_batch = attack_auto.forward(image=x_batch, label=local_y_batch)
    time_auto = time.time() - t0
    succ_auto, linf_auto = evaluate_batch(adv_auto_batch, x_batch, y_true_batch, wrapped_model_full)

    # --- TEST 4: LIBRERIA ART (UNTARGETED) ---
    print("-> Esecuzione 4: Libreria ART (SOTA) sul Batch...")
    loss_fn = nn.CrossEntropyLoss()
    classifier_art = PyTorchClassifier(model=wrapped_model_full, clip_values=(0.0, 1.0), loss=loss_fn, input_shape=(3, 160, 160), nb_classes=8631, device_type='gpu' if torch.cuda.is_available() else 'cpu')
    
    attack_art = CarliniLInfMethod(classifier=classifier_art, targeted=False, max_iter=50, learning_rate=0.01, initial_const=1e-3, largest_const=20.0)
    x_np_batch = x_batch.cpu().numpy()
    y_true_np_batch = y_true_batch.cpu().numpy()
    
    t0 = time.time()
    adv_art_np_batch = attack_art.generate(x=x_np_batch, y=y_true_np_batch)
    time_art = time.time() - t0
    
    adv_art_tensor_batch = torch.tensor(adv_art_np_batch).to(device)
    succ_art, linf_art = evaluate_batch(adv_art_tensor_batch, x_batch, y_true_batch, wrapped_model_full)

    # --- RISULTATI ---
    print("\n======================================================")
    print(" SOMMARIO DEI RISULTATI BATCH (UNTARGETED - 10 IMG) ")
    print("======================================================")
    print(f"Metodo       | Tempo Tot(s)| L_inf Media | Succ. Rate(%)")
    print(f"Custom (ES)  | {time_es:11.2f} | {linf_es:11.4f} | {succ_es:11.1f}%")
    print(f"Custom (Full)| {time_full:11.2f} | {linf_full:11.4f} | {succ_full:11.1f}%")
    print(f"Custom (Auto)| {time_auto:11.2f} | {linf_auto:11.4f} | {succ_auto:11.1f}%")
    print(f"ART (SOTA)   | {time_art:11.2f} | {linf_art:11.4f} | {succ_art:11.1f}%")
    print("======================================================")

    # Plot comparativo usando la PRIMA immagine del batch (indice 0)
    plot_path = str(out_dir / "global_benchmark_cw_v4_untargeted_batch.png")
    plot_benchmark_v4_sample(
        x_np_batch[0], 
        adv_es_batch.cpu().numpy()[0], 
        adv_full_batch.cpu().numpy()[0], 
        adv_auto_batch.cpu().numpy()[0], 
        adv_art_np_batch[0], 
        plot_path
    )

if __name__ == "__main__":
    main()