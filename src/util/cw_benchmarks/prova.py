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

# Assicurati di importare le tue classi
from util.cw_custom import PyTorchCarliniLInf_BinarySteps
from util.cw_benchmarks.cw_pytorch import CarliniLInfMethodPyTorch
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

# Metrica blindata ed esterna per evitare errori di calcolo interni
def evaluate_attack(adv_tensor, clean_tensor, true_id, model_full, max_linf=0.1):
    with torch.no_grad():
        # Calcola la distanza
        linf = float(torch.amax(torch.abs(adv_tensor - clean_tensor)).cpu())
        # Calcola la predizione
        pred = int(torch.argmax(model_full(adv_tensor), dim=1).cpu()[0])
        
        # Un attacco untargeted ha successo SOLO SE cambia classe e resta entro il budget
        is_adv = (pred != true_id)
        is_valid = (linf <= max_linf)
        success = is_adv and is_valid
        
        return linf, success, pred

def plot_benchmark_v5(clean_np, adv_bin_np, adv_full_np, adv_art_np, out_path):
    noise_bin = np.clip(np.abs(adv_bin_np - clean_np) * 10.0, 0, 1)
    noise_full = np.clip(np.abs(adv_full_np - clean_np) * 10.0, 0, 1)
    noise_art = np.clip(np.abs(adv_art_np - clean_np) * 10.0, 0, 1)

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle('C&W UNTARGETED: Binary vs Full vs ART (Budget L_inf < 0.1)', fontsize=16)

    axes[0, 0].imshow(np.transpose(clean_np, (1, 2, 0)))
    axes[0, 0].set_title("1. Originale")
    axes[0, 0].axis('off')

    axes[0, 1].imshow(np.transpose(adv_bin_np, (1, 2, 0)))
    axes[0, 1].set_title("2. Custom (Binary Search)")
    axes[0, 1].axis('off')

    axes[0, 2].imshow(np.transpose(adv_full_np, (1, 2, 0)))
    axes[0, 2].set_title("3. Custom (Full / ART Match)")
    axes[0, 2].axis('off')

    axes[0, 3].imshow(np.transpose(adv_art_np, (1, 2, 0)))
    axes[0, 3].set_title("4. ART (SOTA)")
    axes[0, 3].axis('off')

    axes[1, 0].axis('off') 
    
    axes[1, 1].imshow(np.transpose(noise_bin, (1, 2, 0)))
    axes[1, 1].set_title("Rumore Bin (x10)")
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
    print(f"\n[!] Grafico comparativo salvato in: {out_path}")


def main():
    print("======================================================")
    print(" C&W UNTARGETED: BINARY vs FULL vs ART ")
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

    print(f"[OK] Trovata immagine. True ID: {true_facenet_id}. Vincolo Hacker: L_inf <= 0.1\n")

    wrapped_model_spliced.freeze_target_classes(sample_img_01)
    original_idx_local = int(torch.where(wrapped_model_spliced.active_indices[0] == true_facenet_id)[0].item())
    local_y = torch.tensor([original_idx_local], dtype=torch.long, device=device)

    # --- 1: CUSTOM BINARY ---
    print("-> Esecuzione 1: Custom C&W (Binary Search)...")
    attack_bin = PyTorchCarliniLInf_BinarySteps(model=wrapped_model_spliced, targeted=False, max_iter=50)
    t0 = time.time()
    adv_bin = attack_bin.forward(image=sample_img_01, label=local_y)
    time_bin = time.time() - t0
    linf_bin, succ_bin, pred_bin = evaluate_attack(adv_bin, sample_img_01, true_facenet_id, wrapped_model_full)

    # --- 2: CUSTOM FULL OPTIMIZED (ART MATCH) ---
    print("-> Esecuzione 2: Custom C&W (Full / ART Match)...")
    
    # FIX 1: Wrappare il modello custom dentro un PyTorchClassifier di ART
    # (Usiamo nb_classes=8631 perché il tuo TopKFacenetWrapper è inizializzato con k=8631)
    classifier_spliced = PyTorchClassifier(
        model=wrapped_model_spliced, 
        clip_values=(0.0, 1.0), 
        loss=nn.CrossEntropyLoss(), 
        input_shape=(3, 160, 160), 
        nb_classes=8631, 
        device_type='gpu' if torch.cuda.is_available() else 'cpu'
    )
    
    # Inizializziamo l'attacco passandogli l'estimator di ART
    attack_full = CarliniLInfMethodPyTorch(classifier=classifier_spliced, targeted=False, max_iter=50)
    
    # FIX 2: Passare a numpy per la chiamata a generate()
    x_full_np = sample_img_01.cpu().numpy()
    y_full_np = local_y.cpu().numpy()
    
    t0 = time.time()
    # Usiamo .generate() al posto di .forward()
    adv_full_np = attack_full.generate(x=x_full_np, y=y_full_np)
    time_full = time.time() - t0
    
    # FIX 3: Riconvertire in tensore per le tue funzioni di evaluate_attack e plot
    adv_full = torch.tensor(adv_full_np).to(device)
    
    linf_full, succ_full, pred_full = evaluate_attack(adv_full, sample_img_01, true_facenet_id, wrapped_model_full)

    # --- 3: LIBRERIA ART ---
    print("-> Esecuzione 3: Libreria ART (SOTA)...")
    loss_fn = nn.CrossEntropyLoss()
    classifier_art = PyTorchClassifier(model=wrapped_model_full, clip_values=(0.0, 1.0), loss=loss_fn, input_shape=(3, 160, 160), nb_classes=8631, device_type='gpu' if torch.cuda.is_available() else 'cpu')
    
    attack_art = CarliniLInfMethod(classifier=classifier_art, targeted=False, max_iter=50, learning_rate=0.01, initial_const=1e-3, largest_const=20.0)
    x_np = sample_img_01.cpu().numpy()
    y_true_np = np.array([true_facenet_id])
    
    t0 = time.time()
    adv_art_np = attack_art.generate(x=x_np, y=y_true_np)
    time_art = time.time() - t0
    adv_art_tensor = torch.tensor(adv_art_np).to(device)
    linf_art, succ_art, pred_art = evaluate_attack(adv_art_tensor, sample_img_01, true_facenet_id, wrapped_model_full)

    # --- RISULTATI ---
    print("\n======================================================")
    print(" RISULTATI C&W (Budget Massimo L_inf <= 0.1) ")
    print("======================================================")
    print(f"{'Metodo':<14} | {'Tempo (s)':<9} | {'L_inf Dist':<10} | {'Successo'}")
    print("-" * 54)
    print(f"{'Custom (Bin)':<14} | {time_bin:9.2f} | {linf_bin:10.4f} | {succ_bin} (ID: {pred_bin})")
    print(f"{'Custom (Full)':<14} | {time_full:9.2f} | {linf_full:10.4f} | {succ_full} (ID: {pred_full})")
    print(f"{'ART (SOTA)':<14} | {time_art:9.2f} | {linf_art:10.4f} | {succ_art} (ID: {pred_art})")
    print("======================================================")

    plot_path = str(out_dir / "benchmark_cw_v5_strict.png")
    plot_benchmark_v5(x_np[0], adv_bin.cpu().numpy()[0], adv_full.cpu().numpy()[0], adv_art_np[0], plot_path)

if __name__ == "__main__":
    main()