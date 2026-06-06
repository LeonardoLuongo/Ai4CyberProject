import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

os.environ["TORCH_CUDNN_V8_API_ENABLED"] = "0" 

import numpy as np
import cv2
import torch
import torch.nn as nn

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

from facenet_pytorch import InceptionResnetV1
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import FastGradientMethod

from util.basic_img.utils import load_rgb_image
from util.plot.utils_plot_shared import (
    plot_adversarial_showcase, 
    plot_frequency_spectrum,
    plot_gradcam_shift,
    plot_umap_trajectory
)

print("Inizio Test REALE (Completo di tutte le 4 metriche)...\n")

# --- 1. SETUP ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
resnet.classify = True 

classifier = PyTorchClassifier(
    model=resnet, clip_values=(0.0, 1.0), loss=nn.CrossEntropyLoss(), optimizer=None,
    input_shape=(3, 160, 160), nb_classes=8631, preprocessing=(0.5, 0.5), 
    device_type='gpu' if torch.cuda.is_available() else 'cpu'
)

# --- 2. PREPARAZIONE DATI ---
img_path = os.path.join("dataset", "clean", "test", "000_n007126_n007126", "0000.jpg")
clean_img_255 = load_rgb_image(img_path, debug=False)
clean_img_160 = cv2.resize(clean_img_255, (160, 160))
x_clean = np.expand_dims(np.transpose(clean_img_160, (2, 0, 1)).astype(np.float32) / 255.0, axis=0)

# --- 3. ATTACCO ---
pred_clean_class = np.argmax(classifier.predict(x_clean), axis=1)[0]
attack = FastGradientMethod(estimator=classifier, eps=0.05)
x_adv = attack.generate(x=x_clean)
pred_adv_class = np.argmax(classifier.predict(x_adv), axis=1)[0]

# Prepara immagini per matplotlib (H, W, C)
x_clean_plot = np.transpose(x_clean[0], (1, 2, 0))
x_adv_plot = np.transpose(x_adv[0], (1, 2, 0))

out_dir = "test_output_plots"
os.makedirs(out_dir, exist_ok=True)

# ==========================================
# TEST 1 & 2: Showcase e Spettro (già visti)
# ==========================================
plot_adversarial_showcase(x_clean_plot, x_adv_plot, f"ID: {pred_clean_class}", f"ID: {pred_adv_class}", True, f"{out_dir}/1_real_showcase.png")
plot_frequency_spectrum(x_clean_plot, x_adv_plot, True, f"{out_dir}/2_real_spectrum.png")


# ==========================================
# TEST 3: Saliency Map (Sostituto di GradCAM)
# ==========================================
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

print("Calcolo mappe Grad-CAM VERE...")

# InceptionResnetV1 ha l'ultimo blocco convoluzionale in 'last_linear' o nei blocchi 'block8'
# Il target layer standard per Facenet PyTorch è resnet.block8
target_layers = [resnet.block8] 

# Inizializza GradCAM
cam = GradCAM(model=resnet, target_layers=target_layers)

# Target: Vogliamo vedere dove la rete guarda per predire quelle specifiche classi
targets_clean = [ClassifierOutputTarget(pred_clean_class)]
targets_adv = [ClassifierOutputTarget(pred_adv_class)]

# Calcola Grad-CAM (vuole tensori normali)
input_tensor_clean = torch.tensor(x_clean * 2 - 1).to(device)
input_tensor_adv = torch.tensor(x_adv * 2 - 1).to(device)

clean_cam = cam(input_tensor=input_tensor_clean, targets=targets_clean)[0, :]
adv_cam = cam(input_tensor=input_tensor_adv, targets=targets_adv)[0, :]

# Ora invia alla tua funzione di plot (che va benissimo così com'è!)
plot_gradcam_shift(x_clean_plot, x_adv_plot, clean_cam, adv_cam, True, f"{out_dir}/3_real_gradcam.png")



# ==========================================
# TEST 4: UMAP Reale con "Mini-Dataset" al volo
# ==========================================
print("\n--- TEST 4: UMAP MULTIPLO ---")
print("Creazione di un mini-dataset al volo (5 immagini)...")

# 1. Creiamo 5 varianti "pulite" dell'immagine base per simulare un dataset
clean_batch_np = []
clean_plot_list = []

# Variante 1: Originale
clean_batch_np.append(x_clean)
clean_plot_list.append(x_clean_plot)

# Variante 2: Riflessa orizzontalmente
x_flipped = np.flip(x_clean, axis=3).copy() 
clean_batch_np.append(x_flipped)
clean_plot_list.append(np.transpose(x_flipped[0], (1, 2, 0)))

# Variante 3: Più chiara
x_bright = np.clip(x_clean + 0.1, 0.0, 1.0)
clean_batch_np.append(x_bright)
clean_plot_list.append(np.transpose(x_bright[0], (1, 2, 0)))

# Variante 4: Più scura
x_dark = np.clip(x_clean - 0.1, 0.0, 1.0)
clean_batch_np.append(x_dark)
clean_plot_list.append(np.transpose(x_dark[0], (1, 2, 0)))

# Variante 5: Leggero rumore fotografico
np.random.seed(42) # Per riproducibilità
x_noisy = np.clip(x_clean + np.random.normal(0, 0.02, x_clean.shape).astype(np.float32), 0.0, 1.0)
clean_batch_np.append(x_noisy)
clean_plot_list.append(np.transpose(x_noisy[0], (1, 2, 0)))

print("Generazione degli attacchi avversari per tutte le 5 immagini...")
adv_batch_np = []
adv_plot_list = []

# 2. Generiamo l'attacco per ogni immagine nel nostro mini-dataset
for i, x_in in enumerate(clean_batch_np):
    x_out_adv = attack.generate(x=x_in)
    adv_batch_np.append(x_out_adv)
    adv_plot_list.append(np.transpose(x_out_adv[0], (1, 2, 0)))

print("Estrazione embeddings a 512-Dimensioni...")
resnet.classify = False 

# 3. Concateniamo per passare tutto alla rete in un colpo solo (Batch processing)
x_clean_all = np.concatenate(clean_batch_np, axis=0) # Shape: (5, 3, 160, 160)
x_adv_all = np.concatenate(adv_batch_np, axis=0)     # Shape: (5, 3, 160, 160)

with torch.no_grad():
    # Ricorda la normalizzazione *2 -1 per facenet_pytorch
    tensor_clean = torch.tensor(x_clean_all * 2 - 1).to(device)
    tensor_adv = torch.tensor(x_adv_all * 2 - 1).to(device)
    
    emb_clean = resnet(tensor_clean).cpu().numpy() # Shape: (5, 512)
    emb_adv = resnet(tensor_adv).cpu().numpy()     # Shape: (5, 512)

print("Esecuzione UMAP e Plotting...")
# 4. Passiamo tutto alla nostra funzione (ora sono 10 punti totali, UMAP funzionerà!)
plot_umap_trajectory(
    clean_embeddings=emb_clean, 
    adv_embeddings=emb_adv, 
    clean_imgs=clean_plot_list, 
    adv_imgs=adv_plot_list,     
    save_flag=True, 
    save_path=f"{out_dir}/4_real_umap_multiple.png"
)

resnet.classify = True 
print(f"\nTUTTE LE METRICHE SONO STATE CALCOLATE! Controlla i 4 file nella cartella '{out_dir}'.")