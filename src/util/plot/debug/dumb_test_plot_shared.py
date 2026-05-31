import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import os
from util.plot.utils_plot_shared import (
    plot_adversarial_showcase, 
    plot_frequency_spectrum, 
    plot_umap_trajectory, 
    plot_gradcam_shift
)

# Creiamo una cartella di test temporanea
out_dir = "test_output_plots"
os.makedirs(out_dir, exist_ok=True)


print("Inizio Test Modulo di Plotting Condiviso...\n")

# 1. Creiamo un'immagine "Clean" finta (un gradiente grigio)
clean_img = np.tile(np.linspace(0.2, 0.8, 224), (224, 1)).T
clean_img = np.stack([clean_img]*3, axis=-1) # Immagine RGB finta

# 2. Creiamo del finto rumore avversario (rumore ad alta frequenza)
np.random.seed(42)
fake_noise = np.random.normal(0, 0.05, (224, 224, 3))
adv_img = np.clip(clean_img + fake_noise, 0, 1)

# TEST 1: Showcase
print("Test 1: Generazione Adversarial Showcase...")
plot_adversarial_showcase(
    clean_img, adv_img, 
    true_label_name="n007126 (ID: 0)", 
    adv_label_name="n004952 (ID: 3)", 
    save_flag=True,  # ORA E' TRUE!
    save_path=f"{out_dir}/1_showcase.png"
)

# TEST 2: Frequenze (FFT)
print("Test 2: Generazione e salvataggio Frequency Spectrum...")
plot_frequency_spectrum(fake_noise, save_flag=True, save_path=f"{out_dir}/2_frequency.png")

# TEST 3: GradCAM 
print("Test 3: Generazione e salvataggio GradCAM Shift...")
clean_cam = np.exp(-((np.linspace(-1, 1, 224)[:, None]**2) + (np.linspace(-1, 1, 224)[None, :]**2)) / 0.1)
# Finta attenzione spostata in alto a sinistra per l'avversario
adv_cam = np.exp(-((np.linspace(-1, 1, 224)[:, None] + 0.8)**2 + (np.linspace(-1, 1, 224)[None, :] + 0.8)**2) / 0.1)
plot_gradcam_shift(clean_img, adv_img, clean_cam, adv_cam, save_flag=True, save_path=f"{out_dir}/3_gradcam.png")

# TEST 4: UMAP Trajectory
print("Test 4: Generazione e salvataggio UMAP Trajectory...")
fake_clean_embeds = np.random.randn(3, 512)
# L'attacco sposta brutalmente gli embeddings in una direzione
fake_adv_embeds = fake_clean_embeds + np.random.randn(3, 512) * 5 
fake_labels = ["n007126", "n005986", "n004933"]
plot_umap_trajectory(fake_clean_embeds, fake_adv_embeds, fake_labels, save_flag=True, save_path=f"{out_dir}/4_umap.png")

print("\nTutti i test grafici sono stati eseguiti con successo! Controlla la cartella 'test_output_plots'.")