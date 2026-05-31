# FILE: src/util/plot/utils_plot_shared.py

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import seaborn as sns
import cv2
import umap

# Impostazioni estetiche globali per grafici in stile accademico
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

def save_or_show(save_flag: bool, save_path: str):
    """Utility interna per gestire il salvataggio o la visualizzazione."""
    if save_flag and save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"[PLOT] Grafico salvato in: {save_path}")
    else:
        plt.show()
    plt.close()

def plot_adversarial_showcase(clean_img: np.ndarray, adv_img: np.ndarray, 
                              true_label_name: str, adv_label_name: str, 
                              save_flag: bool = False, save_path: str = None):
    """
    Mostra l'immagine originale, il rumore amplificato e l'immagine avversaria.
    Si aspetta immagini in formato RGB con valori [0, 1] o [0, 255].
    """
    # Assicuriamoci che i valori siano float per il calcolo del rumore
    clean_img_f = clean_img.astype(np.float32) / 255.0 if clean_img.max() > 1.0 else clean_img.copy()
    adv_img_f = adv_img.astype(np.float32) / 255.0 if adv_img.max() > 1.0 else adv_img.copy()

    # Calcolo del rumore e amplificazione per visibilità
    amp_fact = 4
    amp_coeff = 0.0
    noise = adv_img_f - clean_img_f
    noise_visual = np.clip((noise * amp_fact) + amp_coeff, 0, 1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 1. Originale
    axes[0].imshow(clean_img_f)
    axes[0].set_title(f"Clean Image\nPred: {true_label_name}", color='green')
    axes[0].axis('off')

    # 2. Rumore Amplificato
    axes[1].imshow(noise_visual)
    axes[1].set_title(f"Perturbation (noise x {amp_fact}) + {amp_coeff}\nMax L_inf: {np.max(np.abs(noise)):.4f}")
    axes[1].axis('off')

    # 3. Avversario
    axes[2].imshow(adv_img_f)
    color = 'red' if true_label_name != adv_label_name else 'green'
    axes[2].set_title(f"Adversarial Image\nPred: {adv_label_name}", color=color)
    axes[2].axis('off')

    plt.suptitle("Adversarial Attack Showcase", fontsize=16, y=1.05)
    save_or_show(save_flag, save_path)


def plot_frequency_spectrum(clean_img: np.ndarray, adv_img: np.ndarray, save_flag: bool = False, save_path: str = None):
    def get_magnitude_spectrum(img):
        img_gray = np.mean(img, axis=2) if len(img.shape) == 3 else img
        f_shift = np.fft.fftshift(np.fft.fft2(img_gray))
        # log(1 + abs) evita log(0) e appiattisce i picchi per visualizzazione
        return 20 * np.log(np.abs(f_shift) + 1) 

    mag_clean = get_magnitude_spectrum(clean_img)
    mag_adv = get_magnitude_spectrum(adv_img)
    
    # Calcoliamo la differenza nelle frequenze
    mag_diff = np.abs(mag_adv - mag_clean)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    im0 = axes[0].imshow(mag_clean, cmap='magma')
    axes[0].set_title("Clean FFT")
    axes[0].axis('off')

    im1 = axes[1].imshow(mag_adv, cmap='magma')
    axes[1].set_title("Adversarial FFT")
    axes[1].axis('off')

    im2 = axes[2].imshow(mag_diff, cmap='hot') # Mappa 'hot' per evidenziare dove cambia
    axes[2].set_title("Frequency Perturbation (Diff)")
    axes[2].axis('off')

    plt.colorbar(im2, ax=axes, fraction=0.015, pad=0.04, label="Magnitude Diff (dB)")
    plt.suptitle("Frequency Domain Analysis", fontsize=16)
    
    save_or_show(save_flag, save_path)



def plot_umap_trajectory(clean_embeddings: np.ndarray, adv_embeddings: np.ndarray, 
                         clean_imgs: list, adv_imgs: list,
                         save_flag: bool = False, save_path: str = None):
    """
    Usa UMAP per proiettare gli embeddings da 512D a 2D e posiziona le miniature
    delle immagini reali sulle coordinate calcolate, tracciando una freccia.
    """
    num_samples = len(clean_embeddings)
    
    # Uniamo gli embeddings per l'addestramento UMAP
    combined_embeddings = np.vstack((clean_embeddings, adv_embeddings))
    
    print("[UMAP] Calcolo della proiezione in 2D in corso...")
    # NOTA: n_neighbors deve essere <= al numero totale di campioni. 
    # Se passi solo 1 immagine pulita e 1 avversaria (2 campioni totali), forziamo n_neighbors a 2.
    neighbors = min(15, len(combined_embeddings)) 
    if neighbors < 2: neighbors = 2
    
    reducer = umap.UMAP(n_neighbors=neighbors, min_dist=0.1, random_state=42)
    embedding_2d = reducer.fit_transform(combined_embeddings)
    
    clean_2d = embedding_2d[:num_samples]
    adv_2d = embedding_2d[num_samples:]

    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Per impostare dinamicamente i limiti degli assi (indispensabile quando si disegnano immagini invece di punti)
    min_x, max_x = np.min(embedding_2d[:, 0]), np.max(embedding_2d[:, 0])
    min_y, max_y = np.min(embedding_2d[:, 1]), np.max(embedding_2d[:, 1])
    padding_x = (max_x - min_x) * 0.2 if max_x != min_x else 1.0
    padding_y = (max_y - min_y) * 0.2 if max_y != min_y else 1.0
    
    ax.set_xlim(min_x - padding_x, max_x + padding_x)
    ax.set_ylim(min_y - padding_y, max_y + padding_y)

    # Disegniamo frecce, immagini pulite e immagini avversarie
    for i in range(num_samples):
        # 1. Disegna la freccia
        ax.annotate("", xy=(adv_2d[i, 0], adv_2d[i, 1]), xytext=(clean_2d[i, 0], clean_2d[i, 1]),
                    arrowprops=dict(arrowstyle="->", color="gray", lw=2, alpha=0.7))

        # 2. Crea il box per l'immagine pulita (Bordo Verde)
        img_clean_box = OffsetImage(clean_imgs[i], zoom=0.15) # Regola lo zoom se l'immagine è troppo grande/piccola
        ab_clean = AnnotationBbox(img_clean_box, (clean_2d[i, 0], clean_2d[i, 1]), 
                                  frameon=True, bboxprops=dict(edgecolor='green', lw=3))
        ax.add_artist(ab_clean)

        # 3. Crea il box per l'immagine avversaria (Bordo Rosso)
        img_adv_box = OffsetImage(adv_imgs[i], zoom=0.15)
        ab_adv = AnnotationBbox(img_adv_box, (adv_2d[i, 0], adv_2d[i, 1]), 
                                frameon=True, bboxprops=dict(edgecolor='red', lw=3))
        ax.add_artist(ab_adv)

    plt.title("Latent Space Trajectory (UMAP with Real Images)", fontsize=14)
    
    # Rimuoviamo i numeri dagli assi (come suggerito prima, non hanno significato in UMAP)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    
    save_or_show(save_flag, save_path)



def plot_gradcam_shift(clean_img: np.ndarray, adv_img: np.ndarray, 
                       clean_cam: np.ndarray, adv_cam: np.ndarray, 
                       save_flag: bool = False, save_path: str = None):
    """
    Sovrappone le mappe di attivazione (Saliency/GradCAM) sulle immagini.
    Aggiunge una colonna centrale con la differenza assoluta dell'attenzione.
    """
    def overlay_cam(img, mask, colormap=cv2.COLORMAP_JET):
        # Normalizziamo l'immagine tra 0 e 255 (uint8) se non lo è
        img_uint8 = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
        
        # Applichiamo colormap alla maschera
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Sovrapposizione
        return cv2.addWeighted(img_uint8, 0.6, heatmap, 0.4, 0)

    # 1. Mappa Originale (Usa JET)
    over_clean = overlay_cam(clean_img, clean_cam, cv2.COLORMAP_JET)
    
    # 2. Mappa Avversaria (Usa JET)
    over_adv = overlay_cam(adv_img, adv_cam, cv2.COLORMAP_JET)
    
    # 3. Mappa Differenza
    # Calcoliamo quanto è cambiata l'attenzione (valore assoluto)
    cam_diff = np.abs(adv_cam - clean_cam)
    # Normalizziamo la differenza tra 0 e 1 per sicurezza
    if cam_diff.max() > 0:
        cam_diff = cam_diff / cam_diff.max()
    # Sovrapponiamo la differenza sull'immagine pulita usando HOT (Nero -> Rosso -> Giallo/Bianco)
    over_diff = overlay_cam(clean_img, cam_diff, cv2.COLORMAP_HOT)

    # Setup del plot a 3 colonne
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(over_clean)
    axes[0].set_title("Clean Attention")
    axes[0].axis('off')

    axes[1].imshow(over_diff)
    axes[1].set_title("Attention Shift (Difference)")
    axes[1].axis('off')

    axes[2].imshow(over_adv)
    axes[2].set_title("Adversarial Attention")
    axes[2].axis('off')

    plt.suptitle("Explainable AI: Attention Shift Analysis", fontsize=16, y=1.05)
    save_or_show(save_flag, save_path)
