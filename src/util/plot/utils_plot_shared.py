# FILE: src/util/plot/utils_plot_shared.py

import os
import cv2
import umap
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox


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
    amp_fact = 30.0
    
    # FIX: Aggiunto np.abs() per prendere il valore assoluto della perturbazione!
    noise = np.abs(adv_img_f - clean_img_f)
    noise_visual = np.clip(noise * amp_fact, 0, 1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 1. Originale
    axes[0].imshow(clean_img_f)
    axes[0].set_title(f"Clean Image\nPred: {true_label_name}", color='green')
    axes[0].axis('off')

    # 2. Rumore Amplificato
    axes[1].imshow(noise_visual)
    axes[1].set_title(f"Perturbation (noise x {int(amp_fact)})\nMax L_inf: {np.max(noise):.4f}")
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
    Usa una colormap divergente per la differenza: Rosso = Attenzione Guadagnata, Blu = Persa.
    """
    def overlay_cam(img, mask, colormap=cv2.COLORMAP_JET):
        img_uint8 = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        return cv2.addWeighted(img_uint8, 0.6, heatmap, 0.4, 0)

    def overlay_diverging_diff(img, diff_mask):
        min_val = np.min(diff_mask) # e.g., -0.219 (Blu)
        max_val = np.max(diff_mask) # e.g., +0.076 (Rosso)

        print(f"  -> [GradCAM] Max perdita (Blu): {min_val:.3f}")
        print(f"  -> [GradCAM] Max guadagno (Rosso): {max_val:.3f}")

        # Inizializziamo la maschera a 0
        scaled_diff = np.zeros_like(diff_mask)

        # Scaliamo il Blu in modo indipendente (da -1 a 0)
        if min_val < 0:
            scaled_diff[diff_mask < 0] = diff_mask[diff_mask < 0] / abs(min_val)
            
        # Scaliamo il Rosso in modo indipendente (da 0 a 1)
        if max_val > 0:
            scaled_diff[diff_mask > 0] = diff_mask[diff_mask > 0] / max_val

        norm_diff = (scaled_diff + 1.0) / 2.0
        cmap = plt.get_cmap('bwr')
        heatmap_rgb = cmap(norm_diff)[:, :, :3]
        
        # Calcoliamo l'opacità
        alpha_mask = np.abs(scaled_diff)
        
        # THRESHOLD: Eliminiamo il rumore di fondo per ripristinare la nitidezza!
        # Qualsiasi variazione sotto il 20% della forza massima viene resa invisibile.
        alpha_mask[alpha_mask < 0.2] = 0.0 
        
        # Opacità massima all'80% per far vedere i tratti somatici sotto il colore acceso
        alpha_mask = alpha_mask[..., np.newaxis] * 0.8 
        
        img_float = img.astype(np.float32) / 255.0 if img.max() > 1.0 else img.astype(np.float32)
        blended = (1.0 - alpha_mask) * img_float + alpha_mask * heatmap_rgb
        return blended

    # Mappe originali
    over_clean = overlay_cam(clean_img, clean_cam, cv2.COLORMAP_JET)
    over_adv = overlay_cam(clean_img, adv_cam, cv2.COLORMAP_JET)
    
    # Calcolo della differenza REALE (segno + e -)
    cam_diff = adv_cam - clean_cam 
    
    # Sovrapposizione divergente
    over_diff = overlay_diverging_diff(clean_img, cam_diff)

    # Rendering del plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(over_clean)
    axes[0].set_title("Clean Attention")
    axes[0].axis('off')

    axes[1].imshow(over_diff)
    axes[1].set_title("Attention Shift\n(Red = Gained, Blue = Lost)")
    axes[1].axis('off')

    axes[2].imshow(over_adv)
    axes[2].set_title("Adversarial Attention")
    axes[2].axis('off')

    plt.suptitle("Explainable AI: Attention Shift Analysis", fontsize=16, y=1.05)
    save_or_show(save_flag, save_path)



def plot_targeted_umap_trajectory(bg_emb: np.ndarray, bg_labels: np.ndarray, 
                                  src_clean_emb: np.ndarray, src_adv_emb: np.ndarray, 
                                  src_label_name: str, target_pred_name: str,
                                  save_flag: bool = False, save_path: str = None):
    """
    Disegna l'area di decisione 'sicura' di un'identità (Verde) e mostra come 
    i punti avversari (Rossi) vengano spinti fuori verso altre regioni (Grigio).
    """
    import umap
    import matplotlib.patches as mpatches

    print("[UMAP] Mappatura Topologica in 2D...")
    # Uniamo tutto per definire lo spazio topologico
    all_emb = np.vstack([bg_emb, src_clean_emb, src_adv_emb])
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    all_2d = reducer.fit_transform(all_emb)
    
    bg_2d = all_2d[:len(bg_emb)]
    src_clean_2d = all_2d[len(bg_emb) : len(bg_emb)+len(src_clean_emb)]
    src_adv_2d = all_2d[len(bg_emb)+len(src_clean_emb):]
    
    plt.figure(figsize=(12, 8))
    ax = plt.gca()
    
    # 1. Disegniamo le Regioni di Sfondo (Altre Identità)
    unique_bg = np.unique(bg_labels)
    for bg_lbl in unique_bg:
        idx = bg_labels == bg_lbl
        sns.kdeplot(x=bg_2d[idx, 0], y=bg_2d[idx, 1], fill=True, color="gray", alpha=0.15, thresh=0.1, ax=ax)
    
    # 2. Disegniamo la Safe Zone (L'identità originale)
    sns.kdeplot(x=src_clean_2d[:, 0], y=src_clean_2d[:, 1], fill=True, color="forestgreen", alpha=0.4, thresh=0.05, ax=ax)
    plt.scatter(src_clean_2d[:, 0], src_clean_2d[:, 1], c='forestgreen', edgecolor='white', s=100, label='Clean Images (Original)')
    
    # 3. Disegniamo i punti Avversari
    plt.scatter(src_adv_2d[:, 0], src_adv_2d[:, 1], c='firebrick', marker='X', edgecolor='white', s=150, label='Adversarial Images')
    
    # 4. Tracciamo le Frecce (La Traiettoria)
    for i in range(len(src_clean_2d)):
        plt.arrow(src_clean_2d[i, 0], src_clean_2d[i, 1], 
                  src_adv_2d[i, 0] - src_clean_2d[i, 0], src_adv_2d[i, 1] - src_clean_2d[i, 1],
                  color='black', alpha=0.3, head_width=0.2, length_includes_head=True)
                  
    plt.title(f"Latent Space Escape Trajectory\n(Forcing {src_label_name} out of its Decision Region)", fontsize=16, pad=15)
    
    # 5. Creazione della Legenda Accademica
    bg_patch = mpatches.Patch(color='gray', alpha=0.15, label='Foreign Regions (Other Identities)')
    safe_patch = mpatches.Patch(color='forestgreen', alpha=0.4, label=f'Safe Zone ({src_label_name})')
    
    handles, labels = ax.get_legend_handles_labels()
    handles.extend([bg_patch, safe_patch])
    
    # Evitiamo duplicati nella legenda
    by_label = dict(zip(labels, handles))
    
    plt.legend(by_label.values(), by_label.keys(), loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=12)
    plt.xticks([]); plt.yticks([])
    plt.tight_layout()
    
    save_or_show(save_flag, save_path)

def plot_latent_trajectory(bg_emb: np.ndarray, bg_labels: list, 
                           src_clean_emb: np.ndarray, src_adv_emb: np.ndarray, 
                           src_label_name: str, 
                           adv_success_flags: np.ndarray,
                           adv_target_names: list,
                           adv_actual_pred_names: list,
                           tgt_clean_emb: np.ndarray = None, 
                           save_flag: bool = False, save_path: str = None):
    """
    UMAP Trajectory pulito e accademico.
    - Testi spostati nella Legenda.
    - Sfondi con colori tenui distinti.
    - Numerazione dinamica su X e O per mappare i Target specifici di ogni immagine.
    """
    import umap
    import matplotlib.patches as mpatches
    from matplotlib.colors import to_rgba

    print("[UMAP] Mappatura Topologica in 2D...")
    all_emb_list = [bg_emb, src_clean_emb, src_adv_emb]
    if tgt_clean_emb is not None:
        all_emb_list.append(tgt_clean_emb)
        
    all_emb = np.vstack(all_emb_list)
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    all_2d = reducer.fit_transform(all_emb)
    
    bg_2d = all_2d[:len(bg_emb)]
    src_clean_2d = all_2d[len(bg_emb) : len(bg_emb)+len(src_clean_emb)]
    src_adv_2d = all_2d[len(bg_emb)+len(src_clean_emb) : len(bg_emb)+len(src_clean_emb)+len(src_adv_emb)]
    if tgt_clean_emb is not None:
        tgt_clean_2d = all_2d[-len(tgt_clean_emb):]
    
    plt.figure(figsize=(14, 10))
    ax = plt.gca()
    
    legend_handles = []
    
    # 1. Disegniamo le Regioni di Sfondo (Colori pastello)
    unique_bg = list(set(bg_labels))
    bg_palette = sns.color_palette("Pastel1", n_colors=len(unique_bg))
    
    for i, bg_lbl in enumerate(unique_bg):
        # Troviamo gli indici di chi appartiene a questa etichetta
        idx = [j for j, val in enumerate(bg_labels) if val == bg_lbl]
        if len(idx) > 0:
            sns.kdeplot(x=bg_2d[idx, 0], y=bg_2d[idx, 1], fill=True, color=bg_palette[i], alpha=0.3, thresh=0.1, ax=ax)
            legend_handles.append(mpatches.Patch(color=bg_palette[i], alpha=0.3, label=f"BG: {bg_lbl}"))
            
    # 2. Disegniamo l'Area Target Reale (Se abbiamo le foto)
    if tgt_clean_emb is not None:
        sns.kdeplot(x=tgt_clean_2d[:, 0], y=tgt_clean_2d[:, 1], fill=True, color="firebrick", alpha=0.2, thresh=0.05, ax=ax)
        legend_handles.append(mpatches.Patch(color='firebrick', alpha=0.2, label="Known Target Region"))
    
    # 3. Disegniamo la Safe Zone Originale (Verde)
    sns.kdeplot(x=src_clean_2d[:, 0], y=src_clean_2d[:, 1], fill=True, color="forestgreen", alpha=0.3, thresh=0.05, ax=ax)
    plt.scatter(src_clean_2d[:, 0], src_clean_2d[:, 1], c='forestgreen', edgecolor='white', s=80, zorder=3)
    legend_handles.append(mpatches.Patch(color='forestgreen', alpha=0.3, label=f"Safe Zone ({src_label_name})"))
    
    # 4. Tracciamo Traiettorie, Punti Avversari e Numerini
    for i in range(len(src_clean_2d)):
        is_success = adv_success_flags[i]
        tgt_name = adv_target_names[i]
        pred_name = adv_actual_pred_names[i]
        
        # Freccia
        plt.arrow(src_clean_2d[i, 0], src_clean_2d[i, 1], 
                  src_adv_2d[i, 0] - src_clean_2d[i, 0], src_adv_2d[i, 1] - src_clean_2d[i, 1],
                  color='black', alpha=0.3, head_width=0.15, length_includes_head=True, zorder=1)
        
        # Disegno Punto (X o O)
        if is_success:
            marker_style = dict(marker='X', color='firebrick', s=200, edgecolor='white', linewidth=1)
            leg_lbl = f"[{i+1}] Hit Target: {tgt_name}"
            
            # Alone geometrico 2D per simulare l'approssimazione (Solo se non c'è il KDE rosso di sfondo)
            if tgt_clean_emb is None:
                circle = plt.Circle((src_adv_2d[i, 0], src_adv_2d[i, 1]), radius=0.4, 
                                    color='firebrick', alpha=0.15, zorder=2)
                ax.add_patch(circle)
        else:
            marker_style = dict(marker='o', facecolors='none', edgecolors='firebrick', s=150, linewidths=2.5)
            leg_lbl = f"[{i+1}] Miss! Aimed {tgt_name} \u2192 Landed on {pred_name}"
            
        ax.scatter(src_adv_2d[i, 0], src_adv_2d[i, 1], zorder=4, label=leg_lbl, **marker_style)
        
        # Numerino spostato in alto a destra rispetto al punto
        ax.text(src_adv_2d[i, 0] + 0.15, src_adv_2d[i, 1] + 0.15, str(i+1), 
                color='black', fontsize=10, fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1), zorder=5)

    plt.title(f"Latent Space Escape Trajectory\n(Attacking {src_label_name})", fontsize=16, pad=15)
    
    # 5. Generazione Legenda Combinata
    handles, labels = ax.get_legend_handles_labels()
    # Combiniamo le patches di sfondo create all'inizio con i marker dei punti generati dal loop
    all_handles = legend_handles + handles
    all_labels = [h.get_label() for h in legend_handles] + labels
    
    # Evitiamo duplicati (se più punti hanno la stessa label esatta)
    by_label = dict(zip(all_labels, all_handles))
    
    plt.legend(by_label.values(), by_label.keys(), loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=10)
    plt.xticks([]); plt.yticks([])
    plt.tight_layout()
    save_or_show(save_flag, save_path)


def plot_round_robin_plotly_grouped(clean_emb: np.ndarray, clean_labels: np.ndarray, 
                                    adv_emb: np.ndarray, adv_success_flags: np.ndarray,
                                    adv_source_labels: np.ndarray, adv_target_labels: np.ndarray,
                                    save_path: str = "umap_interactive_grouped.html"):
    
    import umap
    import plotly.graph_objects as go
    import plotly.colors as pcolors
    import numpy as np
    from scipy.spatial import ConvexHull
    from sklearn.cluster import DBSCAN

    print("[UMAP] Generazione HTML interattivo con Smart Clustering...")
    
    # Fit UMAP
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.3, random_state=42)
    clean_2d = reducer.fit_transform(clean_emb)
    adv_2d = reducer.transform(adv_emb)
    
    fig = go.Figure()
    
    unique_identities = np.unique(clean_labels)
    colors = pcolors.qualitative.Alphabet 
    color_map = {ident: colors[i % len(colors)] for i, ident in enumerate(unique_identities)}

    # --- FUNZIONE DI SUPPORTO PER IL RAGGRUPPAMENTO ---
    def aggregate_points(pts_2d, targets, eps=0.4):
        """Raggruppa punti spazialmente vicini e conta quanti sono."""
        if len(pts_2d) == 0:
            return [], [], []
        
        centroids, grouped_targets, counts = [], [], []
        # Clustering spaziale per trovare i punti vicini
        clustering = DBSCAN(eps=eps, min_samples=1).fit(pts_2d)
        
        for cluster_id in np.unique(clustering.labels_):
            mask = (clustering.labels_ == cluster_id)
            c_pts = pts_2d[mask]
            c_tgts = targets[mask]
            
            # Calcoliamo il centro del gruppo
            centroids.append(np.mean(c_pts, axis=0))
            counts.append(len(c_pts))
            
            # Se colpiscono target diversi, li uniamo in una stringa per il pop-up
            unique_t = np.unique(c_tgts)
            if len(unique_t) == 1:
                grouped_targets.append(unique_t[0])
            else:
                grouped_targets.append("Target Multipli: " + ", ".join(unique_t))
                
        return np.array(centroids), np.array(grouped_targets), np.array(counts)

    # --- 1. DISEGNIAMO LE AREE E I PUNTI PULITI ---
    for ident in unique_identities:
        idx = (clean_labels == ident)
        if not np.any(idx): continue
        pts = clean_2d[idx]
        
        # Contorno ConvexHull
        if len(pts) >= 3:
            hull = ConvexHull(pts)
            hull_pts = np.append(hull.vertices, hull.vertices[0])
            fig.add_trace(go.Scatter(
                x=pts[hull_pts, 0], y=pts[hull_pts, 1],
                mode='lines', fill='toself', fillcolor=color_map[ident], 
                opacity=0.15, line=dict(color=color_map[ident], width=2, dash='dot'),
                name=f"Area {ident}", legendgroup=ident, showlegend=False, hoverinfo='skip'
            ))

        # Punti Puliti
        fig.add_trace(go.Scatter(
            x=pts[:, 0], y=pts[:, 1],
            mode='markers', marker=dict(color=color_map[ident], size=6, opacity=0.5, line=dict(width=0)),
            name=f"{ident} (Clean)", legendgroup=ident, hoverinfo='text', text=f"Clean Area: {ident}"
        ))
        
        # Etichetta identità pulita
        cx, cy = np.median(pts[:, 0]), np.median(pts[:, 1])
        fig.add_annotation(
            x=cx, y=cy, text=f"<b>{ident}</b>", showarrow=False,
            font=dict(size=11, color="white"),
            bgcolor="#2b2b2b", bordercolor=color_map[ident], borderwidth=1.5, opacity=0.9
        )

    # --- 2. DISEGNIAMO GLI ATTACCHI AGGREGATI ---
    for ident in unique_identities:
        # Maschere per Hit e Miss
        hits_idx = (adv_source_labels == ident) & adv_success_flags
        miss_idx = (adv_source_labels == ident) & ~adv_success_flags
        
        # --- Aggregazione e Plot HIT ---
        if np.any(hits_idx):
            h_cents, h_tgts, h_counts = aggregate_points(adv_2d[hits_idx], adv_target_labels[hits_idx])
            
            hover_texts = [f"<b>Sorgente:</b> {ident}<br><b>Target:</b> {tgt}<br><b>Esito:</b> <span style='color:#00ff00'>HIT</span><br><b>Attacchi qui:</b> {cnt}" 
                           for tgt, cnt in zip(h_tgts, h_counts)]
            
            # Testo da mostrare DIRETTAMENTE sul grafico (vuoto se è 1, numero se >1)
            marker_texts = [str(c) if c > 1 else "" for c in h_counts]
            
            fig.add_trace(go.Scatter(
                x=h_cents[:, 0], y=h_cents[:, 1],
                mode='markers+text', # Abilitiamo il testo sui marker
                text=marker_texts,
                textposition="middle center",
                textfont=dict(color='black', size=9, family="Arial Black"),
                marker=dict(symbol='x', 
                            size=[10 + (np.log(c)*5) for c in h_counts], # Il marker cresce leggermente col logaritmo
                            color=color_map[ident], line=dict(width=1.5, color='white')),
                name=f"  ↳ Hit (da {ident})", legendgroup=ident, hoverinfo='text', hovertext=hover_texts
            ))
            
        # --- Aggregazione e Plot MISS ---
        if np.any(miss_idx):
            m_cents, m_tgts, m_counts = aggregate_points(adv_2d[miss_idx], adv_target_labels[miss_idx])
            
            hover_texts = [f"<b>Sorgente:</b> {ident}<br><b>Target:</b> {tgt}<br><b>Esito:</b> <span style='color:#ff4444'>MISS</span><br><b>Attacchi qui:</b> {cnt}" 
                           for tgt, cnt in zip(m_tgts, m_counts)]
            
            marker_texts = [str(c) if c > 1 else "" for c in m_counts]
            
            fig.add_trace(go.Scatter(
                x=m_cents[:, 0], y=m_cents[:, 1],
                mode='markers+text',
                text=marker_texts,
                textposition="middle center",
                textfont=dict(color='white', size=9, family="Arial Black"),
                marker=dict(symbol='circle-open', 
                            size=[9 + (np.log(c)*4) for c in m_counts], 
                            line=dict(color=color_map[ident], width=2.5)),
                name=f"  ↳ Miss (da {ident})", legendgroup=ident, hoverinfo='text', hovertext=hover_texts
            ))

    # --- 3. LAYOUT DARK MODE ---
    fig.update_layout(
        title="Latent Space Topography (Round-Robin 10x10) - Clustered",
        title_x=0.5, title_font_size=20, autosize=True, template="plotly_dark",
        margin=dict(l=40, r=40, t=80, b=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, scaleanchor="y", scaleratio=1),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        legend=dict(title="Filtra per Identità", itemsizing='constant', tracegroupgap=3, yanchor="top", y=1, xanchor="left", x=1.02),
        hovermode="closest", plot_bgcolor='#121212', paper_bgcolor='#121212'
    )
    
    fig.write_html(save_path)
    print(f"[UMAP] File interattivo salvato in: {save_path}")