import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from PIL import Image

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)


from facenet_pytorch import InceptionResnetV1, MTCNN
from art.estimators.classification import PyTorchClassifier

# Importiamo la tua intuizione geniale (Il Wrapper compatibile ART)
from util.deepfool_wrapped import ARTCompatibleDeepFool
from util.identity_mapper import IdentityMapper

def main():
    print("======================================================")
    print(" SCOUTING HYPER-PARAMS: DEEPFOOL (Error-Generic)      ")
    print("======================================================\n")

    # ==========================================
    # 1. SETUP PATH E PARAMETRI
    # ==========================================
    base_dir = Path.cwd()
    csv_path = base_dir / "dataset" / "clean" / "splits" / "manifest.csv"
    meta_csv_path = base_dir / "dataset" / "clean" / "splits" / "identity_meta.csv"
    output_dir = base_dir / "plots" / "3_Adversarial_Examples" / "error_generic" / "deepfool"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    txt_log_path = output_dir / "deepfool_scouting_report.txt"

    BUDGET_LINF = 0.10
    SAMPLES_PER_ID = 10  # 1 immagine per persona = 100 immagini testate
    
    # Parametri DeepFool (Ora usano i nomi in stile ART per il Wrapper)
    overshoots = [0.25, 0.30]
    max_iters_list = [5, 20, 40]
    
    NB_GRADS = 3 # Lasciamo fisso a 10 per congelare il mini-universo (Velocizza enormemente senza perdere efficacia)

    # ==========================================
    # 2. INIZIALIZZAZIONE RETI
    # ==========================================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"-> Inizializzazione Reti su {device}...")
    
    mapper = IdentityMapper(meta_csv_path)
    mtcnn = MTCNN(image_size=160, margin=0, keep_all=True, post_process=True, device=device)
    
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    resnet.classify = True 
    
    # Il Classifier Ufficiale ART
    classifier = PyTorchClassifier(
        model=resnet, clip_values=(0.0, 1.0), loss=nn.CrossEntropyLoss(), optimizer=None,
        input_shape=(3, 160, 160), nb_classes=8631, preprocessing=(0.5, 0.5), 
        device_type='gpu' if torch.cuda.is_available() else 'cpu'
    )

    df_clean = pd.read_csv(csv_path)

    # ==========================================
    # 3. PRE-FILTRAGGIO E CAMPIONAMENTO 
    # ==========================================
    print(f"\n[FASE 1] Estrazione chirurgica e Campionamento ({SAMPLES_PER_ID} img/ID)...")
    valid_x = []
    valid_y = []
    
    grouped = df_clean.groupby('identity_id')
    
    with torch.no_grad():
        for identity_id, group in tqdm(grouped, desc="Pre-Inferenza"):
            facenet_id = mapper.get_facenet_id_by_class_id(identity_id)
            if facenet_id == -1: continue
            
            samples_taken = 0
            for _, row in group.iterrows():
                if SAMPLES_PER_ID is not None and samples_taken >= SAMPLES_PER_ID:
                    break
                    
                img_path = str(base_dir / row['image_path'])
                try:
                    img_pil = Image.open(img_path).convert('RGB')
                except: continue
                
                faces = mtcnn(img_pil)
                if faces is None: continue
                
                faces = faces.to(device)
                preds_all = torch.argmax(resnet(faces), dim=1).cpu().numpy()
                
                if facenet_id in preds_all:
                    match_idx = int(np.where(preds_all == facenet_id)[0][0])
                    best_face = faces[match_idx] # Range [-1, 1]
                    
                    # Convertiamo in [0, 1] numpy per ART
                    np_img_01 = (best_face.cpu().numpy() + 1.0) / 2.0
                    
                    valid_x.append(np_img_01)
                    valid_y.append(facenet_id)
                    samples_taken += 1

    if not valid_x:
        print("[ERRORE] Nessun campione valido.")
        return

    # Array per ART e Wrapper
    x_clean_arr = np.array(valid_x)
    y_clean_arr = np.array(valid_y)
    total_samples = len(valid_x)
    print(f"-> Immagini valide raccolte: {total_samples}")

    # ==========================================
    # 4. GRID SEARCH (CON WRAPPER)
    # ==========================================
    # Dizionario per il plot: {overshoot: {'acc': [], 'linf': []}}
    plot_data = {ov: {'acc': [], 'linf': []} for ov in overshoots}
    print("\n[FASE 2] Avvio DeepFool Grid Search...\n")
    
    with open(txt_log_path, 'w') as f:
        f.write(f"REPORT SCOUTING DEEPFOOL (ART WRAPPER)\n")
        f.write(f"Campioni testati: {total_samples}\n")
        f.write(f"Budget Massimo L_inf: {BUDGET_LINF}\n\n")

    for ov_shoot in overshoots:
        print(f"\n{'='*50}")
        print(f"Inizio test per OVERSHOOT = {ov_shoot}")
        print(f"{'='*50}")
        
        with open(txt_log_path, 'a') as f:
            f.write(f"\n{'='*50}\nInizio test per OVERSHOOT = {ov_shoot}\n{'='*50}\n")
            
        for steps in max_iters_list:
            log_str = f"\nGenerazione DeepFool con max_iter={steps}, epsilon (overshoot)={ov_shoot}, nb_grads={NB_GRADS}..."
            print(log_str)
            
            # 1. Istanziamo il tuo Wrapper al posto di quello di ART
            attack = ARTCompatibleDeepFool(
                classifier=classifier, 
                max_iter=steps, 
                epsilon=ov_shoot, 
                nb_grads=NB_GRADS
            )
            
            # 2. Generazione (Il wrapper gestisce tutto il casino interno)
            x_adv_arr = attack.generate(x=x_clean_arr)
            
            # 3. Predizione con classifier ART standard
            adv_preds_raw = classifier.predict(x_adv_arr)
            adv_preds = np.argmax(adv_preds_raw, axis=1)
            
            # 4. Calcolo L_inf
            diffs = np.abs(x_adv_arr - x_clean_arr)
            l_infs_np = np.max(diffs, axis=(1, 2, 3))
            
            # Statistiche
            l_min = l_infs_np.min()
            l_mean = l_infs_np.mean()
            l_median = np.median(l_infs_np)
            l_p95 = np.percentile(l_infs_np, 95)
            l_max = l_infs_np.max()
            
            within_budget_mask = l_infs_np <= BUDGET_LINF
            num_within_budget = np.sum(within_budget_mask)
            
            # Successo (Untargeted): L'attacco è nei limiti e la predizione è cambiata
            successful_and_legal_mask = within_budget_mask & (adv_preds != y_clean_arr)
            num_success_legal = np.sum(successful_and_legal_mask)
            
            robust_accuracy = (total_samples - num_success_legal) / total_samples
            
            # Salvataggio per il Plot
            plot_data[ov_shoot]['acc'].append(robust_accuracy * 100)
            plot_data[ov_shoot]['linf'].append(l_max)
            
            stats_str = (
                f"   Linf stats: min={l_min:.4f}, mean={l_mean:.4f}, median={l_median:.4f}, p95={l_p95:.4f}, max={l_max:.4f}\n"
                f"   Within budget (<= {BUDGET_LINF}): {num_within_budget}/{total_samples} ({(num_within_budget/total_samples)*100:.2f}%)\n"
                f"   Successful attacks within budget: {num_success_legal}/{total_samples} ({(num_success_legal/total_samples)*100:.2f}%)\n"
                f"-> Risultato: Robust Accuracy (per eps <= {BUDGET_LINF}) = {robust_accuracy*100:.2f}%\n"
            )
            print(stats_str, end="")
            
            with open(txt_log_path, 'a') as f:
                f.write(log_str + "\n")
                f.write(stats_str)
                
            # --- EARLY STOPPING LOGIC ---
            if robust_accuracy == 0.0:
                msg = f"   [!] Accuracy crollata a 0.00%. Salto max_iter successivi per questo overshoot.\n"
                print(msg)
                with open(txt_log_path, 'a') as f:
                    f.write(msg)
                
                # Riempiamo i dati mancanti per il plot con l'ultimo valore valido
                remaining_steps = len(max_iters_list) - len(plot_data[ov_shoot]['acc'])
                plot_data[ov_shoot]['acc'].extend([0.0] * remaining_steps)
                plot_data[ov_shoot]['linf'].extend([l_max] * remaining_steps)
                
                break

    print(f"\n[OK] Scouting completato. Report salvato in: {txt_log_path}")

    # ==========================================
    # 5. GENERAZIONE GRAFICO DI SCOUTING
    # ==========================================
    print("\n[FASE 3] Generazione Grafico Comparativo...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    colors = sns.color_palette("tab10", n_colors=len(overshoots))
    markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'X']
    
    for i, ov in enumerate(overshoots):
        # 1. JITTERING VISIVO: Aggiungiamo un microscopico offset basato sull'indice 'i'
        # In questo modo, se valgono tutte 0.0, verranno impilate a 0.00, 0.02, 0.04...
        jitter = i * 0.02
        acc_visiva = [val + jitter for val in plot_data[ov]['acc']]
        
        # Plot 1: Robust Accuracy (con i dati jitterati)
        ax1.plot(max_iters_list, acc_visiva, marker=markers[i % len(markers)], 
                 linewidth=2.5, markersize=8, color=colors[i], label=f'Overshoot = {ov}')
        
        # Plot 2: Max L_inf (Restano i dati reali)
        ax2.plot(max_iters_list, plot_data[ov]['linf'], marker=markers[i % len(markers)], 
                 linewidth=2.5, markersize=8, color=colors[i], label=f'Overshoot = {ov}')

    # Estetica Ax1 (Accuracy)
    ax1.set_title("Robust Accuracy vs. Max Iterations", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Max Iterations (steps)", fontsize=12)
    ax1.set_ylabel("Robust Accuracy (%)", fontsize=12)
    
    # 2. ROTAZIONE LABEL X: Evitiamo che i primi step (es. 5, 10, 20) si accavallino
    ax1.set_xticks(max_iters_list)
    ax1.set_xticklabels(max_iters_list, rotation=45) 
    
    # Adattiamo il tetto del grafico in base al massimo registrato (come fatto prima)
    all_accs = [val for ov in overshoots for val in plot_data[ov]['acc']]
    max_acc = max(all_accs) if all_accs else 100
    
    if max_acc <= 5:
        # Teniamo lo zoom stretto, aggiungendo un po' di margine per il jittering
        ax1.set_ylim(-0.05, max_acc + (len(overshoots) * 0.02) + 0.1) 
    elif max_acc <= 25:
        ax1.set_ylim(-1, max_acc + 5)
    else:
        ax1.set_ylim(-5, 105)
        
    # Applichiamo la rotazione anche al secondo grafico
    ax2.set_xticks(max_iters_list)
    ax2.set_xticklabels(max_iters_list, rotation=45)
    
    # Estetica Ax2 (L_inf)
    ax2.set_title(r"Max $L_\infty$ Perturbation vs. Max Iterations", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Max Iterations (steps)", fontsize=12)
    ax2.set_ylabel(r"Max $L_\infty$ Norm", fontsize=12)
    ax2.axhline(y=BUDGET_LINF, color='red', linestyle='--', linewidth=2, label=f"Max Budget ({BUDGET_LINF})")
    
    # Estraiamo le etichette da uno solo dei grafici (ax1)
    handles, labels = ax1.get_legend_handles_labels()
    
    # Se vuoi includere anche la linea tratteggiata rossa del Budget nella legenda
    handles_ax2, labels_ax2 = ax2.get_legend_handles_labels()
    for h, l in zip(handles_ax2, labels_ax2):
        if l not in labels:
            handles.append(h)
            labels.append(l)

    # Posizioniamo la legenda in basso, fuori dai plot, allineata orizzontalmente
    fig.legend(handles, labels, loc='lower center', ncol=len(labels), 
               title="DeepFool Overshoot & Budget", bbox_to_anchor=(0.5, -0.08), 
               fontsize=10, title_fontsize=11, frameon=True)

    plt.suptitle("DeepFool Hyperparameter Tuning Analysis", fontsize=18, y=1.05)
    plt.tight_layout()
    
    plot_path = output_dir / "deepfool_tuning_analysis.png"
    plt.savefig(plot_path, bbox_inches='tight', dpi=300)
    print(f"-> Grafico salvato in: {plot_path}")


if __name__ == "__main__":
    main()