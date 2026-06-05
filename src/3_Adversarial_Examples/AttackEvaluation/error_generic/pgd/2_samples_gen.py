"""
Generatore di Adversarial Samples: PGD UNTARGETED (Error-Generic)
Integra il ritaglio MTCNN (con Caching) e la generazione PGD in Batch.
"""

from __future__ import annotations
import sys
import os
import cv2
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from PIL import Image
import json

# =========================================================================
# WORKAROUND CUDNN
# =========================================================================
os.environ["TORCH_CUDNN_V8_API_ENABLED"] = "0" 
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

PROJECT_ROOT = Path(__file__).resolve().parents[5]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import specifici dell'architettura
from facenet_pytorch import InceptionResnetV1, MTCNN
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import ProjectedGradientDescent
from src.util.identity_mapper import IdentityMapper

# Assumiamo che la cartella util sia accessibile come nel codice del collega
try:
    from util.basic_img.metrics import calculate_linf
except ImportError:
    # Fallback se non trova l'import esatto
    def calculate_linf(img1, img2):
        return float(np.max(np.abs(img1 - img2)))


def main():
    print("======================================================")
    print(" GENERATORE CAMPIONI: PGD UNTARGETED (Error-Generic)  ")
    print("======================================================\n")

    # =========================================================================
    # ⚙️ 1. CONFIGURAZIONE PARAMETRI PGD E PATHS ⚙️
    # =========================================================================
    EPSILONS = [0.015, 0.020, 0.025, 0.050, 0.075, 0.100]    
    MAX_ITER = 4
    STEP_MULT = 1.5
    NUM_INIT = 3
    BATCH_SIZE = 128
    
    base_dir = Path(os.getcwd())
    csv_path = base_dir / "dataset" / "clean" / "splits" / "manifest.csv"
    meta_csv_path = base_dir / "dataset" / "clean" / "splits" / "identity_meta.csv"
    
    # Path di output coerenti con il resto del team
    cropped_clean_dir = base_dir / "dataset" / "clean_cropped" / "NN1"
    output_base_dir = base_dir / "dataset" / "attacks" / "NN1" / "error_generic" / "pgd"
    # =========================================================================

    if not csv_path.exists() or not meta_csv_path.exists():
        raise FileNotFoundError("Errore: manifest.csv o identity_meta.csv mancanti.")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"-> Inizializzazione Reti su {device} (Batch Size: {BATCH_SIZE})...")
    
    mapper = IdentityMapper(meta_csv_path)
    mtcnn = MTCNN(image_size=160, margin=0, keep_all=True, post_process=True, device=device)
    
    # Inizializziamo la rete esattamente come il collega per garantire compatibilità
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    resnet.classify = True 

    # Wrapper ART: accetta input [0, 1] e fa (x - 0.5) / 0.5 -> [-1, 1] internamente
    classifier = PyTorchClassifier(
        model=resnet, clip_values=(0.0, 1.0), loss=nn.CrossEntropyLoss(), optimizer=None,
        input_shape=(3, 160, 160), nb_classes=8631, preprocessing=(0.5, 0.5), 
        device_type='gpu' if torch.cuda.is_available() else 'cpu'
    )

    df_clean = pd.read_csv(csv_path)
    print(f"-> Trovate {len(df_clean)} immagini totali nel manifest.")

    # =========================================================================
    # FASE 1: MTCNN E SCREMATURA (Con sistema di Caching)
    # =========================================================================
    print("\n[FASE 1] Verifica cache e generazione Clean Cropped Dataset...")
    valid_records = []
    cached_count = 0
    
    with torch.no_grad():
        for index, row in tqdm(df_clean.iterrows(), total=len(df_clean), desc="Pre-Inferenza"):
            class_id = str(row['identity_id'])
            
            facenet_id = mapper.get_facenet_id_by_class_id(class_id)
            if facenet_id == -1:
                continue
                
            source_img_path = str(base_dir / row['image_path'])
            identity_dir_name = Path(source_img_path).parent.name
            img_filename = Path(source_img_path).name
            
            out_crop_dir = cropped_clean_dir / identity_dir_name
            out_crop_dir.mkdir(parents=True, exist_ok=True)
            crop_save_path = out_crop_dir / img_filename
            
            row_dict = row.to_dict()
            row_dict['true_facenet_id'] = facenet_id
            row_dict['cropped_image_path'] = str(crop_save_path)
            
            # --- LOGICA DI CACHING ---
            if crop_save_path.exists():
                # L'immagine esiste già! La carichiamo dal disco
                img_bgr = cv2.imread(str(crop_save_path))
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                
                # Normalizziamo in [0, 1] e portiamo in formato CHW per ART
                np_img_01 = np.transpose(img_rgb, (2, 0, 1)).astype(np.float32) / 255.0
                x_clean = np.expand_dims(np_img_01, axis=0) # Shape: (1, 3, 160, 160)
                
                row_dict['x_clean'] = x_clean 
                valid_records.append(row_dict)
                cached_count += 1
                continue
                
            # --- SE NON ESISTE, FACCIAMO L'ESTRAZIONE CON MTCNN ---
            try:
                img_pil = Image.open(source_img_path).convert('RGB')
            except Exception:
                continue
            
            faces = mtcnn(img_pil)
            if faces is None: continue
            
            faces = faces.to(device)
            logits_all = resnet(faces)
            preds_all = torch.argmax(logits_all, dim=1).cpu().numpy()
            
            if facenet_id in preds_all:
                match_idx = np.where(preds_all == facenet_id)[0][0]
                best_face_tensor = faces[match_idx]
                
                # Normalizzazione coerente con il collega [0, 1]
                np_img_01 = (best_face_tensor.cpu().numpy() + 1.0) / 2.0
                x_clean = np.expand_dims(np_img_01, axis=0) 
                
                # Salvataggio fisico dell'immagine Clean Cropped
                img_c_save = (np.transpose(np_img_01, (1, 2, 0)) * 255.0).astype(np.uint8)
                img_c_bgr = cv2.cvtColor(img_c_save, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(crop_save_path), img_c_bgr)
                
                row_dict['x_clean'] = x_clean 
                valid_records.append(row_dict)

    print(f"-> Immagini d'oro pronte: {len(valid_records)} ({cached_count} caricate dalla cache rapida)")

    # =========================================================================
    # FASE 2: GENERAZIONE DEGLI ATTACCHI PGD (Batched + Tracker)
    # =========================================================================
    
    for eps in EPSILONS:
        eps_str = f"eps_{eps:.3f}".replace('.', '_')
        eps_dir = output_base_dir / eps_str
        eps_dir.mkdir(parents=True, exist_ok=True)
        
        eps_tracker_records = []
        eps_step = (eps / MAX_ITER) * STEP_MULT
        
        print(f"\n======================================================")
        print(f" AVVIO GENERAZIONE PGD: {eps_str} (Step: {eps_step:.4f}, Init: {NUM_INIT})")
        print(f"======================================================")
        
        attack = ProjectedGradientDescent(
            estimator=classifier, 
            eps=eps, 
            eps_step=eps_step, 
            max_iter=MAX_ITER, 
            num_random_init=NUM_INIT,
            targeted=False,           # Attacco Error-Generic (Untargeted)
            batch_size=BATCH_SIZE,
            verbose=False
        )
        
        # Iteriamo a blocchi (batch) per massima velocità, ma salviamo singolarmente
        for start_idx in tqdm(range(0, len(valid_records), BATCH_SIZE), desc=f"Generazione {eps_str}"):
            batch_records = valid_records[start_idx:start_idx + BATCH_SIZE]
            
            # Concateniamo i tensori [1, 3, 160, 160] in un unico [BATCH, 3, 160, 160]
            batch_x = np.concatenate([r['x_clean'] for r in batch_records], axis=0)
            
            # 1. Attacco di massa
            x_adv = attack.generate(x=batch_x)
            
            # 2. Salvataggio e calcolo metriche per ogni immagine del batch
            for i, adv_img_np in enumerate(x_adv):
                rec = batch_records[i]
                
                orig_path = Path(rec['image_path'])
                identity_dir_name = orig_path.parent.name
                orig_filename = orig_path.name
                
                # Trasposizione per plot e salvataggio (C,H,W) -> (H,W,C)
                img_c_plot = np.transpose(rec['x_clean'][0], (1, 2, 0))
                img_a_plot = np.transpose(adv_img_np, (1, 2, 0))
                
                # Metriche e Tracker
                actual_linf = calculate_linf(img_c_plot, img_a_plot)
                mean_abs_perturbation = float(np.mean(np.abs(img_a_plot - img_c_plot)))

                out_img_dir = eps_dir / identity_dir_name
                out_img_dir.mkdir(parents=True, exist_ok=True)
                
                adv_save_path = out_img_dir / orig_filename
                
                # Salvataggio OpenCV
                img_a_save = (img_a_plot * 255.0).astype(np.uint8)
                cv2.imwrite(str(adv_save_path), cv2.cvtColor(img_a_save, cv2.COLOR_RGB2BGR))

                rel_source = Path(rec['cropped_image_path']).relative_to(base_dir).as_posix()
                rel_adv = adv_save_path.relative_to(base_dir).as_posix()

                # Aggiungiamo i dati al tracker con gli stessi campi del collega
                eps_tracker_records.append({
                    "attack_type": "pgd",
                    "eps": eps,
                    "targeted": False,
                    "target_strategy": "untargeted",
                    "target_class": None,
                    "dataset_label": rec['dataset_label'],
                    "identity_id": rec['identity_id'],
                    "identity_name": rec['identity_name'],
                    "identity_dir": identity_dir_name,
                    "source_image_path": rel_source,
                    "adversarial_image_path": rel_adv,
                    "linf": round(actual_linf, 6),
                    "mean_abs_perturbation": round(mean_abs_perturbation, 6)
                })

        # --- 3. SALVATAGGIO TRACKER LOCALE A FINE EPSILON ---
        eps_tracker_path = eps_dir / f"tracker_{eps_str}.csv"
        df_tracker = pd.DataFrame(eps_tracker_records)
        df_tracker.to_csv(eps_tracker_path, index=False)
        print(f"-> Tracker CSV salvato in: {eps_tracker_path}")
        
    print("\n[OK] Processo completato! Il dataset PGD è pronto per NN2 e Difese.")

if __name__ == "__main__":
    main()