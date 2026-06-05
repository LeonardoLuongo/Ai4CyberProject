"""
Generatore di Adversarial Samples: PGD TARGETED (Error-Specific)
Integra il ritaglio MTCNN (con Caching), e la generazione PGD in Batch.
"""

import os
# =========================================================================
# WORKAROUND CUDNN (Fondamentale per PGD iterativo)
# =========================================================================
os.environ["TORCH_CUDNN_V8_API_ENABLED"] = "0" 

import cv2
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path
from PIL import Image

# Disabilitiamo CUDNN per evitare il mismatch di librerie durante l'attacco
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

from facenet_pytorch import InceptionResnetV1, MTCNN
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import ProjectedGradientDescent

from util.identity_mapper import IdentityMapper
from util.basic_img.metrics import calculate_linf
from util.attack_error_specific_utils import select_target_label, get_one_hot_target

# =========================================================================
# CONFIGURAZIONE PARAMETRI GLOBALI 
# =========================================================================
# Parametri PGD
MAX_ITER  = 10     
STEP_MULT = 1.5    
NUM_INIT  = 1      
BATCH_SIZE = 64   

# Vettori di Esplorazione
EPSILONS = [0.01, 0.02, 0.03, 0.04, 0.05, 0.10]

# Strategie Targeted
STRATEGIES = ["next_best", "least-likely", "random"]
# =========================================================================

def main():
    print("======================================================")
    print(" GENERATORE CAMPIONI: PGD TARGETED BATCHED            ")
    print("======================================================\n")

    # --- 1. CONFIGURAZIONE PATH E PARAMETRI ---
    base_dir = Path(os.getcwd())
    csv_path = base_dir / "dataset" / "clean" / "splits" / "manifest.csv"
    meta_csv_path = base_dir / "dataset" / "clean" / "splits" / "identity_meta.csv"
    
    output_base_dir = base_dir / "dataset" / "attacks" / "NN1" / "error_specific" / "pgd"
    cropped_clean_dir = base_dir / "dataset" / "clean_cropped" / "NN1"
    
    if not csv_path.exists() or not meta_csv_path.exists():
        raise FileNotFoundError("Errore: manifest.csv o identity_meta.csv mancanti.")

    # --- 2. INIZIALIZZAZIONE MODELLI E MAPPER ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"-> Inizializzazione Reti su {device} (Batch Size: {BATCH_SIZE})...")
    
    mapper = IdentityMapper(meta_csv_path)
    mtcnn = MTCNN(image_size=160, margin=0, keep_all=True, post_process=True, device=device)
    
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    resnet.classify = True 

    # Wrapper ART (Accetta input [0, 1])
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
                # L'immagine esiste già! La carichiamo e calcoliamo i logits al volo
                img_bgr = cv2.imread(str(crop_save_path))
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                
                # Normalizziamo in [0, 1] e portiamo in formato CHW per ART
                np_img_01 = np.transpose(img_rgb, (2, 0, 1)).astype(np.float32) / 255.0
                x_clean = np.expand_dims(np_img_01, axis=0) # [1, 3, 160, 160]
                
                # Per i logits (ci servono per stabilire il target least-likely), passiamo per la rete.
                # FaceNet nativo vuole [-1, 1], quindi (x * 2) - 1
                t_clean = torch.tensor(x_clean * 2.0 - 1.0).to(device)
                best_logits = resnet(t_clean).cpu().numpy()
                
                row_dict['clean_logits'] = best_logits 
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
                best_logits = logits_all[match_idx].cpu().numpy()
                
                # Normalizziamo [0, 1]
                np_img_01 = (best_face_tensor.cpu().numpy() + 1.0) / 2.0
                x_clean = np.expand_dims(np_img_01, axis=0) 
                
                # Salva usando OpenCV
                img_c_save = (np.transpose(np_img_01, (1, 2, 0)) * 255.0).astype(np.uint8)
                img_c_bgr = cv2.cvtColor(img_c_save, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(crop_save_path), img_c_bgr)
                
                row_dict['clean_logits'] = np.expand_dims(best_logits, axis=0) 
                row_dict['x_clean'] = x_clean 
                valid_records.append(row_dict)

    print(f"-> Immagini d'oro pronte: {len(valid_records)} ({cached_count} caricate dalla cache rapida)")

    # =========================================================================
    # FASE 2: GENERAZIONE DEGLI ATTACCHI PGD BATCHED
    # =========================================================================
    
    for strategy in STRATEGIES:
        print(f"\n======================================================")
        print(f" AVVIO PGD BATCHED: STRATEGIA {strategy.upper()}")
        print(f"======================================================")
        
        strat_dir = output_base_dir / strategy
        strat_dir.mkdir(parents=True, exist_ok=True)
        
        for eps in EPSILONS:
            eps_str = f"eps_{eps:.3f}".replace('.', '_')
            eps_dir = strat_dir / eps_str
            eps_dir.mkdir(exist_ok=True)
            
            eps_step = (eps / MAX_ITER) * STEP_MULT
            print(f"\n[>>>] Generazione per Epsilon = {eps:.3f} (Step = {eps_step:.4f}, Iter = {MAX_ITER})")
            
            attack = ProjectedGradientDescent(
                estimator=classifier, 
                eps=eps, 
                eps_step=eps_step, 
                max_iter=MAX_ITER, 
                num_random_init=NUM_INIT,
                targeted=True,
                batch_size=BATCH_SIZE, # La GPU ora respirerà a pieni polmoni
                verbose=False
            )
            
            eps_tracker_records = []
            
            # Iteriamo a blocchi (Batches)
            for start_idx in tqdm(range(0, len(valid_records), BATCH_SIZE), desc=f"Attacco {eps_str} ({strategy})"):
                batch_records = valid_records[start_idx : start_idx + BATCH_SIZE]
                
                x_batch_list = []
                y_batch_list = []
                targets_memory = [] # Per ricordarci il target di ogni sample nel batch
                
                # 1. Preparazione Veloce sulla CPU
                for row in batch_records:
                    x_clean = row['x_clean']
                    true_facenet_id = row['true_facenet_id']
                    clean_logits = row['clean_logits']
                    
                    strat_str = strategy.replace("_", "-")
                    t_id = select_target_label(clean_logits, true_facenet_id, strategy=strat_str, num_classes=mapper.get_num_training_classes())
                    
                    y_target_onehot = get_one_hot_target(t_id, num_classes=mapper.get_num_training_classes())
                    
                    x_batch_list.append(x_clean)
                    y_batch_list.append(y_target_onehot)
                    targets_memory.append(t_id)

                # Concateniamo le liste in tensori Numpy giganti
                X_tensor = np.concatenate(x_batch_list, axis=0) # [B, 3, 160, 160]
                Y_tensor = np.concatenate(y_batch_list, axis=0) # [B, 8631]
                
                # 2. LA GPU LAVORA SUL BATCH INTERO
                X_adv_batch = attack.generate(x=X_tensor, y=Y_tensor)
                
                # 3. La CPU smista e salva i risultati
                for i, row in enumerate(batch_records):
                    target_label_8631 = targets_memory[i]
                    
                    # Estraiamo le singole immagini per il salvataggio e le metriche
                    x_clean_single = row['x_clean'][0] # Rimuoviamo la dimensione del batch: [3, 160, 160]
                    x_adv_single = X_adv_batch[i]
                    
                    img_c_plot = np.transpose(x_clean_single, (1, 2, 0))
                    img_a_plot = np.transpose(x_adv_single, (1, 2, 0))
                    
                    actual_linf = calculate_linf(img_c_plot, img_a_plot)
                    mean_abs_perturbation = float(np.mean(np.abs(img_a_plot - img_c_plot)))

                    source_img_path = str(base_dir / row['image_path'])
                    identity_dir_name = Path(source_img_path).parent.name
                    orig_filename = Path(source_img_path).stem 

                    out_img_dir = eps_dir / identity_dir_name
                    out_img_dir.mkdir(parents=True, exist_ok=True)
                    
                    adv_filename = f"{orig_filename}.jpg"
                    adv_save_path = out_img_dir / adv_filename
                    
                    # Salvataggio su disco
                    img_a_save = (img_a_plot * 255.0).astype(np.uint8)
                    cv2.imwrite(str(adv_save_path), cv2.cvtColor(img_a_save, cv2.COLOR_RGB2BGR))

                    rel_source = Path(row['cropped_image_path']).relative_to(base_dir).as_posix()
                    rel_adv = adv_save_path.relative_to(base_dir).as_posix()

                    eps_tracker_records.append({
                        "attack_type": "pgd",
                        "eps": eps,
                        "targeted": True,
                        "target_strategy": strategy,
                        "target_class": target_label_8631,
                        "dataset_label": row['dataset_label'],
                        "identity_id": row['identity_id'],
                        "identity_name": row['identity_name'],
                        "identity_dir": identity_dir_name,
                        "source_image_path": rel_source,
                        "adversarial_image_path": rel_adv,
                        "linf": round(actual_linf, 6),
                        "mean_abs_perturbation": round(mean_abs_perturbation, 6)
                    })

            # --- SALVATAGGIO TRACKER LOCALE A FINE EPSILON ---
            eps_tracker_path = eps_dir / f"tracker_{eps_str}.csv"
            df_tracker = pd.DataFrame(eps_tracker_records)
            df_tracker.to_csv(eps_tracker_path, index=False)
            print(f"-> Tracker PGD salvato in: {eps_tracker_path}")
        
    print("\n[OK] Processo completato! Il dataset PGD Error-Specific Batched è pronto.")

if __name__ == "__main__":
    main()