import os
import sys
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path
from PIL import Image

# =========================================================================
# RISOLUZIONE ROBUSTA DEI PATH
# =========================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[5]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from facenet_pytorch import InceptionResnetV1, MTCNN
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import FastGradientMethod

from src.util.identity_mapper import IdentityMapper
from src.util.basic_img.metrics import calculate_linf
from src.util.attack_error_specific_utils import get_one_hot_target

def main():
    print("======================================================")
    print(" GENERATORE CAMPIONI: FGSM ERROR-GENERIC (UNTARGETED) ")
    print("======================================================\n")

    base_dir = PROJECT_ROOT
    print(f"-> Project Root impostata a: {base_dir}")

    # --- 1. CONFIGURAZIONE PATH E PARAMETRI ---
    csv_path = base_dir / "dataset" / "clean" / "splits" / "manifest.csv"
    meta_csv_path = base_dir / "dataset" / "clean" / "splits" / "identity_meta.csv"
    
    output_base_dir = base_dir / "dataset" / "attacks" / "NN1" / "error_generic" / "fgsm"
    cropped_clean_dir = base_dir / "dataset" / "clean_cropped" / "NN1"
    
    # Parametri classici FGSM (aggiungi o rimuovi in base alle tue necessità)
    epsilons = [0.025, 0.05, 0.075, 0.10, 0.15, 0.20]
    BATCH_SIZE = 64  
    
    if not csv_path.exists() or not meta_csv_path.exists():
        raise FileNotFoundError(f"Errore: manifest.csv o identity_meta.csv mancanti in {base_dir}")

    # --- 2. INIZIALIZZAZIONE MODELLI E MAPPER ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"-> Inizializzazione Reti su {device}...")
    
    mapper = IdentityMapper(meta_csv_path)
    mtcnn = MTCNN(image_size=160, margin=0, keep_all=True, post_process=True, device=device)
    
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    resnet.classify = True 

    classifier = PyTorchClassifier(
        model=resnet, clip_values=(0.0, 1.0), loss=nn.CrossEntropyLoss(), optimizer=None,
        input_shape=(3, 160, 160), nb_classes=8631, preprocessing=(0.5, 0.5), 
        device_type='gpu' if torch.cuda.is_available() else 'cpu'
    )

    df_clean = pd.read_csv(csv_path)
    print(f"-> Trovate {len(df_clean)} immagini totali nel manifest.")

    # =========================================================================
    # FASE 1: SMART CROPPING (Recupero o Generazione)
    # =========================================================================
    print("\n[FASE 1] Smart Cropping: Caricamento immagini esistenti o calcolo MTCNN...")
    valid_records = []
    cached_count = 0
    
    with torch.no_grad():
        for index, row in tqdm(df_clean.iterrows(), total=len(df_clean), desc="Pre-Inferenza"):
            class_id = str(row['identity_id'])
            facenet_id = mapper.get_facenet_id_by_class_id(class_id)
            if facenet_id == -1: continue
                
            source_img_path = base_dir / row['image_path']
            identity_dir_name = source_img_path.parent.name
            img_filename = f"{source_img_path.stem}.tiff"
                
            out_crop_dir = cropped_clean_dir / identity_dir_name
            out_crop_dir.mkdir(parents=True, exist_ok=True)
            crop_save_path = out_crop_dir / img_filename
            
            # --- LOGICA SMART: Se esiste, salta MTCNN e carica direttamente in RAM ---
            if crop_save_path.exists():
                img_bgr_float32 = cv2.imread(str(crop_save_path), cv2.IMREAD_UNCHANGED)
                if img_bgr_float32 is None:
                    continue
                img_rgb_float32 = cv2.cvtColor(img_bgr_float32, cv2.COLOR_BGR2RGB)
                x_clean = np.expand_dims(
                    np.transpose(img_rgb_float32.astype(np.float32), (2, 0, 1)),
                    axis=0,
                )
                cached_count += 1
            
            # --- Altrimenti, la calcola ex-novo con MTCNN e la salva ---
            else:
                try:
                    img_pil = Image.open(str(source_img_path)).convert('RGB')
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
                    
                    np_img_01 = (best_face_tensor.cpu().numpy() + 1.0) / 2.0
                    x_clean = np.expand_dims(np_img_01, axis=0) 
                    
                    img_c_bgr_float32 = cv2.cvtColor(
                        np.transpose(np_img_01, (1, 2, 0)).astype(np.float32),
                        cv2.COLOR_RGB2BGR,
                    )
                    if not cv2.imwrite(str(crop_save_path), img_c_bgr_float32):
                        print(f"\n[ERRORE] Impossibile salvare clean TIFF: {crop_save_path}")
                        continue
                else:
                    continue # MTCNN ha fallito nel trovare l'identità corretta
                
            row_dict = row.to_dict()
            row_dict['true_facenet_id'] = facenet_id
            row_dict['x_clean'] = x_clean 
            row_dict['cropped_image_path'] = crop_save_path.relative_to(base_dir).as_posix() 
            
            valid_records.append(row_dict)

    total_valid = len(valid_records)
    print(
        f"-> Immagini d'oro pronte per l'attacco FGSM Untargeted: {total_valid} "
        f"({cached_count} caricate da cache TIFF)"
    )

    # =========================================================================
    # FASE 2: GENERAZIONE FGSM UNTARGETED (BATCH DIRETTO)
    # =========================================================================
    print(f"\n======================================================")
    print(f" AVVIO GENERAZIONE ERROR-GENERIC")
    print(f"======================================================")
    
    for eps in epsilons:
        eps_str = f"eps_{eps:.3f}".replace('.', '_')
        eps_dir = output_base_dir / eps_str
        eps_dir.mkdir(parents=True, exist_ok=True)
        
        eps_tracker_records = []
        
        print(f"\n[>>>] Generazione Untargeted Epsilon = {eps:.3f}")
        
        attack = FastGradientMethod(
            estimator=classifier, 
            eps=eps, 
            targeted=False,
            batch_size=BATCH_SIZE 
        )
        
        for start_idx in tqdm(range(0, total_valid, BATCH_SIZE), desc=f"Batch Processing {eps_str}"):
            end_idx = min(start_idx + BATCH_SIZE, total_valid)
            batch_records = valid_records[start_idx:end_idx]
            
            batch_x_list = []
            batch_y_list = []
            
            for row in batch_records:
                batch_x_list.append(row['x_clean'][0])
                batch_y_list.append(get_one_hot_target(row['true_facenet_id'], num_classes=mapper.get_num_training_classes())[0])
                
            batch_x_array = np.stack(batch_x_list)
            batch_y_array = np.stack(batch_y_list)
            
            x_adv_batch = attack.generate(x=batch_x_array, y=batch_y_array)
            
            for i, row in enumerate(batch_records):
                x_clean_single = batch_x_array[i]
                x_adv_single = x_adv_batch[i]
                
                img_c_plot = np.transpose(x_clean_single, (1, 2, 0))
                img_a_plot = np.transpose(x_adv_single, (1, 2, 0))
                
                actual_linf = calculate_linf(img_c_plot, img_a_plot)
                mean_abs_perturbation = float(np.mean(np.abs(img_a_plot - img_c_plot)))
                
                source_img_path = base_dir / row['image_path']
                identity_dir_name = source_img_path.parent.name
                orig_filename = source_img_path.stem 

                out_img_dir = eps_dir / identity_dir_name
                out_img_dir.mkdir(parents=True, exist_ok=True)
                
                adv_filename = f"{orig_filename}.tiff"
                adv_save_path = out_img_dir / adv_filename
                
                img_a_bgr_float32 = cv2.cvtColor(
                    np.clip(img_a_plot, 0.0, 1.0).astype(np.float32),
                    cv2.COLOR_RGB2BGR,
                )
                if not cv2.imwrite(str(adv_save_path), img_a_bgr_float32):
                    print(f"\n[ERRORE FATALE] Impossibile salvare adversarial TIFF: {adv_save_path}")
                    continue

                rel_source = row['cropped_image_path']
                rel_adv = adv_save_path.relative_to(base_dir).as_posix()

                eps_tracker_records.append({
                    "attack_type": "fgsm", 
                    "eps": eps,
                    "targeted": False,
                    "target_strategy": "none",
                    "target_class": -1,
                    "dataset_label": row['dataset_label'],
                    "identity_id": row['identity_id'],
                    "identity_name": row['identity_name'],
                    "identity_dir": identity_dir_name,
                    "source_image_path": rel_source,
                    "adversarial_image_path": rel_adv,
                    "linf": round(actual_linf, 6),
                    "mean_abs_perturbation": round(mean_abs_perturbation, 6)
                })

        df_tracker = pd.DataFrame(eps_tracker_records)
        df_tracker.to_csv(eps_dir / f"tracker_{eps_str}.csv", index=False)
        
    print("\n[OK] Processo FGSM Error-Generic completato con immagini TIFF salvate via OpenCV!")

if __name__ == "__main__":
    main()
