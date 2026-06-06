import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path
from PIL import Image

from facenet_pytorch import InceptionResnetV1, MTCNN
from art.estimators.classification import PyTorchClassifier

from util.identity_mapper import IdentityMapper
from util.basic_img.metrics import calculate_linf
from util.deepfool_wrapped import ARTCompatibleDeepFool

def main():
    print("======================================================")
    print(" GENERATORE CAMPIONI: DEEPFOOL (Error-Generic)        ")
    print("======================================================\n")

    # --- 1. CONFIGURAZIONE PATH E PARAMETRI ---
    base_dir = Path.cwd()
    csv_path = base_dir / "dataset" / "clean" / "splits" / "manifest.csv"
    meta_csv_path = base_dir / "dataset" / "clean" / "splits" / "identity_meta.csv"
    
    # Path di Output
    output_base_dir = base_dir / "dataset" / "attacks" / "error_generic" / "deepfool"
    cropped_clean_dir = base_dir / "dataset" / "clean_cropped" / "NN1"
    
    samples_dir = output_base_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    
    tracker_output_path = output_base_dir / "tracker_deepfool.csv"

    # I parametri ottimali che hai trovato con lo scouting
    DF_OVERSHOOT = 0.15
    DF_MAX_ITER = 5
    DF_NB_GRADS = 3
    
    if not csv_path.exists() or not meta_csv_path.exists():
        raise FileNotFoundError("Errore: manifest.csv o identity_meta.csv mancanti.")

    # --- 2. INIZIALIZZAZIONE MODELLI E MAPPER ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"-> Inizializzazione Reti su {device}...")
    
    mapper = IdentityMapper(meta_csv_path)
    mtcnn = MTCNN(image_size=160, margin=0, keep_all=True, post_process=True, device=device)
    
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    resnet.classify = True 

    # Wrapper ART base
    classifier = PyTorchClassifier(
        model=resnet, clip_values=(0.0, 1.0), loss=nn.CrossEntropyLoss(), optimizer=None,
        input_shape=(3, 160, 160), nb_classes=8631, preprocessing=(0.5, 0.5), 
        device_type='gpu' if torch.cuda.is_available() else 'cpu'
    )
    
    # Istanziamo il tuo Wrapper per DeepFool
    print(f"-> Setup DeepFool (overshoot={DF_OVERSHOOT}, max_iter={DF_MAX_ITER}, nb_grads={DF_NB_GRADS})...")
    attack = ARTCompatibleDeepFool(
        classifier=classifier, 
        max_iter=DF_MAX_ITER, 
        epsilon=DF_OVERSHOOT, 
        nb_grads=DF_NB_GRADS
    )

    df_clean = pd.read_csv(csv_path)

    # =========================================================================
    # FASE 1: MTCNN E SCREMATURA (Filtro Zero-Shot e Caching TIFF 32-bit)
    # =========================================================================
    print("\n[FASE 1] Ritaglio MTCNN e Scrematura (Filtro Zero-Shot / Misclassificate)...")
    valid_records = []
    cached_count = 0
    
    with torch.no_grad():
        for index, row in tqdm(df_clean.iterrows(), total=len(df_clean), desc="Pre-Inferenza"):
            class_id = str(row['identity_id'])
            
            facenet_id = mapper.get_facenet_id_by_class_id(class_id)
            if facenet_id == -1: continue
                
            source_img_path = Path(base_dir / row['image_path'])
            identity_dir_name = source_img_path.parent.name
            
            # --- MODIFICA: Usiamo il TIFF 32-bit float ---
            img_filename_tiff = f"{source_img_path.stem}.tiff"
            
            out_crop_dir = cropped_clean_dir / identity_dir_name
            out_crop_dir.mkdir(parents=True, exist_ok=True)
            crop_save_path = out_crop_dir / img_filename_tiff
            
            # --- LOGICA DI CACHING (TIFF 32-BIT) ---
            if crop_save_path.exists():
                img_bgr_float32 = cv2.imread(str(crop_save_path), cv2.IMREAD_UNCHANGED)
                img_rgb_float32 = cv2.cvtColor(img_bgr_float32, cv2.COLOR_BGR2RGB)
                
                x_clean = np.expand_dims(np.transpose(img_rgb_float32, (2, 0, 1)), axis=0)
                
                # Calcoliamo i logits al volo dalla cache
                t_clean = torch.tensor(x_clean * 2.0 - 1.0).to(device)
                best_logits = resnet(t_clean).cpu().numpy()[0]
                
                valid_records.append({
                    'dataset_label': row['dataset_label'],
                    'identity_id': row['identity_id'],
                    'identity_name': row['identity_name'],
                    'identity_dir_name': identity_dir_name,
                    'img_filename': img_filename_tiff,
                    'cropped_image_path': str(crop_save_path),
                    'true_facenet_id': facenet_id,  
                    'clean_logits': best_logits,
                    'x_clean': x_clean
                })
                cached_count += 1
                continue

            # --- ESTRAZIONE SE NON IN CACHE ---
            try:
                img_pil = Image.open(str(source_img_path)).convert('RGB')
            except Exception: continue
            
            faces = mtcnn(img_pil)
            if faces is None: continue
            
            faces = faces.to(device)
            logits_all = resnet(faces)
            preds_all = torch.argmax(logits_all, dim=1).cpu().numpy()
            
            if facenet_id in preds_all:
                match_idx = np.where(preds_all == facenet_id)[0][0]
                best_face_tensor = faces[match_idx]
                best_logits = logits_all[match_idx].cpu().numpy()
                
                np_img_01 = (best_face_tensor.cpu().numpy() + 1.0) / 2.0
                x_clean = np.expand_dims(np_img_01, axis=0) 
                
                # --- MODIFICA: Salvataggio TIFF 32-bit ---
                img_c_bgr_float32 = cv2.cvtColor(np.transpose(np_img_01, (1, 2, 0)), cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(crop_save_path), img_c_bgr_float32)
                
                valid_records.append({
                    'dataset_label': row['dataset_label'],
                    'identity_id': row['identity_id'],
                    'identity_name': row['identity_name'],
                    'identity_dir_name': identity_dir_name,
                    'img_filename': img_filename_tiff,
                    'cropped_image_path': str(crop_save_path),
                    'true_facenet_id': facenet_id,  
                    'clean_logits': best_logits,
                    'x_clean': x_clean
                })

    print(f"-> Immagini d'oro pronte: {len(valid_records)} ({cached_count} da cache TIFF)")

    # =========================================================================
    # FASE 2: GENERAZIONE DEEPFOOL E INFERENZA AL VOLO
    # =========================================================================
    print(f"\n[FASE 2] Generazione Attacchi Avversari e Inferenza...")
    tracker_records = []
    
    pbar = tqdm(valid_records, desc=f"Attacco DeepFool")
    
    for record in pbar:
        x_clean = record['x_clean']
        true_facenet_id = record['true_facenet_id']
        clean_logits = record['clean_logits']
        
        identity_dir_name = record['identity_dir_name']
        img_filename = record['img_filename']

        # 1. Attacco DeepFool
        x_adv = attack.generate(x=x_clean)
        
        # 2. INFERENZA ON-THE-FLY (Sui Float32 PURI)
        with torch.no_grad():
            x_adv_tensor = torch.tensor(x_adv).to(device)
            # Rete nuda: (x * 2) - 1
            adv_logits = resnet(x_adv_tensor * 2.0 - 1.0)
            adv_pred_class = int(torch.argmax(adv_logits, dim=1).cpu().numpy()[0])
            
            # Salviamo anche la confidenza della classe ORIGINALE
            # (Per il Confidence Degradation Plot)
            adv_probs = torch.nn.functional.softmax(adv_logits, dim=1)
            adv_conf = float(adv_probs[0, true_facenet_id].cpu().numpy())
            
            # Recuperiamo la confidenza pulita dai logits salvati
            clean_probs = torch.nn.functional.softmax(torch.tensor(clean_logits).unsqueeze(0).to(device), dim=1)
            clean_conf = float(clean_probs[0, true_facenet_id].cpu().numpy())

        # 3. Validazione L_inf
        img_c_plot = np.transpose(x_clean[0], (1, 2, 0))
        img_a_plot = np.transpose(x_adv[0], (1, 2, 0))
        
        actual_linf = calculate_linf(img_c_plot, img_a_plot)
        mean_abs_perturbation = float(np.mean(np.abs(img_a_plot - img_c_plot)))

        # Salvataggio Immagine Avversaria in TIFF 32-bit
        out_img_dir = samples_dir / identity_dir_name
        out_img_dir.mkdir(parents=True, exist_ok=True)
        adv_save_path = out_img_dir / img_filename # img_filename ora contiene già ".tiff"
        
        img_a_bgr_float32 = cv2.cvtColor(img_a_plot.astype(np.float32), cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(adv_save_path), img_a_bgr_float32)

        rel_source = Path(record['cropped_image_path']).relative_to(base_dir).as_posix()
        rel_adv = adv_save_path.relative_to(base_dir).as_posix()

        # Nel CSV salviamo ORA tutte le informazioni finali
        tracker_records.append({
            "attack_type": "deepfool",
            "overshoot": DF_OVERSHOOT,
            "max_iter": DF_MAX_ITER,
            "nb_grads": DF_NB_GRADS,
            "targeted": False,
            "target_strategy": "none",
            "target_class": -1,
            "dataset_label": record['dataset_label'],
            "identity_id": record['identity_id'],
            "identity_name": record['identity_name'],
            "identity_dir": identity_dir_name,
            "source_image_path": rel_source,
            "adversarial_image_path": rel_adv,
            "linf": round(actual_linf, 6),
            "mean_abs_perturbation": round(mean_abs_perturbation, 6),
            # --- I DATI PURI ---
            "clean_pred_class": true_facenet_id,
            "adv_pred_class": adv_pred_class,
            "clean_confidence": clean_conf,
            "adv_confidence": adv_conf
        })

    # Salvataggio CSV Globale
    df_tracker = pd.DataFrame(tracker_records)
    df_tracker.to_csv(tracker_output_path, index=False)
            
    print(f"\n[OK] Generazione completata! Tracker salvato in: {tracker_output_path}")

if __name__ == "__main__":
    main()