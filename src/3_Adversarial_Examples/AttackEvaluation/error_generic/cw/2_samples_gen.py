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

# Importiamo la tua classe C&W super-ottimizzata e le utility
from util.cw_benchmarks.cw_pytorch import CarliniLInfMethodPyTorch
from util.identity_mapper import IdentityMapper
from util.basic_img.metrics import calculate_linf

def main():
    print("======================================================")
    print(" GENERATORE CAMPIONI: C&W UNTARGETED (Error-Generic)  ")
    print("======================================================\n")

    # --- 1. CONFIGURAZIONE PATH E PARAMETRI ---
    base_dir = Path.cwd()
    csv_path = base_dir / "dataset" / "clean" / "splits" / "manifest.csv"
    meta_csv_path = base_dir / "dataset" / "clean" / "splits" / "identity_meta.csv"
    output_base_dir = base_dir / "dataset" / "attacks" / "NN1" / "error_generic" / "cw"
    
    # Cartella di caching per i volti puliti già estratti
    cropped_clean_dir = base_dir / "dataset" / "clean_cropped" / "NN1"
    
    # PARAMETRI AGGIORNATI per convergenza fluida
    CW_MAX_ITER = 10
    CW_LR = 0.01
    BATCH_SIZE = 768
    
    if not csv_path.exists() or not meta_csv_path.exists():
        raise FileNotFoundError("Errore: manifest.csv o identity_meta.csv mancanti.")

    # --- 2. INIZIALIZZAZIONE MODELLI E MAPPER ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"-> Inizializzazione Reti su {device}...")
    
    mapper = IdentityMapper(meta_csv_path)
    mtcnn = MTCNN(image_size=160, margin=0, keep_all=True, post_process=True, device=device)
    
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    resnet.classify = True 

    # Inizializziamo il Classifier ART fittizio (serve solo come contenitore per la classe)
    classifier = PyTorchClassifier(
        model=resnet, clip_values=(0.0, 1.0), loss=nn.CrossEntropyLoss(), optimizer=None,
        input_shape=(3, 160, 160), nb_classes=8631, preprocessing=(0.5, 0.5), 
        device_type='gpu' if torch.cuda.is_available() else 'cpu'
    )

    df_clean = pd.read_csv(csv_path)
    print(f"-> Trovate {len(df_clean)} immagini totali nel manifest.")

    # =========================================================================
    # FASE 1: MTCNN E SCREMATURA (Filtro Zero-Shot e Salvataggio Cache TIFF 32-bit)
    # =========================================================================
    print("\n[FASE 1] Verifica cache e generazione Clean Cropped Dataset...")
    valid_records = []
    cached_count = 0
    
    with torch.no_grad():
        for index, row in tqdm(df_clean.iterrows(), total=len(df_clean), desc="Pre-Inferenza"):
            class_id = str(row['identity_id'])
            
            facenet_id = mapper.get_facenet_id_by_class_id(class_id)
            if facenet_id == -1: continue
                
            source_img_path = Path(base_dir / row['image_path'])
            identity_dir_name = source_img_path.parent.name
            
            # --- MODIFICA: Usiamo il TIFF 32-bit float anche per il crop pulito ---
            img_filename_tiff = f"{source_img_path.stem}.tiff"
            
            out_crop_dir = cropped_clean_dir / identity_dir_name
            out_crop_dir.mkdir(parents=True, exist_ok=True)
            crop_save_path = out_crop_dir / img_filename_tiff
            
            row_dict = row.to_dict()
            row_dict['true_facenet_id'] = facenet_id
            row_dict['cropped_image_path'] = str(crop_save_path)
            
            # --- LOGICA DI CACHING (TIFF 32-BIT) ---
            if crop_save_path.exists():
                # Leggiamo il TIFF float32 [0.0, 1.0] (IMREAD_UNCHANGED è fondamentale)
                img_bgr_float32 = cv2.imread(str(crop_save_path), cv2.IMREAD_UNCHANGED)
                img_rgb_float32 = cv2.cvtColor(img_bgr_float32, cv2.COLOR_BGR2RGB)
                
                # È già nel range giusto, facciamo solo transpose
                x_clean = np.expand_dims(np.transpose(img_rgb_float32, (2, 0, 1)), axis=0) 
                
                row_dict['x_clean'] = x_clean 
                valid_records.append(row_dict)
                cached_count += 1
                continue
                
            # --- ESTRAZIONE SE NON IN CACHE ---
            try:
                img_pil = Image.open(str(source_img_path)).convert('RGB')
            except Exception: continue
            
            faces = mtcnn(img_pil)
            if faces is None: continue
            
            faces = faces.to(device)
            preds_all = torch.argmax(resnet(faces), dim=1).cpu().numpy()
            
            if facenet_id in preds_all:
                match_idx = np.where(preds_all == facenet_id)[0][0]
                best_face_tensor = faces[match_idx]
                
                # [0.0, 1.0] float32
                np_img_01 = (best_face_tensor.cpu().numpy() + 1.0) / 2.0
                x_clean = np.expand_dims(np_img_01, axis=0) 
                
                # --- MODIFICA: Salvataggio TIFF 32-bit ---
                img_c_bgr_float32 = cv2.cvtColor(np.transpose(np_img_01, (1, 2, 0)), cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(crop_save_path), img_c_bgr_float32)
                
                row_dict['x_clean'] = x_clean 
                valid_records.append(row_dict)

    print(f"-> Immagini pronte per l'attacco: {len(valid_records)} ({cached_count} caricate da cache TIFF)")

    # =========================================================================
    # FASE 2: GENERAZIONE DEGLI ATTACCHI C&W (BATCHED)
    # =========================================================================
    print(f"\n[FASE 2] Avvio Generazione C&W Untargeted (lr={CW_LR}, max_iter={CW_MAX_ITER})...")
    
    # C&W calcola lui il suo tau ottimale per rompere la rete. 
    # Non ha senso ciclare sugli epsilons [0.025, 0.05...]. C&W si lancia una volta sola.
    
    output_base_dir.mkdir(parents=True, exist_ok=True)
    tracker_output_path = output_base_dir / "tracker_cw_untargeted.csv"
    
    tracker_records = []
    
    # Istanziamo LA TUA CLASSE per un attacco Untargeted
    attack = CarliniLInfMethodPyTorch(
        classifier=classifier, 
        targeted=False, 
        max_iter=CW_MAX_ITER,         
        learning_rate=CW_LR,
        batch_size=BATCH_SIZE,
        verbose=False
    )
    
    for start_idx in tqdm(range(0, len(valid_records), BATCH_SIZE), desc="Generazione Batch C&W"):
        batch_records = valid_records[start_idx:start_idx + BATCH_SIZE]
        
        # Concateniamo i tensori in un unico batch
        batch_x = np.concatenate([r['x_clean'] for r in batch_records], axis=0)
        
        # 1. Attacco di massa 
        x_adv = attack.generate(x=batch_x)
        
        # 2. INFERENZA ON-THE-FLY (Per calcolare l'esito reale evitando quantizzazioni)
        with torch.no_grad():
            x_adv_tensor = torch.tensor(x_adv).to(device)
            adv_logits = resnet(x_adv_tensor * 2.0 - 1.0)
            adv_preds = torch.argmax(adv_logits, dim=1).cpu().numpy()
            adv_probs = torch.nn.functional.softmax(adv_logits, dim=1)
            
            clean_logits = resnet(torch.tensor(batch_x).to(device) * 2.0 - 1.0)
            clean_probs = torch.nn.functional.softmax(clean_logits, dim=1)

        # 3. Salvataggio e tracciamento
        for i, adv_img_np in enumerate(x_adv):
            rec = batch_records[i]
            orig_path = Path(rec['image_path'])
            identity_dir_name = orig_path.parent.name
            orig_filename_no_ext = orig_path.stem 
            true_facenet_id = rec['true_facenet_id']
            
            img_c_plot = np.transpose(rec['x_clean'][0], (1, 2, 0))
            img_a_plot = np.transpose(adv_img_np, (1, 2, 0))
            
            actual_linf = calculate_linf(img_c_plot, img_a_plot)
            mean_abs_perturbation = float(np.mean(np.abs(img_a_plot - img_c_plot)))

            out_img_dir = output_base_dir / identity_dir_name
            out_img_dir.mkdir(parents=True, exist_ok=True)
            
            # --- NUOVO: SALVATAGGIO IN TIFF 32-BIT FLOAT ---
            adv_filename = f"{orig_filename_no_ext}.tiff"
            adv_save_path = out_img_dir / adv_filename
            
            # L'immagine è float32 in RGB. OpenCV vuole BGR. Manteniamo il range [0, 1] e float32.
            img_a_bgr_float32 = cv2.cvtColor(img_a_plot.astype(np.float32), cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(adv_save_path), img_a_bgr_float32)

            rel_source = Path(rec['cropped_image_path']).relative_to(base_dir).as_posix()
            rel_adv = adv_save_path.relative_to(base_dir).as_posix()
            
            adv_conf = float(adv_probs[i, true_facenet_id].cpu().numpy())
            clean_conf = float(clean_probs[i, true_facenet_id].cpu().numpy())

            tracker_records.append({
                "attack_type": "cw_linf",
                "max_iter": CW_MAX_ITER,
                "learning_rate": CW_LR,
                "targeted": False,
                "target_strategy": "none",
                "target_class": -1,
                "dataset_label": rec['dataset_label'],
                "identity_id": rec['identity_id'],
                "identity_name": rec['identity_name'],
                "identity_dir": identity_dir_name,
                "source_image_path": rel_source,
                "adversarial_image_path": rel_adv,
                "linf": round(actual_linf, 6),
                "mean_abs_perturbation": round(mean_abs_perturbation, 6),
                "clean_pred_class": true_facenet_id,
                "adv_pred_class": adv_preds[i],
                "clean_confidence": clean_conf,
                "adv_confidence": adv_conf
            })

    # --- 3. SALVATAGGIO TRACKER GLOBALE ---
    df_tracker = pd.DataFrame(tracker_records)
    df_tracker.to_csv(tracker_output_path, index=False)
    print(f"\n[OK] Generazione completata in TIFF 32-bit! Tracker salvato in: {tracker_output_path}")

if __name__ == "__main__":
    main()