import os
import cv2
import numpy as np
import pandas as pd
import torch

# Permette alla GPU di trovare l'algoritmo FP64 più veloce per immagini 160x160
torch.backends.cudnn.benchmark = True
# Opzionale: disabilitare il deterministic spinge ulteriormente le performance, 
# pur mantenendo una precisione a 10^-8
torch.backends.cudnn.deterministic = False

import torch.nn as nn
from tqdm import tqdm
from pathlib import Path
from PIL import Image

from facenet_pytorch import InceptionResnetV1, MTCNN
from art.estimators.classification import PyTorchClassifier

# Importiamo la tua classe SOTA e le utility
from util.cw_benchmarks.cw_pytorch import CarliniLInfMethodPyTorch
from util.identity_mapper import IdentityMapper
from util.basic_img.metrics import calculate_linf
from util.attack_error_specific_utils import get_one_hot_target

# ==========================================
# WRAPPER (Per Normalizzazione Range e Float64)
# ==========================================
class ARTFloat64Wrapper(nn.Module):
    """Intercetta l'input degradato a Float32 da ART e lo riporta a Float64 per ResNet."""
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x):
        return self.model(x.to(torch.float64))

def main():
    print("======================================================")
    print(" GENERATORE CAMPIONI: C&W UNTARGETED (64-BIT SOTA)    ")
    print("======================================================\n")

    # --- 1. CONFIGURAZIONE PATH E PARAMETRI ---
    base_dir = Path.cwd()
    csv_path = base_dir / "dataset" / "clean" / "splits" / "manifest.csv"
    meta_csv_path = base_dir / "dataset" / "clean" / "splits" / "identity_meta.csv"
    output_base_dir = base_dir / "dataset" / "attacks" / "NN1" / "error_generic" / "cw"
    
    # Cartella di caching per i volti puliti
    cropped_clean_dir = base_dir / "dataset" / "clean_cropped" / "NN1"
    
    # =========================================================================
    # PARAMETRI DEL GENERATORE E CONTROLLO DATASET
    # =========================================================================
    CW_MAX_ITER = 5
    CW_LR = 0.005
    BATCH_SIZE = 224 # Teniamo a 32 per gestire comodamente i Float64 in VRAM
    
    # --- VARIABILI PER IL CONTROLLO RAPIDO DEI CAMPIONI ---
    SAMPLES_PER_ID = 10     # Numero massimo di foto da prelevare per singola identità
    MAX_TOTAL_IMAGES = 1_000     # Limite massimo assoluto di immagini generate (Es: 10 per un test veloce)
    # =========================================================================

    if not csv_path.exists() or not meta_csv_path.exists():
        raise FileNotFoundError("Errore: manifest.csv o identity_meta.csv mancanti.")

    # --- 2. INIZIALIZZAZIONE MODELLI E MAPPER ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"-> Inizializzazione Reti su {device} (Double Precision)...")
    
    mapper = IdentityMapper(meta_csv_path)
    mtcnn = MTCNN(image_size=160, margin=0, keep_all=True, post_process=True, device=device)
    
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    resnet.classify = True 
    resnet.double() # <-- FORZIAMO LA RETE IN 64-BIT PER IL CALCOLO DELL'ATTACCO
    
    art_resnet_shield = ARTFloat64Wrapper(resnet).eval()

    # Inizializziamo il Classifier ART
    classifier = PyTorchClassifier(
        model=art_resnet_shield, clip_values=(0.0, 1.0), loss=nn.CrossEntropyLoss(), optimizer=None,
        input_shape=(3, 160, 160), nb_classes=8631, preprocessing=(0.5, 0.5), 
        device_type='gpu' if torch.cuda.is_available() else 'cpu'
    )

    df_clean = pd.read_csv(csv_path)
    print(f"-> Trovate {len(df_clean)} immagini totali nel manifest originale.")

    # =========================================================================
    # FASE 1: MTCNN E SCREMATURA (Filtro Zero-Shot e Salvataggio Cache TIFF)
    # =========================================================================
    print(f"\n[FASE 1] Estrazione Dataset (Max {MAX_TOTAL_IMAGES} totali, {SAMPLES_PER_ID} per ID)...")
    valid_records = []
    cached_count = 0
    id_counts = {}
    
    with torch.no_grad():
        for index, row in tqdm(df_clean.iterrows(), total=len(df_clean), desc="Pre-Inferenza"):
            
            # --- BLOCCO LIMITE IMMAGINI ---
            if len(valid_records) >= MAX_TOTAL_IMAGES:
                break
                
            class_id = str(row['identity_id'])
            
            if id_counts.get(class_id, 0) >= SAMPLES_PER_ID:
                continue
            # --------------------------------
            
            facenet_id = mapper.get_facenet_id_by_class_id(class_id)
            if facenet_id == -1: continue
                
            source_img_path = Path(base_dir / row['image_path'])
            identity_dir_name = source_img_path.parent.name
            img_filename_tiff = f"{source_img_path.stem}.tiff"
            
            out_crop_dir = cropped_clean_dir / identity_dir_name
            out_crop_dir.mkdir(parents=True, exist_ok=True)
            crop_save_path = out_crop_dir / img_filename_tiff
            
            row_dict = row.to_dict()
            row_dict['true_facenet_id'] = facenet_id
            row_dict['cropped_image_path'] = str(crop_save_path)
            
            # --- LOGICA DI CACHING (TIFF 32-BIT) ---
            if crop_save_path.exists():
                img_bgr_float32 = cv2.imread(str(crop_save_path), cv2.IMREAD_UNCHANGED)
                img_rgb_float32 = cv2.cvtColor(img_bgr_float32, cv2.COLOR_BGR2RGB)
                
                # Salviamo in Float64 per darlo in pasto ad ART
                x_clean = np.expand_dims(np.transpose(img_rgb_float32, (2, 0, 1)), axis=0).astype(np.float64)
                
                row_dict['x_clean'] = x_clean 
                valid_records.append(row_dict)
                id_counts[class_id] = id_counts.get(class_id, 0) + 1
                cached_count += 1
                continue
                
            # --- ESTRAZIONE SE NON IN CACHE ---
            try:
                img_pil = Image.open(str(source_img_path)).convert('RGB')
            except Exception: continue
            
            faces = mtcnn(img_pil)
            if faces is None: continue
            
            # Passiamo i volti a FaceNet in Float64
            faces_device = faces.to(device).double()
            preds_all = torch.argmax(resnet(faces_device), dim=1).cpu().numpy()
            
            if facenet_id in preds_all:
                match_idx = int(np.where(preds_all == facenet_id)[0][0])
                best_face_tensor = faces_device[match_idx]
                
                np_img_01 = (best_face_tensor.cpu().numpy() + 1.0) / 2.0
                x_clean = np.expand_dims(np_img_01, axis=0) # Float64
                
                # Salvataggio TIFF 32-bit (per occupare meno spazio su disco)
                img_c_bgr_float32 = cv2.cvtColor(np.transpose(np_img_01.astype(np.float32), (1, 2, 0)), cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(crop_save_path), img_c_bgr_float32)
                
                row_dict['x_clean'] = x_clean 
                valid_records.append(row_dict)
                id_counts[class_id] = id_counts.get(class_id, 0) + 1

    print(f"-> Immagini pronte per l'attacco: {len(valid_records)} ({cached_count} da cache TIFF)")

    # =========================================================================
    # FASE 2: GENERAZIONE DEGLI ATTACCHI C&W (BATCHED)
    # =========================================================================
    print(f"\n[FASE 2] Avvio Generazione C&W Untargeted (lr={CW_LR}, max_iter={CW_MAX_ITER})...")
    
    output_base_dir.mkdir(parents=True, exist_ok=True)
    tracker_output_path = output_base_dir / "tracker_cw_untargeted.csv"
    
    tracker_records = []
    
    attack = CarliniLInfMethodPyTorch(
        classifier=classifier, 
        targeted=False, # Untargeted!
        max_iter=CW_MAX_ITER,         
        learning_rate=CW_LR,
        batch_size=BATCH_SIZE,
        verbose=False
    )
    
    for start_idx in tqdm(range(0, len(valid_records), BATCH_SIZE), desc="Generazione Batch C&W"):
        batch_records = valid_records[start_idx:start_idx + BATCH_SIZE]
        
        batch_x_list = []
        batch_y_list = []
        
        # Estrazione X e Y
        for r in batch_records:
            batch_x_list.append(r['x_clean'][0])
            y_onehot = get_one_hot_target(r['true_facenet_id'], num_classes=8631)[0]
            batch_y_list.append(y_onehot)
            
        batch_x = np.stack(batch_x_list).astype(np.float64)
        batch_y = np.stack(batch_y_list).astype(np.float64)
        
        # 1. ATTACCO DI MASSA 
        x_adv = attack.generate(x=batch_x, y=batch_y)
        
        # 2. INFERENZA ON-THE-FLY (Per tracciare predizioni e confidenze finali)
        with torch.no_grad():
            x_adv_tensor = torch.tensor(x_adv).to(device)
            # Normalizzazione per la rete [-1, 1]
            adv_logits = resnet(x_adv_tensor * 2.0 - 1.0)
            adv_preds = torch.argmax(adv_logits, dim=1).cpu().numpy()
            adv_probs = torch.nn.functional.softmax(adv_logits, dim=1)
            
            clean_logits = resnet(torch.tensor(batch_x).to(device) * 2.0 - 1.0)
            clean_probs = torch.nn.functional.softmax(clean_logits, dim=1)

        # 3. SALVATAGGIO
        for i, adv_img_np in enumerate(x_adv):
            rec = batch_records[i]
            orig_path = Path(rec['image_path'])
            identity_dir_name = orig_path.parent.name
            orig_filename_no_ext = orig_path.stem 
            true_facenet_id = rec['true_facenet_id']
            
            img_c_plot = np.transpose(batch_x[i], (1, 2, 0))
            img_a_plot = np.transpose(adv_img_np, (1, 2, 0))
            
            actual_linf = calculate_linf(img_c_plot, img_a_plot)
            mean_abs_perturbation = float(np.mean(np.abs(img_a_plot - img_c_plot)))

            out_img_dir = output_base_dir / identity_dir_name
            out_img_dir.mkdir(parents=True, exist_ok=True)
            
            # --- SALVATAGGIO IN TIFF 32-BIT FLOAT ---
            adv_filename = f"{orig_filename_no_ext}.tiff"
            adv_save_path = out_img_dir / adv_filename
            
            # Convertiamo da 64-bit a 32-bit per il salvataggio (OpenCV gestisce float32)
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

    # --- 4. SALVATAGGIO TRACKER GLOBALE ---
    df_tracker = pd.DataFrame(tracker_records)
    df_tracker.to_csv(tracker_output_path, index=False)
    print(f"\n[OK] Generazione completata! Tracker salvato in: {tracker_output_path}")

if __name__ == "__main__":
    main()