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

# Importiamo la tua classe SOTA e le utility
from util.cw_benchmarks.cw_pytorch import CarliniLInfMethodPyTorch
from util.identity_mapper import IdentityMapper
from util.basic_img.metrics import calculate_linf
from util.attack_error_specific_utils import select_target_label, get_one_hot_target
from util.plot.utils_plot_shared import plot_adversarial_showcase

# ==========================================
# WRAPPER PURE-FLOAT64
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
    print(" GENERATORE CAMPIONI: C&W TARGETED (64-BIT PURITY)    ")
    print("======================================================\n")

    # --- 1. CONFIGURAZIONE PATH E PARAMETRI ---
    base_dir = Path.cwd()
    csv_path = base_dir / "dataset" / "clean" / "splits" / "manifest.csv"
    meta_csv_path = base_dir / "dataset" / "clean" / "splits" / "identity_meta.csv"
    output_base_dir = base_dir / "dataset" / "attacks" / "NN1" / "error_specific" / "cw"
    
    cropped_clean_dir = base_dir / "dataset" / "clean_cropped" / "NN1"
    
    # =========================================================================
    # PARAMETRI D'ORO DELLO SCOUTING
    # =========================================================================
    CW_LR = 0.015
    CW_MAX_ITER = 15
    
    BATCH_SIZE = 224 # Mantieni a 32/64 per gestire i Float64
    
    SAMPLES_PER_ID = 10           # Di default 10 per estrarre tutte le foto
    MAX_TOTAL_IMAGES = 1_000     # Esecuzione completa
    
    STRATEGIES = ["next_best", "least-likely", "random"]
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
    resnet.double() # <-- FORZIAMO LA RETE IN 64-BIT
    
    art_resnet_shield = ARTFloat64Wrapper(resnet).eval()

    classifier = PyTorchClassifier(
        model=art_resnet_shield, clip_values=(0.0, 1.0), loss=nn.CrossEntropyLoss(), optimizer=None,
        input_shape=(3, 160, 160), nb_classes=8631, preprocessing=(0.5, 0.5), 
        device_type='gpu' if torch.cuda.is_available() else 'cpu'
    )

    df_clean = pd.read_csv(csv_path)

    # =========================================================================
    # FASE 1: MTCNN E SCREMATURA (Filtro Zero-Shot e Salvataggio Cache TIFF)
    # =========================================================================
    print(f"\n[FASE 1] Estrazione Dataset (Max {MAX_TOTAL_IMAGES} totali, {SAMPLES_PER_ID} per ID)...")
    valid_records = []
    cached_count = 0
    id_counts = {}
    
    with torch.no_grad():
        for index, row in tqdm(df_clean.iterrows(), total=len(df_clean), desc="Pre-Inferenza"):
            
            if len(valid_records) >= MAX_TOTAL_IMAGES:
                break
                
            class_id = str(row['identity_id'])
            if id_counts.get(class_id, 0) >= SAMPLES_PER_ID:
                continue
            
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
                
                # Salviamo in Float64 in RAM per ART
                x_clean = np.expand_dims(np.transpose(img_rgb_float32, (2, 0, 1)), axis=0).astype(np.float64)
                
                # Calcoliamo i logit al volo per la logica Targeted
                best_logits = resnet(torch.tensor(x_clean).to(device) * 2.0 - 1.0).cpu().numpy()
                
                row_dict['clean_logits'] = best_logits
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
            
            faces_device = faces.to(device).double()
            preds_all = torch.argmax(resnet(faces_device), dim=1).cpu().numpy()
            
            if facenet_id in preds_all:
                match_idx = int(np.where(preds_all == facenet_id)[0][0])
                best_face_tensor = faces_device[match_idx]
                
                # Estraiamo i logits per il targeting
                best_logits = resnet(best_face_tensor.unsqueeze(0)).cpu().numpy()
                
                np_img_01 = (best_face_tensor.cpu().numpy() + 1.0) / 2.0
                x_clean = np.expand_dims(np_img_01, axis=0) # Float64
                
                img_c_bgr_float32 = cv2.cvtColor(np.transpose(np_img_01.astype(np.float32), (1, 2, 0)), cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(crop_save_path), img_c_bgr_float32)
                
                row_dict['clean_logits'] = best_logits
                row_dict['x_clean'] = x_clean 
                valid_records.append(row_dict)
                id_counts[class_id] = id_counts.get(class_id, 0) + 1

    print(f"-> Immagini pronte per l'attacco: {len(valid_records)} ({cached_count} da cache TIFF)")

    # =========================================================================
    # FASE 2: GENERAZIONE DEGLI ATTACCHI C&W (BATCHED)
    # =========================================================================
    
    # Istanziamo LA TUA CLASSE in modalità Targeted
    attack = CarliniLInfMethodPyTorch(
        classifier=classifier, 
        targeted=True, 
        max_iter=CW_MAX_ITER,         
        learning_rate=CW_LR,
        batch_size=BATCH_SIZE,
        verbose=False
    )
    
    for strategy in STRATEGIES:
        print(f"\n======================================================")
        print(f" AVVIO C&W BATCHED: STRATEGIA {strategy.upper()}")
        print(f"======================================================")
        
        strat_dir = output_base_dir / strategy
        strat_dir.mkdir(parents=True, exist_ok=True)
        strat_tracker_records = []
        
        for start_idx in tqdm(range(0, len(valid_records), BATCH_SIZE), desc=f"Generazione C&W ({strategy})"):
            batch_records = valid_records[start_idx:start_idx + BATCH_SIZE]
            
            batch_x_list = []
            batch_y_list = []
            targets_memory = []
            
            for r in batch_records:
                batch_x_list.append(r['x_clean'][0])
                
                # Calcolo del target specifico per ogni immagine
                strat_str = strategy.replace("_", "-")
                t_id = select_target_label(r['clean_logits'], r['true_facenet_id'], strategy=strat_str, num_classes=8631)
                
                y_onehot = get_one_hot_target(t_id, num_classes=8631)[0]
                batch_y_list.append(y_onehot)
                targets_memory.append(t_id)
                
            batch_x = np.stack(batch_x_list).astype(np.float64)
            batch_y = np.stack(batch_y_list).astype(np.float64)
            
            # 1. ATTACCO 
            x_adv = attack.generate(x=batch_x, y=batch_y)
            
            # 2. INFERENZA ON-THE-FLY
            with torch.no_grad():
                x_adv_tensor = torch.tensor(x_adv).to(device)
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
                target_label_8631 = targets_memory[i]
                
                img_c_plot = np.transpose(batch_x[i], (1, 2, 0))
                img_a_plot = np.transpose(adv_img_np, (1, 2, 0))
                
                actual_linf = calculate_linf(img_c_plot, img_a_plot)
                mean_abs_perturbation = float(np.mean(np.abs(img_a_plot - img_c_plot)))

                out_img_dir = strat_dir / identity_dir_name
                out_img_dir.mkdir(parents=True, exist_ok=True)
                
                # --- PLOT PERFECT SHOWCASE (Solo al primissimo campione) ---
                if start_idx == 0 and i == 0:
                    showcase_dir = base_dir / "plots" / "3_Adversarial_Examples" / "error_specific" / "cw" / strategy / "visual_progression"
                    showcase_dir.mkdir(parents=True, exist_ok=True)
                    plot_path = str(showcase_dir / f"showcase_perfect_float32.png")
                    
                    plot_adversarial_showcase(
                        img_c_plot, img_a_plot, 
                        f"Orig: {rec['identity_name']}", f"Target: ID {target_label_8631}", 
                        True, plot_path
                    )
                
                # --- SALVATAGGIO TIFF ---
                adv_filename = f"{orig_filename_no_ext}.tiff"
                adv_save_path = out_img_dir / adv_filename
                
                img_a_bgr_float32 = cv2.cvtColor(img_a_plot.astype(np.float32), cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(adv_save_path), img_a_bgr_float32)

                rel_source = Path(rec['cropped_image_path']).relative_to(base_dir).as_posix()
                rel_adv = adv_save_path.relative_to(base_dir).as_posix()
                
                # Confidenze sul TARGET e sull'ORIGINALE
                adv_conf_target = float(adv_probs[i, target_label_8631].cpu().numpy())
                clean_conf_target = float(clean_probs[i, target_label_8631].cpu().numpy())

                strat_tracker_records.append({
                    "attack_type": "cw_linf",
                    "max_iter": CW_MAX_ITER,
                    "learning_rate": CW_LR,
                    "targeted": True,
                    "target_strategy": strategy,
                    "target_class": target_label_8631,
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
                    "clean_target_confidence": clean_conf_target,
                    "adv_target_confidence": adv_conf_target
                })

        # --- 4. SALVATAGGIO TRACKER GLOBALE ---
        tracker_path = strat_dir / f"tracker_{strategy}.csv"
        df_tracker = pd.DataFrame(strat_tracker_records)
        df_tracker.to_csv(tracker_path, index=False)
        print(f"-> Tracker C&W {strategy} salvato in: {tracker_path}")

    print("\n[OK] Generazione Dataset Targeted completata!")

if __name__ == "__main__":
    main()