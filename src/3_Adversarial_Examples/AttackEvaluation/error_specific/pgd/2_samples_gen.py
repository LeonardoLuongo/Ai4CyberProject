import os
import sys
import cv2
import numpy as np
import pandas as pd
import torch
import json

os.environ["TORCH_CUDNN_V8_API_ENABLED"] = "0" 
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

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
from art.attacks.evasion import ProjectedGradientDescent

from src.util.identity_mapper import IdentityMapper
from src.util.basic_img.metrics import calculate_linf
from src.util.attack_error_specific_utils import select_target_label, get_one_hot_target

def main():
    print("======================================================")
    print(" GENERATORE CAMPIONI: PGD TARGETED (Error-Specific)   ")
    print("======================================================\n")

    base_dir = PROJECT_ROOT
    print(f"-> Project Root impostata a: {base_dir}")

    # --- 1. CONFIGURAZIONE PATH E PARAMETRI ---
    csv_path = base_dir / "dataset" / "clean" / "splits" / "manifest.csv"
    meta_csv_path = base_dir / "dataset" / "clean" / "splits" / "identity_meta.csv"
    
    output_base_dir = base_dir / "dataset" / "attacks" / "NN1" / "error_specific" / "pgd"
    cropped_clean_dir = base_dir / "dataset" / "clean_cropped" / "NN1"
    
    # PARAMETRI PGD TARGETED
    EPSILONS = [0.001, 0.005, 0.01, 0.02, 0.04, 0.05]
    MAX_ITER  = 10     
    STEP_MULT = 1.5    
    NUM_INIT  = 1      
    BATCH_SIZE = 64   

    # Strategie Targeted
    STRATEGIES = ["next_best", "least-likely", "random"]
    
    # (Manteniamo il caricamento json per compatibilità futura, anche se non usiamo i rr_ ora)
    rr_json_path = base_dir / "dataset" / "clean" / "splits" / "rr_subsets.json"
    with open(rr_json_path, 'r') as f:
        rr_subsets = json.load(f)

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
    # FASE 1: MTCNN E SCREMATURA (Filtro Zero-Shot e Cache TIFF 32-bit)
    # =========================================================================
    print("\n[FASE 1] Ritaglio MTCNN, Scrematura e Salvataggio Clean Cropped (TIFF)...")
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

            row_dict = row.to_dict()
            row_dict['true_facenet_id'] = facenet_id
            row_dict['cropped_image_path'] = crop_save_path.relative_to(base_dir).as_posix()

            # I TIFF in cache sono float32 BGR nel range [0, 1]
            if crop_save_path.exists():
                img_bgr_float32 = cv2.imread(str(crop_save_path), cv2.IMREAD_UNCHANGED)
                if img_bgr_float32 is None:
                    continue
                img_rgb_float32 = cv2.cvtColor(img_bgr_float32, cv2.COLOR_BGR2RGB)
                x_clean = np.expand_dims(np.transpose(img_rgb_float32.astype(np.float32), (2, 0, 1)), axis=0)

                # Ricalcoliamo i clean_logits, essenziali per gli attacchi mirati
                t_clean = torch.tensor(x_clean * 2.0 - 1.0).to(device)
                best_logits = resnet(t_clean).cpu().numpy()
                row_dict['clean_logits'] = best_logits 

                row_dict['x_clean'] = x_clean
                valid_records.append(row_dict)
                cached_count += 1
                continue
                
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
                best_logits = logits_all[match_idx].cpu().numpy()
                
                np_img_01 = (best_face_tensor.cpu().numpy() + 1.0) / 2.0
                x_clean = np.expand_dims(np_img_01, axis=0) 

                # Salvataggio TIFF 32-bit (float32) in BGR
                img_c_bgr_float32 = cv2.cvtColor(np.transpose(np_img_01, (1, 2, 0)).astype(np.float32), cv2.COLOR_RGB2BGR)
                if not cv2.imwrite(str(crop_save_path), img_c_bgr_float32):
                    print(f"\n[ERRORE FATALE] Impossibile salvare clean TIFF: {crop_save_path}")
                    continue

                row_dict['clean_logits'] = np.expand_dims(best_logits, axis=0)
                row_dict['x_clean'] = x_clean
                valid_records.append(row_dict)

    total_valid = len(valid_records)
    print(f"-> Immagini d'oro pronte: {total_valid} ({cached_count} caricate da cache TIFF)")

    # =========================================================================
    # FASE 2: GENERAZIONE DEGLI ATTACCHI PGD TARGETED (BATCH DIRETTO)
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
            
            eps_tracker_records = []
            eps_step = (eps / MAX_ITER) * STEP_MULT
            
            print(f"\n[>>>] Generazione Targeted Epsilon = {eps:.3f} | Step = {eps_step:.4f} | Iter = {MAX_ITER}")
            
            attack = ProjectedGradientDescent(
                estimator=classifier, 
                eps=eps, 
                eps_step=eps_step, 
                max_iter=MAX_ITER, 
                num_random_init=NUM_INIT,
                targeted=True,
                batch_size=BATCH_SIZE,
                verbose=False
            )
            
            for start_idx in tqdm(range(0, len(valid_records), BATCH_SIZE), desc=f"Batch {eps_str} ({strategy})"):
                end_idx = min(start_idx + BATCH_SIZE, total_valid)
                batch_records = valid_records[start_idx:end_idx]
                
                batch_x_list = []
                batch_y_list = []
                targets_memory = [] 
                
                for row in batch_records:
                    x_clean = row['x_clean']
                    true_facenet_id = row['true_facenet_id']
                    clean_logits = row['clean_logits']
                    
                    strat_str = strategy.replace("_", "-")
                    t_id = select_target_label(clean_logits, true_facenet_id, strategy=strat_str, num_classes=mapper.get_num_training_classes())
                    y_target_onehot = get_one_hot_target(t_id, num_classes=mapper.get_num_training_classes())
                    
                    batch_x_list.append(x_clean[0])
                    batch_y_list.append(y_target_onehot[0])
                    targets_memory.append(t_id)

                batch_x_array = np.stack(batch_x_list).astype(np.float32)
                batch_y_array = np.stack(batch_y_list).astype(np.float32)
                
                # Generazione avversaria PGD (Restituisce float32)
                x_adv_batch = attack.generate(x=batch_x_array, y=batch_y_array)
                x_adv_batch = np.clip(x_adv_batch, 0.0, 1.0).astype(np.float32) # Clip di sicurezza tra 0.0 e 1.0
                
                for i, row in enumerate(batch_records):
                    target_label_8631 = targets_memory[i]
                    
                    img_c_plot = np.transpose(batch_x_array[i], (1, 2, 0))
                    img_a_plot = np.transpose(x_adv_batch[i], (1, 2, 0))
                    
                    actual_linf = calculate_linf(img_c_plot, img_a_plot)
                    mean_abs_perturbation = float(np.mean(np.abs(img_a_plot - img_c_plot)))
                    
                    source_img_path = base_dir / row['image_path']
                    identity_dir_name = source_img_path.parent.name
                    orig_filename_no_ext = source_img_path.stem 

                    out_img_dir = eps_dir / identity_dir_name
                    out_img_dir.mkdir(parents=True, exist_ok=True)
                    
                    if strategy.startswith("rr_"):
                        adv_filename = f"{orig_filename_no_ext}_to_{target_label_8631}.tiff"
                    else:
                        adv_filename = f"{orig_filename_no_ext}.tiff"
                        
                    adv_save_path = out_img_dir / adv_filename
                    
                    # Salvataggio TIFF 32-bit (float32) in BGR
                    img_a_bgr_float32 = cv2.cvtColor(img_a_plot, cv2.COLOR_RGB2BGR)
                    if not cv2.imwrite(str(adv_save_path), img_a_bgr_float32):
                        print(f"\n[ERRORE FATALE] Impossibile salvare adversarial TIFF: {adv_save_path}")
                        continue

                    rel_source = row['cropped_image_path']
                    rel_adv = adv_save_path.relative_to(base_dir).as_posix()

                    eps_tracker_records.append({
                        "attack_type": "pgd_error_specific",
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

            df_tracker = pd.DataFrame(eps_tracker_records)
            df_tracker.to_csv(eps_dir / f"tracker_{eps_str}.csv", index=False)
            print(f"-> Tracker salvato in: {eps_dir / f'tracker_{eps_str}.csv'}")
        
    print("\n[OK] Processo Error-Specific completato con successo (TIFF 32-bit)!")

if __name__ == "__main__":
    main()