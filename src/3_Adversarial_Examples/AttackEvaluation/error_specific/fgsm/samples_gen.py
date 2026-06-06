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
from art.attacks.evasion import FastGradientMethod

from util.identity_mapper import IdentityMapper
from util.basic_img.metrics import calculate_linf
from util.attack_error_specific_utils import select_target_label, get_one_hot_target

def main():
    print("======================================================")
    print(" GENERATORE CAMPIONI: FGSM TARGETED (Error-Specific)  ")
    print("======================================================\n")

    # --- 1. CONFIGURAZIONE PATH E PARAMETRI ---
    base_dir = Path(os.getcwd())
    csv_path = base_dir / "dataset" / "clean" / "splits" / "manifest.csv"
    meta_csv_path = base_dir / "dataset" / "clean" / "splits" / "identity_meta.csv"
    output_base_dir = base_dir / "dataset" / "attacks" / "NN1" / "error_specific" / "fgsm"
    
    # NUOVA CARTELLA: Per salvare le immagini pulite già ritagliate
    cropped_clean_dir = base_dir / "dataset" / "clean_cropped" / "NN1"
    # Epsilon da testare (incluso il 0.20 per vedere l'asintoto)
    epsilons = [0.025, 0.05, 0.075, 0.10, 0.15, 0.20]
    strategies = ["next_best", "random", "rr_lookalikes", "rr_extremes", "rr_diversity", "least-likely"]
    
    rr_json_path = base_dir / "dataset" / "clean" / "splits" / "rr_subsets.json"
    with open(rr_json_path, 'r') as f:
        rr_subsets = json.load(f)

    
    if not csv_path.exists() or not meta_csv_path.exists():
        raise FileNotFoundError("Errore: manifest.csv o identity_meta.csv mancanti.")

    # --- 2. INIZIALIZZAZIONE MODELLI E MAPPER ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"-> Inizializzazione Reti su {device}...")
    
    mapper = IdentityMapper(meta_csv_path)
    
    # MTCNN con keep_all=True per massimizzare la detection
    mtcnn = MTCNN(image_size=160, margin=0, keep_all=True, post_process=True, device=device)
    
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    resnet.classify = True 

    # Wrapper ART: accetta input [0, 1] e fa (x - 0.5) / 0.5 -> [-1, 1] per FaceNet
    classifier = PyTorchClassifier(
        model=resnet, clip_values=(0.0, 1.0), loss=nn.CrossEntropyLoss(), optimizer=None,
        input_shape=(3, 160, 160), nb_classes=8631, preprocessing=(0.5, 0.5), 
        device_type='gpu' if torch.cuda.is_available() else 'cpu'
    )

    df_clean = pd.read_csv(csv_path)
    print(f"-> Trovate {len(df_clean)} immagini totali nel manifest.")

    # =========================================================================
    # FASE 1: MTCNN E SCREMATURA (Salvataggio del Clean Cropped Dataset)
    # =========================================================================
    print("\n[FASE 1] Ritaglio MTCNN, Scrematura e Salvataggio Clean Cropped...")
    valid_records = []
    cached_count = 0
    
    with torch.no_grad():
        for index, row in tqdm(df_clean.iterrows(), total=len(df_clean), desc="Pre-Inferenza"):
            class_id = str(row['identity_id'])
            
            # 1. È nel training set di VGGFace2?
            facenet_id = mapper.get_facenet_id_by_class_id(class_id)
            if facenet_id == -1:
                continue
                
            source_img_path = Path(base_dir / row['image_path'])
            identity_dir_name = source_img_path.parent.name
            img_filename = f"{source_img_path.stem}.tiff"

            out_crop_dir = cropped_clean_dir / identity_dir_name
            out_crop_dir.mkdir(parents=True, exist_ok=True)
            crop_save_path = out_crop_dir / img_filename

            row_dict = row.to_dict()
            row_dict['true_facenet_id'] = facenet_id
            row_dict['cropped_image_path'] = crop_save_path.relative_to(base_dir).as_posix()

            if crop_save_path.exists():
                img_bgr_float32 = cv2.imread(str(crop_save_path), cv2.IMREAD_UNCHANGED)
                if img_bgr_float32 is None:
                    continue
                img_rgb_float32 = cv2.cvtColor(img_bgr_float32, cv2.COLOR_BGR2RGB)
                x_clean = np.expand_dims(
                    np.transpose(img_rgb_float32.astype(np.float32), (2, 0, 1)),
                    axis=0,
                )
                clean_tensor = torch.from_numpy(x_clean).to(device)
                clean_logits = resnet(clean_tensor * 2.0 - 1.0).cpu().numpy()

                row_dict['clean_logits'] = clean_logits
                row_dict['x_clean'] = x_clean
                valid_records.append(row_dict)
                cached_count += 1
                continue
                
            try:
                img_pil = Image.open(str(source_img_path)).convert('RGB')
            except Exception:
                continue
            
            # MTCNN restituisce [N, 3, 160, 160] facce
            faces = mtcnn(img_pil)
            if faces is None: continue
            
            faces = faces.to(device)
            logits_all = resnet(faces)
            preds_all = torch.argmax(logits_all, dim=1).cpu().numpy()
            
            # IL FILTRO CHIRURGICO: Cerchiamo SE la faccia giusta è tra quelle trovate
            if facenet_id in preds_all:
                # Troviamo l'indice della prima faccia corretta nell'array di N facce
                match_idx = np.where(preds_all == facenet_id)[0][0]
                
                # Isoliamo il tensore, i logits e convertiamo in formato ART
                best_face_tensor = faces[match_idx]
                best_logits = logits_all[match_idx].cpu().numpy()
                
                # Da [-1, 1] a [0, 1]
                np_img_01 = (best_face_tensor.cpu().numpy() + 1.0) / 2.0
                x_clean = np.expand_dims(np_img_01, axis=0) # Shape: (1, 3, 160, 160)
                
                # --- NOVITÀ: Salvataggio fisico dell'immagine Clean Cropped ---
                out_crop_dir = cropped_clean_dir / identity_dir_name
                out_crop_dir.mkdir(parents=True, exist_ok=True)
                crop_save_path = out_crop_dir / img_filename
                
                # Salva usando OpenCV (richiede conversione RGB -> BGR)
                img_c_bgr_float32 = cv2.cvtColor(
                    np.transpose(np_img_01, (1, 2, 0)).astype(np.float32),
                    cv2.COLOR_RGB2BGR,
                )
                if not cv2.imwrite(str(crop_save_path), img_c_bgr_float32):
                    print(f"\n[ERRORE] Impossibile salvare clean TIFF: {crop_save_path}")
                    continue
                
                # Espandiamo dim per compatibilità con select_target_label: (1, 8631)
                row_dict['clean_logits'] = np.expand_dims(best_logits, axis=0) 
                row_dict['x_clean'] = x_clean 
                
                valid_records.append(row_dict)

    print(f"-> Immagini d'oro pronte per l'attacco: {len(valid_records)} su {len(df_clean)} "
          f"({(len(valid_records)/len(df_clean))*100:.1f}%, {cached_count} da cache TIFF)")

    # =========================================================================
    # FASE 2: GENERAZIONE DEGLI ATTACCHI (Multi-Strategia)
    # =========================================================================
    
    for strategy in strategies:
        print(f"\n======================================================")
        print(f" AVVIO GENERAZIONE STRATEGIA: {strategy.upper()}")
        print(f"======================================================")
        
        strat_dir = output_base_dir / strategy
        strat_dir.mkdir(parents=True, exist_ok=True)
        
        for eps in epsilons:
            eps_str = f"eps_{eps:.3f}".replace('.', '_')
            eps_dir = strat_dir / eps_str
            eps_dir.mkdir(exist_ok=True)
            
            eps_tracker_records = []
            
            print(f"\n[>>>] Generazione per Epsilon = {eps:.3f}")
            attack = FastGradientMethod(estimator=classifier, eps=eps, targeted=True)
            
            if strategy.startswith("rr_"):
                allowed_ids = rr_subsets[strategy]
                active_records = [r for r in valid_records if r['identity_id'] in allowed_ids]
                pbar = tqdm(active_records, desc=f"Attacco {eps_str} ({strategy})")
            else:
                active_records = valid_records
                pbar = tqdm(active_records, desc=f"Attacco {eps_str}")

            for row in pbar:
                x_clean = row['x_clean']
                true_facenet_id = row['true_facenet_id']
                clean_logits = row['clean_logits']
                
                source_img_path = Path(base_dir / row['image_path'])
                identity_dir_name = source_img_path.parent.name
                orig_filename = source_img_path.stem

                targets_to_attack = []
                
                # Logica Multi-Target
                if strategy.startswith("rr_"):
                    for tgt_id in allowed_ids:
                        if tgt_id != row['identity_id']:
                            tgt_facenet_id = mapper.get_facenet_id_by_class_id(tgt_id)
                            targets_to_attack.append(tgt_facenet_id)
                else:
                    strat_str = "next-best" if strategy == "next_best" else strategy
                    t_id = select_target_label(clean_logits, true_facenet_id, strategy=strat_str, num_classes=mapper.get_num_training_classes())
                    targets_to_attack.append(t_id)

                for target_label_8631 in targets_to_attack:
                    y_target_onehot = get_one_hot_target(target_label_8631, num_classes=mapper.get_num_training_classes())

                    x_adv = attack.generate(x=x_clean, y=y_target_onehot)
                    
                    img_c_plot = np.transpose(x_clean[0], (1, 2, 0))
                    img_a_plot = np.transpose(x_adv[0], (1, 2, 0))
                    
                    actual_linf = calculate_linf(img_c_plot, img_a_plot)
                    mean_abs_perturbation = float(np.mean(np.abs(img_a_plot - img_c_plot)))

                    out_img_dir = eps_dir / identity_dir_name
                    out_img_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Se RR, aggiungiamo l'ID target al nome per non sovrascrivere i file!
                    adv_filename = f"{orig_filename}_to_{target_label_8631}.tiff" if strategy.startswith("rr_") else f"{orig_filename}.tiff"
                    adv_save_path = out_img_dir / adv_filename
                    
                    img_a_bgr_float32 = cv2.cvtColor(
                        np.clip(img_a_plot, 0.0, 1.0).astype(np.float32),
                        cv2.COLOR_RGB2BGR,
                    )
                    if not cv2.imwrite(str(adv_save_path), img_a_bgr_float32):
                        print(f"\n[ERRORE] Impossibile salvare adversarial TIFF: {adv_save_path}")
                        continue

                    rel_source = Path(row['cropped_image_path']).as_posix()
                    rel_adv = adv_save_path.relative_to(base_dir).as_posix()

                    eps_tracker_records.append({
                        "attack_type": "fgsm",
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

            # --- 6. SALVATAGGIO TRACKER LOCALE A FINE EPSILON ---
            eps_tracker_path = eps_dir / f"tracker_{eps_str}.csv"
            df_tracker = pd.DataFrame(eps_tracker_records)
            df_tracker.to_csv(eps_tracker_path, index=False)
            print(f"-> Tracker salvato in: {eps_tracker_path}")
        
    print("\n[OK] Processo completato per tutte le strategie ed Epsilon!")

if __name__ == "__main__":
    main()
