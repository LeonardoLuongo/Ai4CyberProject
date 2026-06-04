import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path
from PIL import Image

# =========================================================================
# RISOLUZIONE ROBUSTA DEI PATH (Previene gli errori di VS Code)
# =========================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[5]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from facenet_pytorch import InceptionResnetV1, MTCNN
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import BasicIterativeMethod

# Import corretti partendo dalla root
from src.util.identity_mapper import IdentityMapper
from src.util.basic_img.metrics import calculate_linf
from src.util.attack_error_specific_utils import select_target_label, get_one_hot_target


def project_relative_path(path: Path, base_dir: Path) -> str:
    try:
        return path.relative_to(base_dir).as_posix()
    except ValueError:
        return str(path)


def save_rgb_image(path: Path, image_hwc_01: np.ndarray) -> None:
    """Salva in modo robusto su Windows anche con path Unicode."""
    path.parent.mkdir(parents=True, exist_ok=True)

    image_hwc_01 = np.asarray(image_hwc_01)
    if image_hwc_01.ndim != 3 or image_hwc_01.shape[2] != 3:
        raise ValueError(f"shape immagine non valida: {image_hwc_01.shape}")

    image_uint8 = np.rint(np.clip(image_hwc_01, 0.0, 1.0) * 255.0).astype(np.uint8)
    Image.fromarray(image_uint8).save(path)

    if not path.exists() or path.stat().st_size == 0:
        raise OSError(f"file non creato correttamente: {path}")


def main():
    print("======================================================")
    print(" GENERATORE CAMPIONI: BIM TARGETED (BULLETPROOF SAVE) ")
    print("======================================================\n")

    # Fissiamo la base_dir alla vera ROOT del progetto
    base_dir = PROJECT_ROOT
    print(f"-> Project Root impostata a: {base_dir}")

    # --- 1. CONFIGURAZIONE PATH E PARAMETRI ---
    csv_path = base_dir / "dataset" / "clean" / "splits" / "manifest.csv"
    meta_csv_path = base_dir / "dataset" / "clean" / "splits" / "identity_meta.csv"
    
    output_base_dir = base_dir / "dataset" / "attacks" / "error_specific" / "bim"
    cropped_clean_dir = base_dir / "dataset" / "clean_cropped" / "NN1"
    
    epsilons = [0.025, 0.05, 0.075, 0.10, 0.15, 0.20]
    
    # STRATEGIE AGGIORNATE
    strategies = ["next_best", "random", "least-likely"]
    
    # =========================================================================
    # PARAMETRI OTTIMIZZATI (Regola di Madry)
    # =========================================================================
    BIM_MAX_ITER = 10 
    BIM_MULT = 2.5 
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
    # FASE 1: MTCNN E SCREMATURA (Salvataggio Fisico Controllato)
    # =========================================================================
    print("\n[FASE 1] Ritaglio MTCNN, Scrematura e Salvataggio Clean Cropped...")
    valid_records = []
    
    with torch.no_grad():
        for index, row in tqdm(df_clean.iterrows(), total=len(df_clean), desc="Pre-Inferenza"):
            class_id = str(row['identity_id'])
            facenet_id = mapper.get_facenet_id_by_class_id(class_id)
            if facenet_id == -1: continue
                
            source_img_path = base_dir / row['image_path']
            identity_dir_name = source_img_path.parent.name
            img_filename = source_img_path.name
                
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
                
                np_img_01 = np.clip((best_face_tensor.cpu().numpy() + 1.0) / 2.0, 0.0, 1.0)
                x_clean = np.expand_dims(np_img_01, axis=0) 
                
                out_crop_dir = cropped_clean_dir / identity_dir_name
                out_crop_dir.mkdir(parents=True, exist_ok=True)
                crop_save_path = out_crop_dir / img_filename
                
                img_c_plot = np.transpose(np_img_01, (1, 2, 0))
                try:
                    save_rgb_image(crop_save_path, img_c_plot)
                except Exception as exc:
                    rel_crop_path = project_relative_path(crop_save_path, base_dir)
                    print(f"\n[ERRORE FATALE] Impossibile salvare foto clean in: {rel_crop_path} -> {exc}")
                    continue
                
                row_dict = row.to_dict()
                row_dict['true_facenet_id'] = facenet_id
                row_dict['clean_logits'] = np.expand_dims(best_logits, axis=0) 
                row_dict['x_clean'] = x_clean 
                row_dict['cropped_image_path'] = crop_save_path.relative_to(base_dir).as_posix() 
                
                valid_records.append(row_dict)

    total_valid = len(valid_records)
    print(f"-> Immagini d'oro pronte per l'attacco: {total_valid}")

    # =========================================================================
    # FASE 2: GENERAZIONE DEGLI ATTACCHI (BATCH DIRETTO)
    # =========================================================================
    
    for strategy in strategies:
        print(f"\n======================================================")
        print(f" AVVIO GENERAZIONE STRATEGIA: {strategy.upper()}")
        print(f"======================================================")
        
        strat_dir = output_base_dir / strategy
        strat_dir.mkdir(parents=True, exist_ok=True)
        
        strat_str = "next-best" if strategy == "next_best" else strategy
        
        for eps in epsilons:
            eps_str = f"eps_{eps:.3f}".replace('.', '_')
            eps_dir = strat_dir / eps_str
            eps_dir.mkdir(exist_ok=True)
            
            eps_tracker_records = []
            
            current_eps_step = (eps / BIM_MAX_ITER) * BIM_MULT
            print(f"\n[>>>] Generazione Epsilon = {eps:.3f} | Step = {current_eps_step:.4f} | Iter = {BIM_MAX_ITER}")
            
            attack = BasicIterativeMethod(
                estimator=classifier, 
                eps=eps, 
                eps_step=current_eps_step,
                max_iter=BIM_MAX_ITER,
                targeted=True,
                batch_size=BATCH_SIZE 
            )
            
            # CICLO SUI BATCH 
            for start_idx in tqdm(range(0, total_valid, BATCH_SIZE), desc=f"Batch Processing {eps_str}"):
                end_idx = min(start_idx + BATCH_SIZE, total_valid)
                batch_records = valid_records[start_idx:end_idx]
                
                batch_x_list = []
                batch_y_list = []
                batch_targets_raw = []
                
                for row in batch_records:
                    batch_x_list.append(row['x_clean'][0])
                    
                    t_id = select_target_label(
                        row['clean_logits'], 
                        row['true_facenet_id'], 
                        strategy=strat_str, 
                        num_classes=mapper.get_num_training_classes()
                    )
                    batch_targets_raw.append(t_id)
                    batch_y_list.append(get_one_hot_target(t_id, num_classes=mapper.get_num_training_classes())[0])
                    
                batch_x_array = np.stack(batch_x_list)
                batch_y_array = np.stack(batch_y_list)
                
                x_adv_batch = attack.generate(x=batch_x_array, y=batch_y_array)
                
                for i, row in enumerate(batch_records):
                    target_label_8631 = batch_targets_raw[i]
                    x_clean_single = batch_x_array[i]
                    x_adv_single = x_adv_batch[i]
                    
                    img_c_plot = np.transpose(x_clean_single, (1, 2, 0))
                    img_a_plot = np.clip(np.transpose(x_adv_single, (1, 2, 0)), 0.0, 1.0)
                    
                    actual_linf = calculate_linf(img_c_plot, img_a_plot)
                    mean_abs_perturbation = float(np.mean(np.abs(img_a_plot - img_c_plot)))
                    
                    source_img_path = base_dir / row['image_path']
                    identity_dir_name = source_img_path.parent.name
                    orig_filename = source_img_path.stem 

                    out_img_dir = eps_dir / identity_dir_name
                    out_img_dir.mkdir(parents=True, exist_ok=True)
                    
                    adv_filename = f"{orig_filename}.jpg"
                    adv_save_path = out_img_dir / adv_filename
                    
                    try:
                        save_rgb_image(adv_save_path, img_a_plot)
                    except Exception as exc:
                        rel_adv_path = project_relative_path(adv_save_path, base_dir)
                        print(f"\n[ERRORE FATALE] Impossibile salvare l'attacco in: {rel_adv_path} -> {exc}")
                        continue

                    rel_source = row['cropped_image_path'] # Questo è già corretto e relativo
                    rel_adv = adv_save_path.relative_to(base_dir).as_posix()

                    eps_tracker_records.append({
                        "attack_type": "bim", 
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
        
    print("\n[OK] Processo BIM Targeted completato con successo e immagini salvate!")

if __name__ == "__main__":
    main()
