import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path
from PIL import Image
import pickle

# =========================================================================
# RISOLUZIONE ROBUSTA DEI PATH
# =========================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[5]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import FastGradientMethod

from src.util.identity_mapper import IdentityMapper
from src.util.basic_img.metrics import calculate_linf
from src.util.attack_error_specific_utils import get_one_hot_target

# =========================================================================
# IMPORTIAMO LA SENET50 DALLA CARTELLA LOCALE
# =========================================================================
from src.models.senet import senet50

def load_caffe_weights(model, fname):
    """Funzione ufficiale dell'autore per il caricamento dei pesi .pkl (Caffe to PyTorch)"""
    with open(fname, 'rb') as f:
        weights = pickle.load(f, encoding='latin1')

    own_state = model.state_dict()
    for name, param in weights.items():
        if name in own_state:
            own_state[name].copy_(torch.from_numpy(param))
        else:
            print(f"[WARNING] Chiave inaspettata: {name}")

def main():
    print("======================================================")
    print(" GENERATORE CAMPIONI: FGSM ERROR-GENERIC (NN2 - SENET50)")
    print("======================================================\n")

    base_dir = PROJECT_ROOT
    print(f"-> Project Root impostata a: {base_dir}")

    # --- 1. CONFIGURAZIONE PATH E PARAMETRI (Aggiornati per NN2) ---
    csv_path = base_dir / "dataset" / "clean" / "splits" / "manifest.csv"
    meta_csv_path = base_dir / "dataset" / "clean" / "splits" / "identity_meta.csv"
    
    # Path output per gli attacchi NN2
    output_base_dir = base_dir / "dataset" / "attacks" / "NN2" / "error_generic" / "fgsm"
    
    # Puntiamo alla cartella di NN1 per prendere le immagini già validate!
    cropped_nn1_dir = base_dir / "dataset" / "clean_cropped" / "NN1"
    
    epsilons = [0.025, 0.05, 0.075, 0.10, 0.15, 0.20]
    BATCH_SIZE = 64  
    IMAGE_SIZE = 224 # Dimensione corretta per SENet50
    
    if not csv_path.exists() or not meta_csv_path.exists():
        raise FileNotFoundError(f"Errore: manifest.csv o identity_meta.csv mancanti in {base_dir}")

    # --- 2. INIZIALIZZAZIONE MODELLI E MAPPER ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"-> Inizializzazione Reti su {device}...")
    
    mapper = IdentityMapper(meta_csv_path)
    
    # =========================================================================
    # INIZIALIZZAZIONE SENET50 E CARICAMENTO PESI (CON PICKLE LATIN-1)
    # =========================================================================
    print("-> Caricamento SENet50 (NN2)...")
    resnet = senet50(num_classes=8631, include_top=True)
    
    weights_filename = "senet50_ft_weight.pkl" 
    weights_path = base_dir / "src" / "models" / weights_filename
    
    if weights_path.exists():
        # Utilizziamo la funzione custom per aggirare il problema dell'encoding
        load_caffe_weights(resnet, weights_path)
        print(f"-> Pesi caricati correttamente da: {weights_filename}")
    else:
        print(f"\n[ATTENZIONE ERRORE] File dei pesi non trovato in: {weights_path}")
        print("Assicurati di correggere il nome del file alla riga 75!")
        return 
        
    resnet = resnet.eval().to(device)

    # Il classificatore ART ora si aspetta input (3, 224, 224)
    classifier = PyTorchClassifier(
        model=resnet, clip_values=(0.0, 1.0), loss=nn.CrossEntropyLoss(), optimizer=None,
        input_shape=(3, IMAGE_SIZE, IMAGE_SIZE), nb_classes=8631, preprocessing=(0.5, 0.5), 
        device_type='gpu' if torch.cuda.is_available() else 'cpu'
    )

    df_clean = pd.read_csv(csv_path)
    print(f"-> Trovate {len(df_clean)} immagini totali nel manifest.")

    # =========================================================================
    # FASE 1: CARICAMENTO "TEST SET ORO" (Validato da NN1)
    # =========================================================================
    print(f"\n[FASE 1] Caricamento immagini validate da NN1 (Resize automatico a {IMAGE_SIZE}x{IMAGE_SIZE})...")
    valid_records = []
    
    with torch.no_grad():
        for index, row in tqdm(df_clean.iterrows(), total=len(df_clean), desc="Lettura Clean_Cropped"):
            class_id = str(row['identity_id'])
            facenet_id = mapper.get_facenet_id_by_class_id(class_id)
            if facenet_id == -1: continue
                
            source_img_path = base_dir / row['image_path']
            identity_dir_name = source_img_path.parent.name
            img_filename = source_img_path.name
                
            # Cerchiamo l'immagine direttamente nella cartella di NN1
            crop_load_path = cropped_nn1_dir / identity_dir_name / img_filename
            
            if not crop_load_path.exists():
                # Se l'immagine non c'è, NN1 l'aveva scartata. La ignoriamo anche qui.
                continue 
            
            try:
                # Carichiamo l'immagine (che è 160x160) e facciamo l'upscale a 224x224 per SENet50
                img_pil = Image.open(crop_load_path).convert('RGB')
                if img_pil.size != (IMAGE_SIZE, IMAGE_SIZE):
                    img_pil = img_pil.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
                
                np_img_01 = np.transpose(np.asarray(img_pil), (2, 0, 1)).astype(np.float32) / 255.0
                x_clean = np.expand_dims(np_img_01, axis=0)
            except Exception as e:
                print(f"\n[ERRORE] Impossibile leggere l'immagine in: {crop_load_path} - {e}")
                continue 
                
            row_dict = row.to_dict()
            row_dict['true_facenet_id'] = facenet_id
            row_dict['x_clean'] = x_clean 
            # Il path di origine rimane quello di NN1
            row_dict['cropped_image_path'] = crop_load_path.relative_to(base_dir).as_posix() 
            
            valid_records.append(row_dict)

    total_valid = len(valid_records)
    print(f"-> Immagini d'oro pronte per l'attacco FGSM Untargeted (NN2): {total_valid}")

    # =========================================================================
    # FASE 2: GENERAZIONE FGSM UNTARGETED (BATCH DIRETTO) su NN2
    # =========================================================================
    print(f"\n======================================================")
    print(f" AVVIO GENERAZIONE ERROR-GENERIC (NN2)")
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
                # ART userà questo One-Hot Encoding per allontanarsi dalla classe corretta
                batch_y_list.append(get_one_hot_target(row['true_facenet_id'], num_classes=mapper.get_num_training_classes())[0])
                
            batch_x_array = np.stack(batch_x_list)
            batch_y_array = np.stack(batch_y_list)
            
            # Generazione attacco
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
                
                adv_filename = f"{orig_filename}.jpg"
                adv_save_path = out_img_dir / adv_filename
                
                img_a_save = (img_a_plot * 255.0).astype(np.uint8)
                try:
                    Image.fromarray(img_a_save).save(adv_save_path)
                except Exception as e:
                    print(f"\n[ERRORE FATALE] Impossibile salvare l'attacco in: {adv_save_path} - {e}")

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
        
    print("\n[OK] Processo Error-Generic su NN2 completato con successo!")

if __name__ == "__main__":
    main()