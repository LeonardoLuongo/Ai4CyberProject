import os
import sys
import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from pathlib import Path
import pickle

# =========================================================================
# WORKAROUND CUDNN
# =========================================================================
os.environ["TORCH_CUDNN_V8_API_ENABLED"] = "0" 
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

PROJECT_ROOT = Path(__file__).resolve().parents[5]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import specifici
from util.google_logger import GoogleSheetLogger
from util.identity_mapper import IdentityMapper

try:
    from models.senet import senet50
except ImportError:
    raise ImportError("ERRORE: 'senet.py' non trovato nella cartella models.")

# --- COSTANTI DELL'AUTORE PER SENET50 ---
MEAN_BGR = np.array([91.4953, 103.8827, 131.0912], dtype=np.float32)

def load_caffe_weights(model, fname):
    """Caricamento dei pesi .pkl (Caffe to PyTorch) con fix per Python 2/3"""
    with open(fname, 'rb') as f:
        weights = pickle.load(f, encoding='latin1')
    own_state = model.state_dict()
    for name, param in weights.items():
        if name in own_state:
            own_state[name].copy_(torch.from_numpy(param))

def preprocess_for_senet(img_bgr_224):
    """Sottrazione della media Caffe e HWC -> CHW"""
    img = img_bgr_224.astype(np.float32)
    img -= MEAN_BGR
    img = img.transpose(2, 0, 1)
    return img

def evaluate_transferability(tracker_path, nn2_model, device, mapper, logger, base_dir, attack_category):
    """
    Funzione core per la valutazione Black-Box.
    Legge il CSV di NN1 e testa le immagini su NN2.
    """
    df = pd.read_csv(tracker_path)
    if df.empty: return
    
    eps = df['eps'].iloc[0]
    strategy = df['target_strategy'].iloc[0]
    attack_type = df['attack_type'].iloc[0].upper()
    
    # Determiniamo se è Targeted o Untargeted
    is_targeted = df['targeted'].iloc[0] if 'targeted' in df.columns else (attack_category == 'error_specific')
    
    print(f"\n[>>>] Valutazione Trasferibilità: {attack_type} | Eps: {eps:.3f} | Strat: {strategy}")
    
    batch_size = 64
    total_valid_clean = 0
    resisted = 0
    successes = 0
    untargeted = 0
    
    # Processiamo in Batch per massimizzare la GPU
    with torch.no_grad():
        for start_idx in tqdm(range(0, len(df), batch_size), desc="Inferenza SENet50", leave=False):
            batch_df = df.iloc[start_idx : start_idx + batch_size]
            
            x_clean_list, x_adv_list = [], []
            true_ids, target_ids = [], []
            
            for _, row in batch_df.iterrows():
                true_id = mapper.get_facenet_id_by_class_id(str(row['identity_id']))
                tgt_id = int(row['target_class']) if is_targeted else -1
                
                c_path = base_dir / str(row['source_image_path'])
                a_path = base_dir / str(row['adversarial_image_path'])
                
                # 1. Caricamento (Le immagini sono 160x160 generate da NN1)
                c_bgr = cv2.imread(str(c_path))
                a_bgr = cv2.imread(str(a_path))
                
                if c_bgr is None or a_bgr is None: continue
                
                # 2. L'OSTACOLO: Upscaling Bilineare a 224x224 (Diluisce l'attacco)
                c_bgr_224 = cv2.resize(c_bgr, (224, 224), interpolation=cv2.INTER_LINEAR)
                a_bgr_224 = cv2.resize(a_bgr, (224, 224), interpolation=cv2.INTER_LINEAR)
                
                # 3. Normalizzazione Caffe
                x_clean_list.append(preprocess_for_senet(c_bgr_224))
                x_adv_list.append(preprocess_for_senet(a_bgr_224))
                true_ids.append(true_id)
                target_ids.append(tgt_id)
                
            if not x_clean_list: continue
            
            t_clean = torch.tensor(np.array(x_clean_list)).to(device)
            t_adv = torch.tensor(np.array(x_adv_list)).to(device)
            
            # 4. Inferenza su NN2
            clean_preds = torch.argmax(nn2_model(t_clean), dim=1).cpu().numpy()
            adv_preds = torch.argmax(nn2_model(t_adv), dim=1).cpu().numpy()
            
            true_ids = np.array(true_ids)
            target_ids = np.array(target_ids)
            
            # --- LOGICA DI TRASFERIBILITÀ (Solo se NN2 riconosceva l'immagine pulita) ---
            valid_mask = (clean_preds == true_ids)
            valid_count = valid_mask.sum()
            total_valid_clean += valid_count
            
            if valid_count > 0:
                v_adv_preds = adv_preds[valid_mask]
                v_true_ids = true_ids[valid_mask]
                v_target_ids = target_ids[valid_mask]
                
                # Mutuamente esclusivi
                res_mask = (v_adv_preds == v_true_ids)
                if is_targeted:
                    tgt_mask = (v_adv_preds == v_target_ids) & (~res_mask)
                else:
                    tgt_mask = np.zeros_like(res_mask, dtype=bool) # Sempre falso per Error-Generic
                    
                untgt_mask = (~res_mask) & (~tgt_mask)
                
                resisted += res_mask.sum()
                successes += tgt_mask.sum()
                untargeted += untgt_mask.sum()

    if total_valid_clean > 0:
        robust_accuracy = resisted / total_valid_clean
        targeted_asr = successes / total_valid_clean
        untargeted_asr = untargeted / total_valid_clean
        
        print(f" -> Analizzate {total_valid_clean} immagini validabili da NN2.")
        print(f" -> Robust Accuracy: {robust_accuracy*100:.2f}% | Targeted ASR: {targeted_asr*100:.2f}% | Untargeted ASR: {untargeted_asr*100:.2f}%")
        
        try:
            logger.log_attack_metrics(
                tester="Francesco",
                attack_type=f"{attack_type} Transfer (NN1->NN2)",
                strategy=strategy if is_targeted else "Untargeted",
                epsilon=eps,
                defense_type="None",
                robust_accuracy=robust_accuracy,
                targeted_asr=targeted_asr,
                untargeted_asr=untargeted_asr,
                notes=f"Black-Box. Caffe Norm. Upscaling 160->224"
            )
        except Exception as e:
            print(f"[WARNING] Errore Google Logger: {e}")
    else:
        print(" -> Nessuna immagine pulita è stata riconosciuta da NN2. Impossibile valutare la trasferibilità.")


def main():
    print("======================================================")
    print(" PUNTO 5: TEST DI TRASFERIBILITÀ (NN1 -> NN2)         ")
    print("======================================================\n")

    base_dir = Path.cwd()
    meta_csv_path = base_dir / "dataset" / "clean" / "splits" / "identity_meta.csv"
    weights_path = base_dir / "src" / "models" / "senet50_ft_weight.pkl" 
    
    if not weights_path.exists(): raise FileNotFoundError(f"Errore: pesi NN2 mancanti in {weights_path}")

    print("-> Inizializzazione Logger e Mapper...")
    logger = GoogleSheetLogger()
    mapper = IdentityMapper(meta_csv_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"-> Caricamento NN2 (SENet50) su {device}...")
    
    nn2 = senet50(num_classes=8631, include_top=True)
    load_caffe_weights(nn2, weights_path)
    nn2.eval().to(device)
    
    # ==========================================
    # DEFINIAMO LE DIRECTORY DEGLI ATTACCHI DA TESTARE
    # ==========================================
    attack_folders = [
        # 1. PGD Untargeted (Error-Generic)
        {"path": base_dir / "dataset" / "attacks" / "NN1" / "error_generic" / "pgd", "category": "error_generic"},
        
        # 2. PGD Targeted (Error-Specific)
        {"path": base_dir / "dataset" / "attacks" / "NN1" / "error_specific" / "pgd", "category": "error_specific"},
        
        # 3. FGSM Targeted (Error-Specific)
        {"path": base_dir / "dataset" / "attacks" / "NN1" / "error_specific" / "fgsm", "category": "error_specific"}
    ]
    
    for attack in attack_folders:
        folder = attack["path"]
        category = attack["category"]
        
        if not folder.exists():
            print(f"\n[SKIP] Cartella non trovata: {folder}")
            continue
            
        print(f"\n======================================================")
        print(f" ESPLORAZIONE CARTELLA: {folder.name.upper()} ({category})")
        print(f"======================================================")
        
        # Cerca iterativamente in tutte le sottocartelle (strategie e epsilon) i file tracker_eps_*.csv
        tracker_files = list(folder.rglob("tracker_eps_*.csv"))
        print(f"-> Trovati {len(tracker_files)} file CSV di generazione.")
        
        for tracker in tracker_files:
            evaluate_transferability(tracker, nn2, device, mapper, logger, base_dir, category)

    print("\n[OK] Test di Trasferibilità concluso! Controlla i risultati su Google Sheets.")

if __name__ == "__main__":
    main()