import os
os.environ["TORCH_CUDNN_V8_API_ENABLED"] = "0" 
import cv2
import numpy as np
import pandas as pd
import torch

torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

from tqdm import tqdm
from pathlib import Path
from PIL import Image
import pickle 

from facenet_pytorch import MTCNN

from util.google_logger import GoogleSheetLogger
from util.identity_mapper import IdentityMapper

try:
    from models.senet import senet50
except ImportError:
    raise ImportError("ERRORE: 'senet.py' non trovato nella cartella models.")

# --- COSTANTI DELL'AUTORE ---
MEAN_BGR = np.array([91.4953, 103.8827, 131.0912], dtype=np.float32)

def load_caffe_weights(model, fname):
    """Funzione ufficiale dell'autore per il caricamento dei pesi .pkl"""
    with open(fname, 'rb') as f:
        weights = pickle.load(f, encoding='latin1')

    own_state = model.state_dict()
    for name, param in weights.items():
        if name in own_state:
            own_state[name].copy_(torch.from_numpy(param))
        else:
            print(f"[WARNING] Chiave inaspettata: {name}")

def preprocess_for_senet(img_bgr):
    """Applica l'esatta trasformazione usata dall'autore in VGG_Faces2 dataset"""
    img = img_bgr.astype(np.float32)
    img -= MEAN_BGR
    img = img.transpose(2, 0, 1) # HWC -> CHW
    return torch.from_numpy(img).float()

def main():
    print("======================================================")
    print(" NN2 (SENet50) - CLEAN EVALUATION SUL TEST SET        ")
    print("======================================================\n")

    # ==========================================
    # 1. SETUP PATHS
    # ==========================================
    base_dir = Path.cwd()
    
    csv_path = base_dir / "dataset" / "clean" / "splits" / "manifest.csv"
    meta_csv_path = base_dir / "dataset" / "clean" / "splits" / "identity_meta.csv"
    weights_path = base_dir / "src" / "models" / "senet50_ft_weight.pkl" 
    
    cropped_clean_dir = base_dir / "dataset" / "clean_cropped" / "NN2"
    
    if not csv_path.exists(): raise FileNotFoundError(f"Errore: manifest.csv mancante")
    if not weights_path.exists(): raise FileNotFoundError(f"Errore: pesi NN2 mancanti in {weights_path}")

    # ==========================================
    # 2. INIZIALIZZAZIONE MAPPER E MODELLI
    # ==========================================
    print("-> Inizializzazione IdentityMapper...")
    mapper = IdentityMapper(meta_csv_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"-> Inizializzazione NN2 (SENet50) su {device}...")
    
    nn2 = senet50(num_classes=8631, include_top=True)
    load_caffe_weights(nn2, weights_path)
    nn2.eval().to(device)
    
    # ==========================================
    # 3. RETE DI CROPPING A 224x224
    # ==========================================
    mtcnn = MTCNN(image_size=224, margin=0, keep_all=True, post_process=True, device=device)

    df = pd.read_csv(csv_path)
    
    total_valid_images = 0
    correct_no_crop = 0
    correct_with_crop = 0
    faces_not_found = 0
    cached_count = 0

    # ==========================================
    # 4. LOOP DI TEST A/B E CACHING
    # ==========================================
    with torch.no_grad():
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Test A/B su NN2"):
            identity_id = str(row['identity_id'])
            
            true_facenet_id = mapper.get_facenet_id_by_class_id(identity_id)
            if true_facenet_id == -1: continue
                
            img_path = base_dir / str(row['image_path']) 
            identity_dir_name = img_path.parent.name
            img_filename = img_path.name
            
            try:
                img_pil = Image.open(str(img_path)).convert('RGB')
            except Exception:
                continue
            
            total_valid_images += 1

            # ------------------------------------------
            # METODO A: NESSUN CROP (Raw Resize a 224)
            # ------------------------------------------
            img_224_pil = img_pil.resize((224, 224), Image.BILINEAR)
            img_bgr_nocrop = cv2.cvtColor(np.array(img_224_pil), cv2.COLOR_RGB2BGR)
            
            t_raw = preprocess_for_senet(img_bgr_nocrop).unsqueeze(0).to(device)
            logits_raw = nn2(t_raw)
            pred_raw = int(torch.argmax(logits_raw, dim=1).cpu().numpy()[0])
            
            if pred_raw == true_facenet_id:
                correct_no_crop += 1

            # ------------------------------------------
            # METODO B: CON CROP MTCNN E CACHING
            # ------------------------------------------
            out_crop_dir = cropped_clean_dir / identity_dir_name
            out_crop_dir.mkdir(parents=True, exist_ok=True)
            crop_save_path = out_crop_dir / img_filename
            
            if crop_save_path.exists():
                # CACHE HIT
                img_bgr = cv2.imread(str(crop_save_path))
                t_cropped = preprocess_for_senet(img_bgr).unsqueeze(0).to(device)
                
                logits_cropped = nn2(t_cropped)
                pred_cropped = int(torch.argmax(logits_cropped, dim=1).cpu().numpy()[0])
                
                if pred_cropped == true_facenet_id:
                    correct_with_crop += 1
                cached_count += 1
                
            else:
                # CACHE MISS
                faces = mtcnn(img_pil)
                if faces is not None:
                    faces = faces.to(device)
                    # MTCNN restituisce facce in range [-1, 1] RGB
                    # Dobbiamo convertirle in [0, 255] BGR per la SENet50
                    
                    # Troviamo la faccia corretta passando per NN2 (serve la conversione temporanea)
                    best_match_idx = -1
                    for i, face_tensor in enumerate(faces):
                        np_face_01 = (face_tensor.cpu().numpy() + 1.0) / 2.0
                        face_bgr_255 = cv2.cvtColor((np.transpose(np_face_01, (1, 2, 0)) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
                        t_face_caffe = preprocess_for_senet(face_bgr_255).unsqueeze(0).to(device)
                        
                        logits = nn2(t_face_caffe)
                        if int(torch.argmax(logits, dim=1)) == true_facenet_id:
                            best_match_idx = i
                            break
                    
                    if best_match_idx != -1:
                        correct_with_crop += 1
                        
                        # Salvataggio fisico per la cache
                        np_face_01 = (faces[best_match_idx].cpu().numpy() + 1.0) / 2.0
                        img_c_save = np.clip(np.transpose(np_face_01, (1, 2, 0)) * 255.0, 0, 255).astype(np.uint8)
                        img_c_bgr = cv2.cvtColor(img_c_save, cv2.COLOR_RGB2BGR)
                        cv2.imwrite(str(crop_save_path), img_c_bgr)
                else:
                    faces_not_found += 1

    # ==========================================
    # 5. REPORT FINALE
    # ==========================================
    print("\n" + "="*50)
    print(" RISULTATI ESPERIMENTO A/B SU NN2 ")
    print("="*50)
    print(f"Immagini Totali Valutate  : {total_valid_images}")
    print(f"Immagini caricate da Cache: {cached_count}")
    print(f"Volti NON trovati da MTCNN: {faces_not_found}\n")
    
    acc_no_crop = (correct_no_crop / total_valid_images) if total_valid_images > 0 else 0
    acc_crop = (correct_with_crop / total_valid_images) if total_valid_images > 0 else 0
    
    print(f"Accuracy NN2 NESSUN CROP   : {acc_no_crop*100:.2f}% ({correct_no_crop}/{total_valid_images})")
    print(f"Accuracy NN2 CON MTCNN CROP: {acc_crop*100:.2f}% ({correct_with_crop}/{total_valid_images})")

    # =======================================
    # 6. LOGGING SU GOOGLE SHEETS
    # =======================================
    try:
        logger = GoogleSheetLogger()
        
        logger.log_attack_metrics(
            tester="Francesco",
            attack_type="NN2 Baseline",
            strategy="Clean Evaluation",
            epsilon=0.0,
            defense_type="Senza MTCNN (Resize)",
            robust_accuracy=acc_no_crop,
            targeted_asr=0.0, 
            untargeted_asr=1.0 - acc_no_crop,
            notes="SENet50, Classificazione Top-1 a 224x224 (Caffe Norm)"
        )
        
        logger.log_attack_metrics(
            tester="Francesco",
            attack_type="NN2 Baseline",
            strategy="Clean Evaluation",
            epsilon=0.0,
            defense_type="Con MTCNN",
            robust_accuracy=acc_crop,
            targeted_asr=0.0, 
            untargeted_asr=1.0 - acc_crop,
            notes=f"SENet50, MTCNN a 224x224 (Caffe Norm, Falliti {faces_not_found})"
        )
    except Exception as e:
        print(f"[WARNING] Impossibile inviare a Google Sheets: {e}")

if __name__ == "__main__":
    main()