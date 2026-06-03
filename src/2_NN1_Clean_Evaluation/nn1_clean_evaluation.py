import os
import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from pathlib import Path
from PIL import Image
import numpy as np

from facenet_pytorch import InceptionResnetV1, MTCNN

# Grazie al PYTHONPATH=src, questi import sono diretti e puliti
from util.google_logger import GoogleSheetLogger
from util.identity_mapper import IdentityMapper

def main():
    print("======================================================")
    print(" ESPERIMENTO A/B: IMPATTO DEL CROPPING (MTCNN) SULL'ACCURACY ")
    print("======================================================\n")

    # ==========================================
    # 1. SETUP PATHS
    # ==========================================
    # Assumiamo che lo script venga lanciato dalla root (Ai4CyberProject)
    base_dir = Path.cwd()
    
    # Path aggiornati secondo l'albero del tuo progetto
    csv_path = base_dir / "dataset" / "clean" / "splits" / "manifest.csv"
    meta_csv_path = base_dir / "dataset" / "clean" / "splits" / "identity_meta.csv"
    
    if not csv_path.exists():
        raise FileNotFoundError(f"Errore: manifest.csv mancante in {csv_path}")
    if not meta_csv_path.exists():
        raise FileNotFoundError(f"Errore: identity_meta.csv mancante in {meta_csv_path}")

    # ==========================================
    # 2. INIZIALIZZAZIONE MAPPER E MODELLI
    # ==========================================
    print("-> Inizializzazione IdentityMapper...")
    mapper = IdentityMapper(meta_csv_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"-> Inizializzazione Reti su {device}...")
    
    # Rete di Classificazione
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    resnet.classify = True
    
    # Rete di Cropping
    # MODIFICA: keep_all=True per trovare tutte le facce presenti nell'immagine
    mtcnn = MTCNN(image_size=160, margin=0, keep_all=True, post_process=True, device=device)

    df = pd.read_csv(csv_path)
    
    # ==========================================
    # 3. VARIABILI METRICHE
    # ==========================================
    total_valid_images = 0
    correct_no_crop = 0
    correct_with_crop = 0
    faces_not_found = 0

    # ==========================================
    # 4. LOOP DI TEST
    # ==========================================
    with torch.no_grad():
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Test A/B Classification"):
            identity_id = str(row['identity_id'])
            
            true_facenet_id = mapper.get_facenet_id_by_class_id(identity_id)
            if true_facenet_id == -1:
                continue
                
            img_path = base_dir / str(row['image_path']) 
            
            # --- CARICAMENTO IDENTICO AL COLLEGA (PIL) ---
            try:
                img_pil = Image.open(str(img_path)).convert('RGB')
            except Exception:
                continue
            
            total_valid_images += 1

            # ------------------------------------------
            # METODO A: NESSUN CROP (Senza OpenCV)
            # ------------------------------------------
            # Facciamo il resize brutale usando PIL invece di cv2
            img_160_pil = img_pil.resize((160, 160), Image.BILINEAR)
            # Convertiamo l'immagine PIL in Numpy array solo per passarla alla rete
            x_raw = np.transpose(np.array(img_160_pil), (2, 0, 1)).astype(np.float32) / 255.0
            t_raw = torch.tensor(x_raw).unsqueeze(0).to(device)
            
            logits_raw = resnet(t_raw * 2 - 1)
            pred_raw = int(torch.argmax(logits_raw, dim=1).cpu().numpy()[0])
            
            if pred_raw == true_facenet_id:
                correct_no_crop += 1

            # ------------------------------------------
            # METODO B: CON CROP MTCNN (Identico al collega)
            # ------------------------------------------
            # Passiamo DIRETTAMENTE l'oggetto PIL a MTCNN
            faces = mtcnn(img_pil)
            
            if faces is not None:
                faces = faces.to(device)
                logits_cropped = resnet(faces) 
                preds = torch.argmax(logits_cropped, dim=1).cpu().numpy()
                
                if true_facenet_id in preds:
                    correct_with_crop += 1
            else:
                faces_not_found += 1

    # ==========================================
    # 5. REPORT FINALE
    # ==========================================
    print("\n" + "="*50)
    print(" RISULTATI ESPERIMENTO A/B ")
    print("="*50)
    print(f"Immagini Totali Valutate  : {total_valid_images}")
    print(f"Volti NON trovati da MTCNN: {faces_not_found}\n")
    
    acc_no_crop = (correct_no_crop / total_valid_images) if total_valid_images > 0 else 0
    acc_crop = (correct_with_crop / total_valid_images) if total_valid_images > 0 else 0
    
    print(f"Accuracy NESSUN CROP   : {acc_no_crop*100:.2f}% ({correct_no_crop}/{total_valid_images})")
    print(f"Accuracy CON MTCNN CROP: {acc_crop*100:.2f}% ({correct_with_crop}/{total_valid_images})")
    
    if acc_crop > acc_no_crop:
        print(f"\n-> CONCLUSIONE: MTCNN migliora la classificazione del {(acc_crop - acc_no_crop)*100:.2f}%!")
    else:
        print(f"\n-> CONCLUSIONE: MTCNN non migliora l'accuratezza in questo subset.")

    # =======================================
    # 6. LOGGING SU GOOGLE SHEETS
    # =======================================
    try:
        logger = GoogleSheetLogger()
        
        # Essendo un test di classificazione pura, mettiamo EER, FAR, FRR a 0.
        logger.log_biometric_metrics(
            tester="Leonardo", 
            phase="A/B Crop Test",
            attack_type="None",
            epsilon=0.0,
            defense_type="Senza MTCNN (Resize)",
            accuracy=acc_no_crop,
            eer=0.0, far=0.0, frr=0.0, threshold=0.0,
            notes="Classificazione Top-1"
        )
        
        logger.log_biometric_metrics(
            tester="Leonardo",
            phase="A/B Crop Test",
            attack_type="None",
            epsilon=0.0,
            defense_type="Con MTCNN (All Faces)",
            accuracy=acc_crop,
            eer=0.0, far=0.0, frr=0.0, threshold=0.0,
            notes=f"Classificazione Top-1, Keep_all=True (Falliti {faces_not_found})"
        )
    except Exception as e:
        print(f"[WARNING] Impossibile inviare a Google Sheets: {e}")

if __name__ == "__main__":
    main()