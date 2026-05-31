import os
import numpy as np
import cv2

def load_rgb_image(path: str, debug=False) -> np.array:
    # 1. OpenCV legge sempre in BGR di default
    img_bgr = cv2.imread(path)
    
    if img_bgr is None:
        print(f"Errore: Impossibile caricare l'immagine {path}")
        return None
        
    # 2. Convertiamo subito in RGB. 
    # Questo è l'array corretto che userai per il tuo modello/dataset!
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    print(f"Successo! Immagine RGB caricata con shape: {img_rgb.shape}")

    if debug:
        show_rgb_image(image_rgb=img_rgb)

    # Ritorna l'immagine in RGB così puoi usarla nel resto del codice
    return img_rgb

def show_rgb_image(image_rgb: np.array):
    # 3. cv2.imshow VUOLE il BGR per mostrare i colori corretti.
    # Quindi gli passiamo l'immagine riconvertita al volo solo per lo schermo.
    cv2.imshow("Debug Window", cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
    
    print("Premi un tasto qualsiasi sulla finestra dell'immagine per chiuderla...")
    cv2.waitKey(0) 
    cv2.destroyAllWindows()

def save_rgb_img(image_rgb: np.ndarray, path: str, debug=False):
    # Crea la cartella di destinazione se non esiste
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Converti e salva
    img_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    success = cv2.imwrite(path, img_bgr)
    
    if debug:
        if success:
            print(f"Immagine salvata in: {path}")
        else:
            print(f"Errore durante il salvataggio in: {path}")
