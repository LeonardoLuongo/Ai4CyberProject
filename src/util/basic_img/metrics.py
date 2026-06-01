import numpy as np

def calculate_linf(clean_img: np.ndarray, adv_img: np.ndarray) -> float:
    """
    Calcola la norma L_inf (la massima perturbazione assoluta) tra due immagini.
    Indipendentemente dal fatto che l'input sia [0, 255] o [0, 1],
    il risultato viene scalato e restituito nel range [0.0, 1.0].
    
    :param clean_img: Immagine originale (numpy array).
    :param adv_img: Immagine avversaria (numpy array).
    :return: Valore float rappresentante l'Epsilon L_inf effettivo.
    """
    # 1. Cast a float per evitare problemi di overflow/underflow con uint8 (es. 0 - 1 = 255)
    c_img_f = clean_img.astype(np.float32)
    a_img_f = adv_img.astype(np.float32)
    
    # 2. Normalizziamo in [0, 1] se i valori sono in [0, 255]
    if c_img_f.max() > 1.0 or a_img_f.max() > 1.0:
        c_img_f /= 255.0
        a_img_f /= 255.0

    # 3. La formula dell'L_inf: max(|Adv - Clean|)
    diff = np.abs(a_img_f - c_img_f)
    l_inf = np.max(diff)
    
    return float(l_inf)


def validate_linf_batch(clean_batch: np.ndarray, 
                        adv_batch: np.ndarray, 
                        threshold: float = 0.1, 
                        tolerance: float = 1e-5) -> list[dict]:
    """
    Validatore batch. Calcola l'L_inf per ogni coppia di immagini e restituisce 
    una lista di dizionari con il report di validazione per ogni sample.
    
    :param clean_batch: Batch di immagini originali (es. shape (B, C, H, W) o (B, H, W, C)).
    :param adv_batch: Batch di immagini avversarie (stessa shape).
    :param threshold: Il budget Epsilon massimo consentito (default 0.1).
    :param tolerance: Tolleranza per errori di arrotondamento floating-point.
    :return: Lista di dizionari [{ 'epsilon': float, 'is_valid': bool }, ...]
    """
    if clean_batch.shape != adv_batch.shape:
        raise ValueError(f"Le shape non coincidono: {clean_batch.shape} vs {adv_batch.shape}")

    batch_size = clean_batch.shape[0]
    validation_report = []
    
    # Cast e Normalizzazione batch
    c_batch_f = clean_batch.astype(np.float32)
    a_batch_f = adv_batch.astype(np.float32)
    
    if c_batch_f.max() > 1.0 or a_batch_f.max() > 1.0:
        c_batch_f /= 255.0
        a_batch_f /= 255.0

    for i in range(batch_size):
        # Calcolo vettorizzato per la singola immagine (indipendentemente dagli assi H,W,C)
        diff = np.abs(a_batch_f[i] - c_batch_f[i])
        actual_epsilon = float(np.max(diff))
        
        # Validazione con tolleranza (per evitare che 0.10000001 dia False)
        # 1 = Valido (sotto o uguale alla soglia), 0 = Invalido (sfora la soglia)
        is_valid = 1 if actual_epsilon <= (threshold + tolerance) else 0
        
        validation_report.append({
            "sample_index": i,
            "actual_epsilon": round(actual_epsilon, 6),
            "is_valid": is_valid
        })

    return validation_report