import numpy as np

def select_target_label(clean_predictions: np.ndarray, true_label: int, strategy: str = "random", num_classes: int = 8631) -> int:
    """
    Seleziona la classe bersaglio per un attacco Targeted.
    """
    if strategy == "random":
        target = true_label
        while target == true_label:
            target = np.random.randint(0, num_classes)
        return target
        
    elif strategy == "next-best":
        # np.argsort ordina le probabilità dal più basso al più alto
        sorted_indices = np.argsort(clean_predictions[0])
        
        # L'ultimo (-1) è la classe con probabilità più alta (top-1)
        if sorted_indices[-1] == true_label:
            return sorted_indices[-2] # Prende la seconda più alta
        else:
            return sorted_indices[-1]
            
    elif strategy == "least-likely":
        # La classe meno probabile in assoluto è il primo elemento dell'array ordinato
        sorted_indices = np.argsort(clean_predictions[0])
        
        # Nel rarissimo caso in cui la classe vera fosse quella con probabilità più bassa (rete totalmente rotta),
        # prendiamo la penultima peggiore per sicurezza.
        if sorted_indices[0] == true_label:
            return sorted_indices[1]
        else:
            return sorted_indices[0]
            
    else:
        raise ValueError(f"Strategia '{strategy}' non supportata. Scegli tra: random, next-best, least-likely.")


def get_one_hot_target(target_class: int, num_classes: int = 8631) -> np.ndarray:
    """
    Converte l'ID del target nel formato One-Hot richiesto da ART.
    Ritorna un array di shape (1, num_classes).
    """
    one_hot = np.zeros((1, num_classes), dtype=np.float32)
    one_hot[0, target_class] = 1.0
    return one_hot