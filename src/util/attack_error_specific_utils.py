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
        sorted_indices = np.argsort(clean_predictions[0])
        if sorted_indices[-1] == true_label:
            return sorted_indices[-2]
        else:
            return sorted_indices[-1]
            
    else:
        raise ValueError("Strategia non supportata.")

def get_one_hot_target(target_class: int, num_classes: int = 8631) -> np.ndarray:
    """
    Converte l'ID del target nel formato One-Hot richiesto da ART.
    Ritorna un array di shape (1, num_classes).
    """
    one_hot = np.zeros((1, num_classes), dtype=np.float32)
    one_hot[0, target_class] = 1.0
    return one_hot