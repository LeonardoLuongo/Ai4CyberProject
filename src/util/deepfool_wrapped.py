import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import torchattacks

# ==========================================
# CLASSIFICATORI PROXY (Per aggirare la VRAM)
# ==========================================
class TopKFacenetWrapper(nn.Module):
    """Congela l'universo delle classi alle Top-K per alleggerire i gradienti."""
    def __init__(self, model, k=10):
        super().__init__()
        self.model = model
        self.k = k
        self.active_indices = None
        
    def freeze_target_classes(self, x):
        with torch.no_grad():
            x_scaled = (x * 2.0) - 1.0
            logits = self.model(x_scaled)
            _, self.active_indices = torch.topk(logits, self.k, dim=1)

    def forward(self, x):
        x_scaled = (x * 2.0) - 1.0
        logits = self.model(x_scaled)
        gathered = torch.gather(logits, 1, self.active_indices)
        return gathered

# ==========================================
# IL WRAPPER PRINCIPALE STILE "ART"
# ==========================================
class ARTCompatibleDeepFool:
    def __init__(self, classifier, max_iter=50, epsilon=0.02, nb_grads=10, batch_size=1):
        """
        Adapter che simula il comportamento di art.attacks.evasion.DeepFool.
        
        :param classifier: Il PyTorchClassifier di ART (da cui estraiamo il modello PyTorch nudo)
        :param max_iter: Il numero massimo di step per torchattacks.
        :param epsilon: L'overshoot per torchattacks.
        :param nb_grads: La 'K' per limitare le classi (Top-K) e non fondere la VRAM.
        :param batch_size: Ignorato/Forzato a 1 internamente per limiti di torchattacks.
        """
        # Estraiamo il vero modello PyTorch (InceptionResnetV1) dall'oggetto ART
        self.base_model = classifier.model
        
        # Mappiamo i parametri ART su quelli equivalenti del nostro sistema
        self.steps = max_iter
        self.overshoot = epsilon
        self.k_classes = nb_grads
        
        # Inizializziamo il nostro wrapper TopK
        self.proxy_model = TopKFacenetWrapper(self.base_model, k=self.k_classes).eval()
        
        # Inizializziamo l'attacco di torchattacks
        self.attack = torchattacks.DeepFool(self.proxy_model, steps=self.steps, overshoot=self.overshoot)

    def generate(self, x: np.ndarray, y=None) -> np.ndarray:
        """
        Riceve e restituisce Numpy Arrays [0, 1] per essere identico ad ART.
        Gestisce internamente il loop a batch_size=1.
        """
        device = next(self.base_model.parameters()).device
        
        x_adv_list = []
        
        # Convertiamo l'input NumPy [N, 3, H, W] in Tensore
        x_tensor = torch.tensor(x, dtype=torch.float32, device=device)
        total_samples = x_tensor.size(0)
        
        # Disabilitiamo temporaneamente le barre tqdm interne se x ha solo 1 campione
        # per non intasare l'output del tuo samples_gen.py
        disable_tqdm = (total_samples == 1)
        
        for i in tqdm(range(total_samples), desc="DeepFool Engine", disable=disable_tqdm, leave=False):
            # Torchattacks fallisce sui batch, isoliamo 1 campione per volta
            x_single = x_tensor[i:i+1] 
            
            # 1. Congeliamo l'universo delle classi per QUESTA specifica immagine
            self.proxy_model.freeze_target_classes(x_single)
            
            # 2. Nel mini-universo Top-K ordinato, la classe originale (che ci fa da "safe zone")
            # si troverà matematicamente all'indice 0. Passiamo 0 come target fittizio.
            y_local = torch.zeros(1, dtype=torch.long, device=device)
            
            # 3. Lanciamo l'attacco
            # Torchattacks ci ridà l'immagine in [0, 1]
            x_adv_single = self.attack(x_single, y_local)
            
            # Spostiamo su CPU e convertiamo in Numpy per accodarlo
            x_adv_list.append(x_adv_single.cpu().numpy())
            
        # Riaffianchiamo l'array NumPy finale [N, 3, H, W]
        return np.concatenate(x_adv_list, axis=0)