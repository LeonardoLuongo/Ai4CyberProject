import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from facenet_pytorch import InceptionResnetV1
from pathlib import Path
from util.google_logger import GoogleSheetLogger

# ==========================================
# FUNZIONI E CLASSI
# ==========================================
def get_project_root(current_path: Path) -> Path:
    """
    Risale le cartelle a partire dal path corrente finché non trova 
    la cartella che contiene il file '.env'.
    """
    for parent in [current_path, *current_path.parents]:
        if (parent / ".env").exists():
            return parent
    raise FileNotFoundError("Impossibile trovare il file .env per stabilire la root del progetto.")

def extract_embeddings(dataloader, model, device):
    """
    Passa le immagini nel modello e restituisce gli embeddings normalizzati (L2).
    """
    embeddings_list = []
    labels_list = []
    
    print("Estrazione degli embeddings in corso...")
    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs = imgs.to(device)
            # Estraiamo gli embeddings
            embs = model(imgs)
            # Normalizziamo gli embeddings per stabilizzare la Cosine Similarity
            embs = F.normalize(embs, p=2, dim=1)
            
            embeddings_list.append(embs.cpu())
            labels_list.append(labels.cpu())
            
    return torch.cat(embeddings_list), torch.cat(labels_list)

def calculate_biometric_metrics(embeddings, labels):
    """
    Calcola le metriche biometriche (FAR, FRR, EER, Accuracy) e la soglia ottima.
    """
    N = embeddings.size(0)
    
    # 1. Matrice delle distanze (Cosine Similarity NxN)
    similarity_matrix = F.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=2)
    
    # 2. Matrice di verità (True se stessa identità, False altrimenti)
    is_same_identity = labels.unsqueeze(1) == labels.unsqueeze(0)
    
    # 3. Estraiamo solo il triangolo superiore per evitare coppie duplicate e auto-confronti
    idx_i, idx_j = torch.triu_indices(N, N, offset=1)
    
    sim_scores = similarity_matrix[idx_i, idx_j]
    truth_labels = is_same_identity[idx_i, idx_j]
    
    # 4. Separiamo i punteggi genuini dagli impostori
    genuine_scores = sim_scores[truth_labels]
    impostor_scores = sim_scores[~truth_labels]
    
    print(f"Coppie genuine calcolate: {genuine_scores.size(0)}")
    print(f"Coppie impostori calcolate: {impostor_scores.size(0)}")
    
    # 5. Iteriamo su possibili soglie per trovare l'Equal Error Rate
    thresholds = torch.linspace(-1.0, 1.0, steps=1000)
    far_list, frr_list = [], []
    
    for th in thresholds:
        false_accepts = (impostor_scores >= th).sum().item()
        far = false_accepts / impostor_scores.size(0)
        
        false_rejects = (genuine_scores < th).sum().item()
        frr = false_rejects / genuine_scores.size(0)
        
        far_list.append(far)
        frr_list.append(frr)
        
    far_tensor = torch.tensor(far_list)
    frr_tensor = torch.tensor(frr_list)
    
    # L'EER è il punto dove FAR e FRR sono più vicini
    diffs = torch.abs(far_tensor - frr_tensor)
    min_idx = torch.argmin(diffs)
    
    eer = (far_tensor[min_idx] + frr_tensor[min_idx]) / 2.0
    optimal_threshold = thresholds[min_idx].item()
    
    # 6. Calcolo dell'Accuratezza con la soglia ottima
    true_accepts = (genuine_scores >= optimal_threshold).sum().item()
    true_rejects = (impostor_scores < optimal_threshold).sum().item()
    total_pairs = genuine_scores.size(0) + impostor_scores.size(0)
    accuracy = (true_accepts + true_rejects) / total_pairs
    
    return {
        "EER": eer.item(),
        "Optimal_Threshold": optimal_threshold,
        "Accuracy": accuracy,
        "FAR_at_opt": far_tensor[min_idx].item(),
        "FRR_at_opt": frr_tensor[min_idx].item()
    }


# ==========================================
# ESECUZIONE PRINCIPALE 
# ==========================================
if __name__ == '__main__':
    # Disattivazione CUDNN bypass per Windows (evita i crash dei driver NVIDIA)
    torch.backends.cudnn.enabled = False
    
    # 1. Device Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Esecuzione su device: {device}")

    # 2. Path Setup dinamico
    current_dir = Path(__file__).resolve().parent
    root_dir = get_project_root(current_dir)
    dataset_path = root_dir / "dataset" / "clean" / "test"

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset non trovato al percorso: {dataset_path}")
    else:
        print(f"Root identificata: {root_dir}")
        print(f"Dataset trovato con successo in: {dataset_path}")

    # 3. Model Setup 
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

    # 4. DataLoader Setup
    # Manteniamo i valori nel range [0, 1] per semplificare il clamping negli attacchi
    transform_pipeline = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor()
    ])

    dataset = datasets.ImageFolder(root=dataset_path, transform=transform_pipeline)
    
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

    # 5. Extract & Evaluate
    embeddings, labels = extract_embeddings(dataloader, resnet, device)
    print(f"Embeddings estratti: {embeddings.shape} per {len(labels)} immagini.")

    metrics = calculate_biometric_metrics(embeddings, labels)

    print("\n--- RISULTATI DELLA VALUTAZIONE (Senza Normalizzazione) ---")
    print(f"Optimal Threshold (Cosine Similarity): {metrics['Optimal_Threshold']:.4f}")
    print(f"Equal Error Rate (EER): {metrics['EER']:.4%}")
    print(f"Verification Accuracy:  {metrics['Accuracy']:.4%}")
    print(f"FAR alla soglia ottima: {metrics['FAR_at_opt']:.4%}")
    print(f"FRR alla soglia ottima: {metrics['FRR_at_opt']:.4%}")

    # ==================================
    # SALVATAGGIO AUTOMATICO SU GOOGLE
    # ==================================
    logger = GoogleSheetLogger()
    logger.log_biometric_metrics(
        tester="Francesco",
        phase="NN1 Clean Evaluation",
        attack_type="None",
        epsilon=0.0,
        defense_type="None",
        accuracy=metrics['Accuracy'],
        eer=metrics['EER'],
        far=metrics['FAR_at_opt'],
        frr=metrics['FRR_at_opt'],
        threshold=metrics['Optimal_Threshold'],
        notes="Valutazione baseline su 1000 img clean"
    )