"""Script per il tuning degli iperparametri di BIM.

Integra MTCNN per il crop iniziale e scarta i sample misclassificati
nella fase clean, applicando l'attacco solo sui volti validi.
"""

from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[5]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.util.attack_common import (
    batched, 
    build_nn1_art_classifier,
    discover_test_images
)

from art.attacks.evasion import BasicIterativeMethod
from src.util.identity_mapper import IdentityMapper
from facenet_pytorch import MTCNN

# Inizializza il mapper puntando al file CSV
mapper = IdentityMapper(Path("dataset/clean/splits/identity_meta.csv"))

def main() -> int:
    # =========================================================================
    # PARAMETRI DI TUNING E SETUP
    # =========================================================================
    input_dir = Path("dataset/clean/test")
    batch_size = 128  
    max_images = None  # Immagini totali da analizzare prima del filtro

    epsilons = [0.025, 0.05, 0.075, 0.1]  
    max_iters_list = [1, 2, 4, 8, 16, 20, 24]  
    
    plots_dir = Path("plots/3_Adversarial_Examples/error_generic/bim")
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # =========================================================================

    samples = discover_test_images(input_dir)
    if max_images is not None:
        samples = samples[:max_images]

    print("Caricamento NN1 e ART PyTorchClassifier...")
    classifier, num_classes, device_type = build_nn1_art_classifier()
    
    print("Inizializzazione MTCNN...")
    # keep_all=True ci permette di valutare tutte le facce nella foto
    mtcnn = MTCNN(image_size=160, margin=0, keep_all=True, post_process=True, device=device)

    # =========================================================================
    # FASE 1: PRE-FILTRAGGIO CLEAN (MTCNN + VALUTAZIONE)
    # =========================================================================
    print(f"\n--- Fase 1: Estrazione e Filtraggio di {len(samples)} immagini ---")
    
    valid_faces_numpy = []
    valid_ground_truths = []
    faces_not_found = 0
    misclassified = 0

    # Estraiamo il modello PyTorch puro da ART per fare un'inferenza velocissima
    resnet_model = classifier.model 
    
    for sample in samples:
        true_id = mapper.get_facenet_id_by_class_id(sample.identity_id)
        if true_id == -1:
            continue # Classe non valida, saltiamo
            
        try:
            img_pil = Image.open(sample.image_path).convert('RGB')
        except Exception:
            continue
            
        # 1. Crop con MTCNN
        faces = mtcnn(img_pil)
        
        if faces is not None:
            faces = faces.to(device)
            
            # 2. Classificazione Clean sui volti trovati
            with torch.no_grad():
                logits_cropped = resnet_model(faces)
                preds = torch.argmax(logits_cropped, dim=1).cpu().numpy()
            
            # 3. Controllo del Successo
            if true_id in preds:
                # Troviamo l'indice esatto del volto che ha fatto centro
                correct_idx = np.where(preds == true_id)[0][0]
                
                # Salviamo il tensore specifico (convertito in numpy per ART)
                # La shape sarà [3, 160, 160]
                correct_face = faces[correct_idx].cpu().numpy()
                correct_face_01 = (correct_face + 1.0) / 2.0
                
                valid_faces_numpy.append(correct_face_01)
                valid_ground_truths.append(true_id)
            else:
                misclassified += 1
        else:
            faces_not_found += 1

    total_valid = len(valid_faces_numpy)
    print(f"Risultati Pre-filtraggio:")
    print(f" -> Facce non rilevate da MTCNN: {faces_not_found}")
    print(f" -> Misclassificate pulite: {misclassified}")
    print(f" -> 🟢 Volti validi pronti per l'attacco: {total_valid}")

    if total_valid == 0:
        print("\n[ERRORE] Nessuna immagine ha superato il filtro. Interruzione.")
        return 1

    # Convertiamo le liste in array numpy globali per fare slicing veloce
    X_valid = np.stack(valid_faces_numpy) # Shape: [N, 3, 160, 160]
    Y_valid = np.array(valid_ground_truths)

    # =========================================================================
    # FASE 2: GRID SEARCH AVVERSARIO
    # =========================================================================
    results_accuracy = {eps: [] for eps in epsilons}

    print("\n--- Fase 2: Inizio Tuning Hyperparametri ---")
    for eps in epsilons:
        print(f"\n{'='*50}\nInizio test per EPSILON = {eps}\n{'='*50}")
        eps_step = eps / 24.0 
        
        for max_iter in max_iters_list:
            print(f"\nGenerazione attacco con max_iter={max_iter}, eps_step={eps_step:.4f}...")
            
            attack = BasicIterativeMethod(
                estimator=classifier, 
                eps=eps, 
                eps_step=eps_step if eps_step else 0.0001, 
                max_iter=max_iter, 
                targeted=False,
                batch_size=batch_size
            )
            
            correct_predictions = 0
            
            # Non usiamo più load_batch_for_nn1, iteriamo direttamente sugli array NumPy validi!
            for start_idx in range(0, total_valid, batch_size):
                end_idx = min(start_idx + batch_size, total_valid)
                
                batch_x = X_valid[start_idx:end_idx]
                batch_y = Y_valid[start_idx:end_idx]
                
                # Attacco
                x_adv = attack.generate(x=batch_x)
                
                # Predizione
                adv_preds_raw = classifier.predict(x_adv)
                adv_preds = np.argmax(adv_preds_raw, axis=1)
                
                # Confronto diretto vettoriale 
                correct_predictions += np.sum(adv_preds == batch_y)
            
            # Calcolo accuracy usando il nuovo totale filtrato!
            accuracy = correct_predictions / total_valid
            results_accuracy[eps].append(accuracy)
            
            print(f"-> Risultato: Robust Accuracy = {accuracy * 100:.2f}%")

            # =========================================================================
            # EARLY STOPPING LOGIC
            # =========================================================================
            if correct_predictions == 0:
                print("   [!] Accuracy crollata a 0.00%. Salto iterazioni successive.")
                remaining_iters = len(max_iters_list) - len(results_accuracy[eps])
                results_accuracy[eps].extend([0.0] * remaining_iters)
                break

    # =========================================================================
    # FASE 3: GENERAZIONE DEL GRAFICO
    # =========================================================================
    print("\n--- Generazione del Grafico in corso ---")
    plt.figure(figsize=(10, 6))
    
    markers = ['o', 's', '^', 'D', 'v']
    for i, eps in enumerate(epsilons):
        plt.plot(
            max_iters_list, 
            results_accuracy[eps], 
            marker=markers[i % len(markers)], 
            linestyle='-', 
            linewidth=2,
            label=f'eps = {eps} (step = {eps/24:.4f})'
        )

    plt.title(f'BIM Hyperparameter Tuning\n(Evaluated on {total_valid} clean-verified crops)', fontsize=14, fontweight='bold')
    plt.xlabel('Numero di Iterazioni (max_iter)', fontsize=12)
    plt.ylabel('Model Robust Accuracy', fontsize=12)
    plt.xticks(max_iters_list)
    plt.ylim([-0.05, 1.05])
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title='Epsilon Values')
    
    save_path = plots_dir / "bim_hyperparameter_tuning_cropped.png"
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    
    print(f"✅ Grafico salvato in: {save_path}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())