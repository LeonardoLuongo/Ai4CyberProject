"""Script per il tuning degli iperparametri di PGD.
"""

from __future__ import annotations
import sys
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image

# =========================================================================
# WORKAROUND CUDNN: Risolve l'errore CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH
# =========================================================================
os.environ["TORCH_CUDNN_V8_API_ENABLED"] = "0" 
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
# =========================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[5]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.util.attack_common import (
    batched, 
    build_nn1_art_classifier,
    discover_test_images
)

from art.attacks.evasion import ProjectedGradientDescent
from src.util.identity_mapper import IdentityMapper
from facenet_pytorch import MTCNN

# Inizializza il mapper puntando al file CSV
mapper = IdentityMapper(Path("dataset/clean/splits/identity_meta.csv"))

def main() -> int:
    # =========================================================================
    # 1. PARAMETRI DI TUNING E SETUP
    # =========================================================================
    input_dir = Path("dataset/clean/test")
    batch_size = 128  
    max_images = None  # Immagini totali da analizzare prima del filtro

   
    BEST_MAX_ITER = 4 
    
    # VARIABILI DA ESPLORARE PER PGD
    epsilons = [0.001, 0.005, 0.01, 0.015, 0.02, 0.025, 0.05]
    step_multipliers = [1.0, 1.5, 2.5]  
    num_random_inits = [1, 3, 5]        
    
    plots_dir = Path("plots/3_Adversarial_Examples/error_generic/pgd")
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # =========================================================================

    samples = discover_test_images(input_dir)
    if max_images is not None:
        samples = samples[:max_images]

    print("Caricamento NN1 e ART PyTorchClassifier...")
    classifier, num_classes, device_type = build_nn1_art_classifier()
    
    print("Inizializzazione MTCNN...")
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

    X_valid = np.stack(valid_faces_numpy) # Shape: [N, 3, 160, 160]
    Y_valid = np.array(valid_ground_truths)

    # =========================================================================
    # FASE 2: GRID SEARCH AVVERSARIO PGD 
    # =========================================================================
    results_accuracy = {
        eps: {mult: [] for mult in step_multipliers} 
        for eps in epsilons
    }

    print("\n--- Fase 2: Inizio Tuning Hyperparametri PGD ---")
    print(f"Impostato BEST_MAX_ITER = {BEST_MAX_ITER} (Fisso)")

    for eps in epsilons:
        print(f"\n{'='*50}\nInizio test per EPSILON = {eps}\n{'='*50}")
        
        for step_mult in step_multipliers:
            # Formula corretta: passo dinamico dipendente da max_iter
            eps_step = (eps / BEST_MAX_ITER) * step_mult
            
            for num_init in num_random_inits:
                print(f"\nGenerazione PGD: eps={eps}, eps_step={eps_step:.4f} (mult={step_mult}), num_init={num_init}...")
                
                attack = ProjectedGradientDescent(
                    estimator=classifier, 
                    eps=eps, 
                    eps_step=eps_step, 
                    max_iter=BEST_MAX_ITER, 
                    num_random_init=num_init,
                    targeted=False,
                    batch_size=batch_size,
                    verbose=False
                )
                
                correct_predictions = 0
                
                # Iterazione veloce direttamente sugli array filtrati
                for start_idx in range(0, total_valid, batch_size):
                    end_idx = min(start_idx + batch_size, total_valid)
                    
                    batch_x = X_valid[start_idx:end_idx]
                    batch_y = Y_valid[start_idx:end_idx]
                    
                    # Attacco
                    x_adv = attack.generate(x=batch_x)
                    
                    # Predizione
                    adv_preds_raw = classifier.predict(x_adv)
                    adv_preds = np.argmax(adv_preds_raw, axis=1)
                    
                    # Confronto e update accuratezza
                    correct_predictions += np.sum(adv_preds == batch_y)
                
                # Calcolo robust accuracy sui sample validi
                accuracy = correct_predictions / total_valid
                results_accuracy[eps][step_mult].append(accuracy)
                
                print(f"-> Risultato: Robust Accuracy = {accuracy * 100:.2f}%")

    # =========================================================================
    # FASE 3: GENERAZIONE DEL GRAFICO 
    # =========================================================================
    print("\n--- Generazione del Grafico in corso ---")
    plt.figure(figsize=(10, 6))

    num_init_list = [1, 3, 5]
    markers = ['o', 's', '^', 'D', 'v', '*']

    line_idx = 0
    # Iteriamo direttamente sul dizionario che hai creato
    for eps, mult_dict in results_accuracy.items():
        for mult, accuracies in mult_dict.items():
            
            # 'accuracies' è già la lista pronta da tracciare: [acc_init1, acc_init3, acc_init5]
            plt.plot(
                num_init_list, 
                accuracies, 
                marker=markers[line_idx % len(markers)], 
                linestyle='-', 
                linewidth=2,
                label=f'eps = {eps} (mult = {mult})'
            )
            line_idx += 1

    plt.title(f'PGD Hyperparameter Tuning\n(Evaluated on {total_valid} valid crops | max_iter=4)', fontsize=14, fontweight='bold')
    plt.xlabel('Numero di Inizializzazioni (num_init)', fontsize=12)
    plt.ylabel('Model Robust Accuracy', fontsize=12)
    plt.xticks(num_init_list)
    plt.ylim([-0.05, 1.05])
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.legend(title='Epsilon & Step Multiplier', bbox_to_anchor=(1.05, 1), loc='upper left')

    save_path = plots_dir / "pgd_hyperparameter_tuning_cropped.png"
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)

    print(f"✅ Grafico salvato in: {save_path}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())