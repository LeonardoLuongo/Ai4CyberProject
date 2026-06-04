"""Script per il tuning degli iperparametri di Carlini & Wagner L_inf.

Integra MTCNN per il crop iniziale e scarta i sample misclassificati
nella fase clean. Genera attacchi in RAM per testare il Learning Rate.
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

from art.attacks.evasion import CarliniLInfMethod
from src.util.identity_mapper import IdentityMapper
from facenet_pytorch import MTCNN

# Inizializza il mapper
mapper = IdentityMapper(Path("dataset/clean/splits/identity_meta.csv"))

def main() -> int:
    # =========================================================================
    # PARAMETRI DI TUNING E SETUP
    # =========================================================================
    input_dir = Path("dataset/clean/test")
    
    # BATCH SIZE RIDOTTO: C&W usa un ottimizzatore interno che satura la VRAM.
    batch_size = 8  
    
    # Fissiamo l'Epsilon su cui testare l'efficienza dell'ottimizzatore
    test_eps = 0.1 

    # Iperparametri da testare
    learning_rates = [0.003, 0.005]
    max_iters_list = [5] 
    
    plots_dir = Path("plots/3_Adversarial_Examples/error_generic/cw")
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # =========================================================================

    samples = discover_test_images(input_dir)
    
    # CAMPIONAMENTO: Prendiamo 1 immagine ogni 10 per coprire identità diverse
    # senza dover lanciare C&W su migliaia di foto per un semplice tuning.
    samples = samples[::10] 
    
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

    resnet_model = classifier.model 
    
    for sample in samples:
        true_id = mapper.get_facenet_id_by_class_id(sample.identity_id)
        if true_id == -1:
            continue
            
        try:
            img_pil = Image.open(sample.image_path).convert('RGB')
        except Exception:
            continue
            
        faces = mtcnn(img_pil)
        
        if faces is not None:
            faces = faces.to(device)
            with torch.no_grad():
                logits_cropped = resnet_model(faces)
                preds = torch.argmax(logits_cropped, dim=1).cpu().numpy()
            
            if true_id in preds:
                correct_idx = np.where(preds == true_id)[0][0]
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
        return 1

    X_valid = np.stack(valid_faces_numpy)
    Y_valid = np.array(valid_ground_truths)

    # =========================================================================
    # FASE 2: GRID SEARCH AVVERSARIO
    # =========================================================================
    results_accuracy = {lr: [] for lr in learning_rates}

    print("\n--- Fase 2: Inizio Tuning Hyperparametri ---")
    
    for lr in learning_rates:
        print(f"\n{'='*50}\nInizio test per LEARNING RATE = {lr}\n{'='*50}")
        
        for max_iter in max_iters_list:
            print(f"\nGenerazione C&W con max_iter={max_iter}, lr={lr}...")
            
            # NOTA: Rimosso eps=test_eps! Seguiamo la documentazione ufficiale.
            attack = CarliniLInfMethod(
                classifier=classifier, 
                max_iter=max_iter, 
                learning_rate=lr,
                targeted=False,
                batch_size=batch_size,
                verbose=False
            )
            
            # Qui contiamo quante volte il modello "sopravvive"
            model_survivals = 0
            all_perturbations = []
            valid_successes = 0
            within_budget_count = 0
            
            for start_idx in range(0, total_valid, batch_size):
                end_idx = min(start_idx + batch_size, total_valid)
                batch_x = X_valid[start_idx:end_idx]
                batch_y = Y_valid[start_idx:end_idx]
                
                # Generazione attacco libero
                x_adv = attack.generate(x=batch_x, y=batch_y)
                
                # Predizione avversaria
                adv_preds_raw = classifier.predict(x_adv)
                adv_preds = np.argmax(adv_preds_raw, axis=1)
                
                # --- L'ARBITRO (Il tuo metrics.py inline per velocità) ---
                # Calcola la perturbazione massima su ogni asse spaziale e di canale
                # Questo equivale alla tua funzione validate_linf_batch
                perturbations = np.max(np.abs(x_adv - batch_x), axis=(1, 2, 3))
                is_within_budget = perturbations <= (test_eps + 1e-5) # Tolleranza float
                all_perturbations.extend(perturbations.tolist())
                within_budget_count += np.sum(is_within_budget)
                valid_successes += np.sum((adv_preds != batch_y) & is_within_budget)
                
                for i in range(len(batch_y)):
                    prediction_is_correct = (adv_preds[i] == batch_y[i])
                    
                    if prediction_is_correct:
                        # 1. Il modello ha indovinato. Ha resistito!
                        model_survivals += 1
                    elif not is_within_budget[i]:
                        # 2. Il modello ha sbagliato, MA l'attacco ha sforato il budget di 0.05.
                        # Per le nostre regole, l'attacco ha fallito, quindi il modello è "salvo".
                        model_survivals += 1
                    else:
                        # 3. L'attacco ha ingannato il modello E ha rispettato il budget.
                        # Il modello non sopravvive.
                        pass

            all_perturbations = np.array(all_perturbations)
            print(
                f"   Linf stats: "
                f"min={np.min(all_perturbations):.4f}, "
                f"mean={np.mean(all_perturbations):.4f}, "
                f"median={np.median(all_perturbations):.4f}, "
                f"p95={np.percentile(all_perturbations, 95):.4f}, "
                f"max={np.max(all_perturbations):.4f}"
            )
            print(
                f"   Within budget: {within_budget_count}/{total_valid} "
                f"({within_budget_count / total_valid * 100:.2f}%)"
            )
            print(
                f"   Successful attacks within budget: {valid_successes}/{total_valid} "
                f"({valid_successes / total_valid * 100:.2f}%)"
            )

            accuracy = model_survivals / total_valid
            results_accuracy[lr].append(accuracy)
            
            print(f"-> Risultato: Robust Accuracy (per eps <= {test_eps}) = {accuracy * 100:.2f}%")

            # =========================================================================
            # EARLY STOPPING LOGIC
            # =========================================================================
            if model_survivals == 0:
                print("   [!] Accuracy crollata a 0.00%. Salto iterazioni successive.")
                remaining_iters = len(max_iters_list) - len(results_accuracy[lr])
                results_accuracy[lr].extend([0.0] * remaining_iters)
                break

    # =========================================================================
    # FASE 3: GENERAZIONE DEL GRAFICO
    # =========================================================================
    print("\n--- Generazione del Grafico in corso ---")
    plt.figure(figsize=(10, 6))
    
    markers = ['o', 's', '^', 'D', 'v']
    for i, lr in enumerate(learning_rates):
        plt.plot(
            max_iters_list, 
            results_accuracy[lr], 
            marker=markers[i % len(markers)], 
            linestyle='-', 
            linewidth=2,
            label=f'Learning Rate = {lr}'
        )

    plt.title(f'C&W Hyperparameter Tuning (eps={test_eps})\n(Evaluated on {total_valid} clean-verified crops)', fontsize=14, fontweight='bold')
    plt.xlabel('Numero di Iterazioni (max_iter)', fontsize=12)
    plt.ylabel('Model Robust Accuracy', fontsize=12)
    plt.xticks(max_iters_list)
    plt.ylim([-0.05, 1.05])
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title='Optimizer LR')
    
    save_path = plots_dir / "cw_hyperparameter_tuning_cropped.png"
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    
    print(f"✅ Grafico salvato in: {save_path}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())