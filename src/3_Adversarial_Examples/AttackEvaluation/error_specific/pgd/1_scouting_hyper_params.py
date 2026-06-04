"""Script unificato per il tuning degli iperparametri di PGD (Error-Specific).

Integra MTCNN per il crop iniziale e seleziona SOLO i sample classificati
correttamente. Per ogni sample calcola il target 'least-likely' usando
le utils condivise e tenta un attacco mirato. 
Ottimizza eps_step e num_random_init misurando il t-ASR.
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

# Setup Path (coerente con i vostri script)
PROJECT_ROOT = Path(__file__).resolve().parents[5]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from util.attack_common import build_nn1_art_classifier, discover_test_images
from util.identity_mapper import IdentityMapper
from util.attack_error_specific_utils import select_target_label, get_one_hot_target

from facenet_pytorch import MTCNN
from art.attacks.evasion import ProjectedGradientDescent

def main() -> int:
    print("======================================================")
    print(" SCOUTING IPERPARAMETRI: PGD TARGETED (least-likely)  ")
    print("======================================================\n")

    # =========================================================================
    # 1. PARAMETRI DI TUNING E SETUP
    # =========================================================================
    base_dir = Path(os.getcwd())
    input_dir = base_dir / "dataset" / "clean" / "test"
    meta_csv_path = base_dir / "dataset" / "clean" / "splits" / "identity_meta.csv"
    
    batch_size = 64  # Abbassato leggermente rispetto all'untargeted per la memoria extra
    max_images = 300 # Cap suggerito per lo scouting (least-likely è il target più tosto)

    # -> N.B. Per attacchi targeted, BEST_MAX_ITER spesso richiede valori più alti 
    # rispetto al generic (es. 10 o 20) per forzare l'immagine verso il target.
    BEST_MAX_ITER = 10 
    
    # VARIABILI DA ESPLORARE PER PGD SPECIFIC
    epsilons = [0.01, 0.02, 0.03, 0.04, 0.05] # Valori solitamente più alti per il targeted
    step_multipliers = [1.0, 1.5, 2.0]  
    num_random_inits = [1, 3]        
    
    # Cartella di output per lo scouting
    plots_dir = base_dir / "plots" / "3_Adversarial_Examples" / "error_specific" / "pgd" / "scouting"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mapper = IdentityMapper(meta_csv_path)
    # =========================================================================

    samples = discover_test_images(input_dir)
    if max_images is not None:
        samples = samples[:max_images]

    print(f"-> Inizializzazione Reti su {device}...")
    classifier, num_classes, device_type = build_nn1_art_classifier()
    mtcnn = MTCNN(image_size=160, margin=0, keep_all=True, post_process=True, device=device)
    resnet_model = classifier.model 

    # =========================================================================
    # FASE 1: PRE-FILTRAGGIO CLEAN E SELEZIONE TARGET "LEAST-LIKELY"
    # =========================================================================
    print(f"\n[FASE 1] Estrazione e Calcolo Target su {len(samples)} immagini...")
    
    valid_faces_numpy = []
    valid_targets = []
    
    faces_not_found = 0
    misclassified = 0
    
    for sample in samples:
        true_id = mapper.get_facenet_id_by_class_id(sample.identity_id)
        if true_id == -1: continue
            
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
                
                # Salvataggio del tensore formattato per ART
                correct_face = faces[correct_idx].cpu().numpy()
                correct_face_01 = (correct_face + 1.0) / 2.0
                
                # ---> LOGICA ERROR-SPECIFIC <---
                # Estraiamo i logit di QUELLA specifica faccia
                best_logits = logits_cropped[correct_idx].cpu().numpy()
                
                # Espandiamo la dimensione (1, 8631) come richiesto dalla funzione del tuo collega
                best_logits_exp = np.expand_dims(best_logits, axis=0)
                
                # Calcoliamo la least-likely
                target_id = select_target_label(
                    best_logits_exp, 
                    true_id, 
                    strategy="least-likely", 
                    num_classes=mapper.get_num_training_classes()
                )
                
                valid_faces_numpy.append(correct_face_01)
                valid_targets.append(target_id)
            else:
                misclassified += 1
        else:
            faces_not_found += 1

    total_valid = len(valid_faces_numpy)
    print(f"-> Risultati Pre-filtraggio:")
    print(f"   * Facce non rilevate: {faces_not_found}")
    print(f"   * Misclassificate: {misclassified}")
    print(f"   * 🟢 Volti validi pronti per l'attacco Targeted: {total_valid}")

    if total_valid == 0:
        print("\n[ERRORE] Nessuna immagine ha superato il filtro. Interruzione.")
        return 1

    X_valid = np.stack(valid_faces_numpy) # Shape: [N, 3, 160, 160]
    Y_targets = np.array(valid_targets)   # Shape: [N]

    # =========================================================================
    # FASE 2: GRID SEARCH AVVERSARIO PGD TARGETED
    # =========================================================================
    results_t_asr = {eps: {mult: [] for mult in step_multipliers} for eps in epsilons}

    print("\n[FASE 2] Inizio Tuning Hyperparametri PGD TARGETED")
    print(f"-> Impostato BEST_MAX_ITER = {BEST_MAX_ITER} (Fisso)")

    for eps in epsilons:
        print(f"\n{'-'*50}\nInizio test per EPSILON = {eps}\n{'-'*50}")
        
        for step_mult in step_multipliers:
            eps_step = (eps / BEST_MAX_ITER) * step_mult
            
            for num_init in num_random_inits:
                print(f" -> Generazione: eps_step={eps_step:.4f} (mult={step_mult}), init={num_init}...")
                
                # Inizializziamo PGD in modalità TARGETED
                attack = ProjectedGradientDescent(
                    estimator=classifier, 
                    eps=eps, 
                    eps_step=eps_step, 
                    max_iter=BEST_MAX_ITER, 
                    num_random_init=num_init,
                    targeted=True,          # <--- FONDAMENTALE
                    batch_size=batch_size,
                    verbose=False
                )
                
                successful_targeted_attacks = 0
                
                for start_idx in range(0, total_valid, batch_size):
                    end_idx = min(start_idx + batch_size, total_valid)
                    
                    batch_x = X_valid[start_idx:end_idx]
                    batch_y_tgt = Y_targets[start_idx:end_idx]
                    
                    # INTEGRAZIONE COLLEGHI: get_one_hot_target restituisce (1, num_classes).
                    # Dobbiamo concatenarli lungo l'asse 0 per formare il batch corretto: (BATCH, num_classes)
                    batch_y_one_hot = np.concatenate([
                        get_one_hot_target(tgt, num_classes=mapper.get_num_training_classes()) 
                        for tgt in batch_y_tgt
                    ], axis=0)
                    
                    # Generazione attacco fornendo le etichette y
                    x_adv = attack.generate(x=batch_x, y=batch_y_one_hot)
                    
                    # Predizione sulle immagini perturbate
                    adv_preds_raw = classifier.predict(x_adv)
                    adv_preds = np.argmax(adv_preds_raw, axis=1)
                    
                    # Calcolo dei successi (Quanti sono stati classificati ESATTAMENTE come il target?)
                    successful_targeted_attacks += np.sum(adv_preds == batch_y_tgt)
                
                # t-ASR: % di immagini che sono state misclassificate nella classe voluta
                t_asr = successful_targeted_attacks / total_valid
                results_t_asr[eps][step_mult].append(t_asr)
                
                print(f"    Risultato: t-ASR = {t_asr * 100:.2f}%")

    # =========================================================================
    # FASE 3: GENERAZIONE DEL GRAFICO (t-ASR)
    # =========================================================================
    print("\n[FASE 3] Generazione del Grafico in corso...")
    plt.figure(figsize=(10, 6))

    markers = ['o', 's', '^', 'D', 'v', '*']
    line_idx = 0
    
    for eps, mult_dict in results_t_asr.items():
        for mult, asrs in mult_dict.items():
            plt.plot(
                num_random_inits, 
                asrs, 
                marker=markers[line_idx % len(markers)], 
                linestyle='-', 
                linewidth=2,
                label=f'eps = {eps} (mult = {mult})'
            )
            line_idx += 1

    # N.B. In questo grafico, linee PIÙ ALTE significano prestazioni MIGLIORI
    plt.title(f'PGD Targeted (least-likely) Hyperparameter Tuning\nEvaluated on {total_valid} valid crops | max_iter={BEST_MAX_ITER}', fontsize=14, fontweight='bold')
    plt.xlabel('Numero di Inizializzazioni (num_init)', fontsize=12)
    plt.ylabel('Targeted Attack Success Rate (t-ASR)', fontsize=12) 
    plt.xticks(num_random_inits)
    plt.ylim([-0.05, 1.05])
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.legend(title='Epsilon & Step Multiplier', bbox_to_anchor=(1.05, 1), loc='upper left')

    save_path = plots_dir / "pgd_specific_hyperparameter_tuning.png"
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)

    print(f"✅ Grafico salvato in: {save_path}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())