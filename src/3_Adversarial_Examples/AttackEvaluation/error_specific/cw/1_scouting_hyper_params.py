"""Script per il tuning degli iperparametri di PGD TARGETED (Distribuzione a 3 stati).

Coerente con lo script BIM del collega: traccia la distribuzione a 3 stati 
(Resisted, Untargeted Success, Targeted Success) valutando l'evoluzione
dell'attacco all'aumentare dell'Epsilon.
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
# WORKAROUND CUDNN
# =========================================================================
os.environ["TORCH_CUDNN_V8_API_ENABLED"] = "0" 
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

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
    print(" SCOUTING PGD: DISTRIBUZIONE A 3 STATI (Worst-Case)   ")
    print("======================================================\n")

    # =========================================================================
    # PARAMETRI DI TUNING E SETUP
    # =========================================================================
    base_dir = Path(os.getcwd())
    input_dir = base_dir / "dataset" / "clean" / "test"
    meta_csv_path = base_dir / "dataset" / "clean" / "splits" / "identity_meta.csv"
    
    batch_size = 64  
    max_images = None 

    # Fissiamo i parametri che abbiamo scoperto essere ottimali prima
    BEST_MAX_ITER = 10
    NUM_INIT = 1
    
    # Variabiliamo Epsilon e Moltiplicatore per il grafico coerente col collega
    epsilons = [0.01, 0.02, 0.03, 0.04, 0.05]
    step_multipliers = [1.0, 1.5, 2.0]  
    
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
    # FASE 1: PRE-FILTRAGGIO E SALVATAGGIO CLASSI REALI E TARGET
    # =========================================================================
    print(f"\n[FASE 1] Estrazione Volti e Calcolo Hardest Target (Least-Likely)...")
    
    valid_faces = []
    hardest_targets = []
    true_classes = [] # FONDAMENTALE PER CALCOLARE I 3 STATI
    
    for sample in samples:
        true_id = mapper.get_facenet_id_by_class_id(sample.identity_id)
        if true_id == -1: continue
            
        try:
            img_pil = Image.open(sample.image_path).convert('RGB')
        except Exception: continue
            
        faces = mtcnn(img_pil)
        if faces is not None:
            faces = faces.to(device)
            with torch.no_grad():
                logits = resnet_model(faces)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
            
            if true_id in preds:
                correct_idx = np.where(preds == true_id)[0][0]
                
                face_tensor = faces[correct_idx].cpu().numpy()
                face_tensor_01 = (face_tensor + 1.0) / 2.0
                
                clean_logits = np.expand_dims(logits[correct_idx].cpu().numpy(), axis=0)
                
                least_likely_class = select_target_label(
                    clean_predictions=clean_logits, 
                    true_label=true_id, 
                    strategy="least-likely", 
                    num_classes=num_classes
                )
                
                valid_faces.append(face_tensor_01)
                hardest_targets.append(least_likely_class)
                true_classes.append(true_id)

    total_valid = len(valid_faces)
    print(f" -> 🟢 Volti validi pronti per l'attacco Worst-Case: {total_valid}")
    if total_valid == 0: return 1

    X_valid = np.stack(valid_faces)
    Y_targets = np.array(hardest_targets)
    Y_true = np.array(true_classes)

    # =========================================================================
    # FASE 2: GRID SEARCH CON ANALISI A 3 STATI
    # =========================================================================
    # Struttura dati per il grafico
    results_dist = {mult: {'resisted': [], 'untargeted': [], 'targeted': []} for mult in step_multipliers}

    print("\n[FASE 2] Inizio Generazione PGD (Analisi 3 Stati)")
    
    for mult in step_multipliers:
        print(f"\n{'-'*60}\nInizio test per Moltiplicatore = {mult}\n{'-'*60}")
        
        for eps in epsilons:
            eps_step = (eps / BEST_MAX_ITER) * mult
            
            attack = ProjectedGradientDescent(
                estimator=classifier, 
                eps=eps, 
                eps_step=eps_step, 
                max_iter=BEST_MAX_ITER, 
                num_random_init=NUM_INIT,
                targeted=True,
                batch_size=batch_size,
                verbose=False
            )
            
            count_resisted = 0
            count_targeted = 0
            count_untargeted = 0
            
            for start_idx in range(0, total_valid, batch_size):
                end_idx = min(start_idx + batch_size, total_valid)
                batch_x = X_valid[start_idx:end_idx]
                batch_targets = Y_targets[start_idx:end_idx]
                batch_true_ids = Y_true[start_idx:end_idx]
                
                batch_y_onehot = np.concatenate([get_one_hot_target(t, num_classes) for t in batch_targets], axis=0)
                
                x_adv = attack.generate(x=batch_x, y=batch_y_onehot)
                adv_preds = np.argmax(classifier.predict(x_adv), axis=1)
                
                # --- CALCOLO DEI 3 STATI (Identico al collega) ---
                resisted_mask = (adv_preds == batch_true_ids)
                count_resisted += np.sum(resisted_mask)
                
                targeted_mask = (adv_preds == batch_targets)
                count_targeted += np.sum(targeted_mask)
                
                untargeted_mask = (~resisted_mask) & (~targeted_mask)
                count_untargeted += np.sum(untargeted_mask)
            
            p_res = count_resisted / total_valid
            p_unt = count_untargeted / total_valid
            p_tar = count_targeted / total_valid
            
            results_dist[mult]['resisted'].append(p_res)
            results_dist[mult]['untargeted'].append(p_unt)
            results_dist[mult]['targeted'].append(p_tar)
            
            print(f"Eps: {eps:.2f} | 🔴 Target: {p_tar*100:5.2f}% | 🟡 Untarg: {p_unt*100:5.2f}% | 🟢 Resist: {p_res*100:5.2f}%")

    # =========================================================================
    # FASE 3: GENERAZIONE DEL GRAFICO (Subplots con Barre Impilate)
    # =========================================================================
    print("\n[FASE 3] Generazione Grafico a Barre Impilate...")
    
    num_mults = len(step_multipliers)
    fig, axes = plt.subplots(1, num_mults, figsize=(6 * num_mults, 7), sharey=True)
    
    # Colori allineati al grafico BIM del collega
    color_resisted = 'forestgreen'
    color_untargeted = 'gold'
    color_targeted = 'firebrick'
    
    x_positions = np.arange(len(epsilons))
    bar_width = 0.6
    
    for i, mult in enumerate(step_multipliers):
        ax = axes[i] if num_mults > 1 else axes
        
        y_res = np.array(results_dist[mult]['resisted']) * 100
        y_unt = np.array(results_dist[mult]['untargeted']) * 100
        y_tar = np.array(results_dist[mult]['targeted']) * 100
        
        bar1 = ax.bar(x_positions, y_res, bar_width, label='Model Resisted', color=color_resisted, edgecolor='white')
        bar2 = ax.bar(x_positions, y_unt, bar_width, bottom=y_res, label='Untargeted Success', color=color_untargeted, edgecolor='white')
        bar3 = ax.bar(x_positions, y_tar, bar_width, bottom=y_res+y_unt, label='Targeted Success', color=color_targeted, edgecolor='white')
        
        # Testi dentro le barre
        for j in range(len(epsilons)):
            if y_res[j] > 4:
                ax.text(x_positions[j], y_res[j]/2, f"{y_res[j]:.1f}%", ha='center', va='center', color='white', fontweight='bold', fontsize=10)
            if y_unt[j] > 4:
                # Scritta nera sul giallo per maggiore leggibilità
                ax.text(x_positions[j], y_res[j] + y_unt[j]/2, f"{y_unt[j]:.1f}%", ha='center', va='center', color='black', fontweight='bold', fontsize=10)
            if y_tar[j] > 4:
                ax.text(x_positions[j], y_res[j] + y_unt[j] + y_tar[j]/2, f"{y_tar[j]:.1f}%", ha='center', va='center', color='white', fontweight='bold', fontsize=10)

        ax.set_title(f"Step Multiplier: {mult}", fontsize=14, fontweight='bold')
        ax.set_xticks(x_positions)
        ax.set_xticklabels([f"{e:.2f}" for e in epsilons], fontsize=12)
        ax.set_xlabel(r'Perturbation Budget ($\epsilon$)', fontsize=13)
        ax.grid(axis='y', linestyle='--', alpha=0.7)

    if num_mults > 1:
        axes[0].set_ylabel('Percentage of Test Set (%)', fontsize=14)
    else:
        axes.set_ylabel('Percentage of Test Set (%)', fontsize=14)

    # Legenda globale in alto
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles[::-1], labels[::-1], loc='upper center', bbox_to_anchor=(0.5, 1.08), ncol=3, fontsize=12)
    
    plt.suptitle(f"PGD Targeted Distribution Analysis (Least-Likely, max_iter={BEST_MAX_ITER})", fontsize=18, fontweight='bold', y=1.15)
    
    save_path = plots_dir / "pgd_targeted_distribution_tuning10.png"
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    print(f"✅ Grafico a Barre Impilate salvato in: {save_path}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())