"""Script per il tuning degli iperparametri di BIM TARGETED.

Simula lo scenario peggiore (Worst-Case) puntando alla classe "Least Likely".
Usa la regola di Madry (eps_step = eps / max_iter * 2.5) per testare l'efficienza
al variare del budget Epsilon.
"""

from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[5]
if sys.path[0] != str(PROJECT_ROOT):
    sys.path.insert(0, str(PROJECT_ROOT))

from src.util.attack_common import (
    batched, 
    build_nn1_art_classifier,
    discover_test_images
)
from src.util.attack_error_specific_utils import select_target_label, get_one_hot_target
from art.attacks.evasion import BasicIterativeMethod
from src.util.identity_mapper import IdentityMapper
from facenet_pytorch import MTCNN

mapper = IdentityMapper(Path("dataset/clean/splits/identity_meta.csv"))

def main() -> int:
    # =========================================================================
    # PARAMETRI DI TUNING E SETUP
    # =========================================================================
    input_dir = Path("dataset/clean/test")
    batch_size = 32  
    max_images = None  # Set to None per processare tutte le immagini disponibili  
    
    # Nuova logica: testiamo la convergenza su 4 budget Epsilon diversi
    epsilons = [0.025, 0.05, 0.1]
    max_iters_list = [4, 8, 16, 32] 
    
    # Fattore moltiplicativo di Madry (Permette esplorazione sul bordo)
    BIM_MULT = 2.5 
    
    plots_dir = Path("plots/3_Adversarial_Examples/error_specific/bim")
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # =========================================================================

    samples = discover_test_images(input_dir)[:max_images]
    
    print("Caricamento NN1 e ART PyTorchClassifier...")
    classifier, num_classes, device_type = build_nn1_art_classifier()
    mtcnn = MTCNN(image_size=160, margin=0, keep_all=True, post_process=True, device=device)

    # =========================================================================
    # FASE 1: PRE-FILTRAGGIO E SALVATAGGIO CLASSI REALI
    # =========================================================================
    print(f"\n--- Fase 1: Estrazione Volti e Calcolo Hardest Target ---")
    
    valid_faces = []
    hardest_targets = []
    true_classes = [] 
    
    resnet_model = classifier.model 
    
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
    # FASE 2: GRID SEARCH AVVERSARIO TARGETED CON ANALISI A 3 STATI
    # =========================================================================
    # Struttura dati: per ogni Epsilon, salviamo le liste dei 3 stati
    results_dist = {eps: {'resisted': [], 'untargeted': [], 'targeted': []} for eps in epsilons}

    print("\n--- Fase 2: Inizio Tuning Hyperparametri Targeted (Madry's Rule) ---")
    
    for eps in epsilons:
        print(f"\n{'='*60}\nInizio test per EPSILON = {eps}\n{'='*60}")
        
        for max_iter in max_iters_list:
            
            # === CALCOLO DINAMICO DEL PASSO ===
            eps_step = (eps / max_iter) * BIM_MULT
            
            attack = BasicIterativeMethod(
                estimator=classifier, 
                eps=eps, 
                eps_step=eps_step, 
                max_iter=max_iter, 
                targeted=True,
                batch_size=batch_size
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
                
                # Attacco Mirato
                x_adv = attack.generate(x=batch_x, y=batch_y_onehot)
                adv_preds = np.argmax(classifier.predict(x_adv), axis=1)
                
                # --- CALCOLO DEI 3 STATI ---
                resisted_mask = (adv_preds == batch_true_ids)
                count_resisted += np.sum(resisted_mask)
                
                targeted_mask = (adv_preds == batch_targets)
                count_targeted += np.sum(targeted_mask)
                
                untargeted_mask = (~resisted_mask) & (~targeted_mask)
                count_untargeted += np.sum(untargeted_mask)
            
            # Percentuali
            p_res = count_resisted / total_valid
            p_unt = count_untargeted / total_valid
            p_tar = count_targeted / total_valid
            
            results_dist[eps]['resisted'].append(p_res)
            results_dist[eps]['untargeted'].append(p_unt)
            results_dist[eps]['targeted'].append(p_tar)
            
            print(f"Iter: {max_iter:2d} (step: {eps_step:.4f}) | "
                  f"🔴 Target: {p_tar*100:5.2f}% | "
                  f"🟡 Untarg: {p_unt*100:5.2f}% | "
                  f"🟢 Resist: {p_res*100:5.2f}%")

    # =========================================================================
    # FASE 3: GENERAZIONE DEL GRAFICO (Subplots con Barre Impilate)
    # =========================================================================
    print("\n--- Generazione Grafico a Barre Impilate ---")
    
    num_eps = len(epsilons)
    fig, axes = plt.subplots(1, num_eps, figsize=(6 * num_eps, 7), sharey=True)
    
    color_resisted = 'forestgreen'
    color_untargeted = 'gold'
    color_targeted = 'firebrick'
    
    x_positions = np.arange(len(max_iters_list))
    bar_width = 0.6
    
    for i, eps in enumerate(epsilons):
        ax = axes[i] if num_eps > 1 else axes
        
        y_res = np.array(results_dist[eps]['resisted']) * 100
        y_unt = np.array(results_dist[eps]['untargeted']) * 100
        y_tar = np.array(results_dist[eps]['targeted']) * 100
        
        # Disegno delle barre
        bar1 = ax.bar(x_positions, y_res, bar_width, label='Model Resisted', color=color_resisted)
        bar2 = ax.bar(x_positions, y_unt, bar_width, bottom=y_res, label='Untargeted Success', color=color_untargeted)
        bar3 = ax.bar(x_positions, y_tar, bar_width, bottom=y_res+y_unt, label='Targeted Success', color=color_targeted)
        
        # Aggiunta delle percentuali all'interno delle barre
        for j in range(len(max_iters_list)):
            if y_res[j] > 3:
                ax.text(x_positions[j], y_res[j]/2, f"{y_res[j]:.1f}%", ha='center', va='center', color='white', fontweight='bold', fontsize=9)
            if y_unt[j] > 3:
                ax.text(x_positions[j], y_res[j] + y_unt[j]/2, f"{y_unt[j]:.1f}%", ha='center', va='center', color='white', fontweight='bold', fontsize=9)
            if y_tar[j] > 3:
                ax.text(x_positions[j], y_res[j] + y_unt[j] + y_tar[j]/2, f"{y_tar[j]:.1f}%", ha='center', va='center', color='white', fontweight='bold', fontsize=9)

        ax.set_title(f"Budget L\u221E: \u03B5 = {eps}", fontsize=14, fontweight='bold')
        ax.set_xticks(x_positions)
        ax.set_xticklabels(max_iters_list, fontsize=12)
        ax.set_xlabel('Max Iterations', fontsize=13)
        ax.grid(axis='y', linestyle='--', alpha=0.7)

    if num_eps > 1:
        axes[0].set_ylabel('Percentage of Test Set (%)', fontsize=14)
    else:
        axes.set_ylabel('Percentage of Test Set (%)', fontsize=14)

    # Legenda globale in alto
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles[::-1], labels[::-1], loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=3, fontsize=12)
    
    plt.suptitle(f"BIM Targeted Distribution Analysis (PGD Heuristic: \u03B1 = \u03B5/N * 2.5)", fontsize=18, fontweight='bold', y=1.12)
    
    save_path = plots_dir / "bim_targeted_distribution_tuning.png"
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    print(f"✅ Grafico a Barre Impilate salvato in: {save_path}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())