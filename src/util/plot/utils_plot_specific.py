# FILE: src/util/plot/utils_plot_specific.py

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Impostazioni estetiche globali
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

def save_or_show(save_flag: bool, save_path: str):
    if save_flag and save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"[PLOT] Grafico salvato in: {save_path}")
    else:
        plt.show(block=True)
    plt.close()


def plot_targeted_success_curve(epsilons: list, 
                                asr_dict: dict, 
                                model_name: str = "NN1",
                                save_flag: bool = False, 
                                save_path: str = None):
    """
    Genera la curva del Targeted Attack Success Rate (t-ASR).
    Al contrario dell'accuratezza, questa curva SALE all'aumentare dell'efficacia dell'attacco.
    """
    plt.figure(figsize=(10, 6))
    colors = sns.color_palette("Dark2", n_colors=len(asr_dict))
    
    is_fraction = any(max(asr) <= 1.0 for asr in asr_dict.values() if len(asr) > 0)
    multiplier = 100.0 if is_fraction else 1.0

    for idx, (attack_name, asr_list) in enumerate(asr_dict.items()):
        asr_percent = [a * multiplier for a in asr_list]
        plt.plot(epsilons, asr_percent, marker='s', linewidth=2.5, markersize=8, 
                 color=colors[idx], label=attack_name)

    plt.title(f"Targeted Attack Success Rate (t-ASR) - {model_name}\nError-Specific Attacks", fontsize=16, pad=15)
    plt.xlabel(r"Perturbation Budget ($L_\infty$ $\epsilon$)", fontsize=14)
    plt.ylabel("Targeted Success Rate (%)", fontsize=14)
    
    # Asse Y da 0 a 100%
    plt.ylim(-5, 105)
    plt.xlim(min(epsilons) - 0.005, max(epsilons) + 0.005)
    
    plt.legend(loc='upper left', fontsize=12)
    save_or_show(save_flag, save_path)


def plot_target_confidence_growth(epsilons: list, 
                                  target_confidence_data: list, 
                                  attack_name: str = "Targeted PGD",
                                  save_flag: bool = False, 
                                  save_path: str = None):
    """
    Genera un Boxplot che mostra come la probabilità della classe BERSAGLIO 
    cresca all'aumentare di epsilon, forzando la mano della rete.
    """
    plt.figure(figsize=(10, 6))
    
    # Palette verde/blu ascendente per indicare il "guadagno" dell'attaccante
    sns.boxplot(data=target_confidence_data, palette="YlGnBu")
    
    plt.xticks(ticks=range(len(epsilons)), labels=[f"{eps:.2f}" for eps in epsilons])
    
    plt.title(f"Target Class Confidence Growth\nAttack: {attack_name}", fontsize=16, pad=15)
    plt.xlabel(r"Perturbation Budget ($L_\infty$ $\epsilon$)", fontsize=14)
    plt.ylabel("Softmax Probability of Target Class", fontsize=14)
    plt.ylim(-0.05, 1.05)
    
    save_or_show(save_flag, save_path)


def plot_source_target_heatmap(matrix_data: np.ndarray, 
                               source_labels: list, 
                               target_labels: list,
                               save_flag: bool = False, 
                               save_path: str = None):
    """
    Genera la matrice di vulnerabilità Source-to-Target (Matrice di Impersonificazione).
    Oscura automaticamente la diagonale (Target == Source).
    """
    plt.figure(figsize=(10, 8))
    
    # Creiamo una maschera per la diagonale
    mask = np.eye(len(matrix_data), dtype=bool)
    
    # Utilizziamo una mappa colori "Reds" (il rosso indica alta vulnerabilità/successo dell'attacco)
    ax = sns.heatmap(matrix_data, mask=mask, annot=True, fmt=".0f", cmap="Reds",
                     xticklabels=target_labels, yticklabels=source_labels,
                     cbar_kws={'label': 'Targeted Success Rate (%)'}, vmin=0, vmax=100)
    
    # Personalizziamo le celle oscurate (la diagonale) colorandole di grigio/nero
    ax.set_facecolor('lightgray')
    
    plt.title("Impersonation Vulnerability Matrix\n(Source ID $\\rightarrow$ Target ID)", fontsize=16, pad=20)
    plt.xlabel("Target Identity (Goal)", fontsize=14, labelpad=15)
    plt.ylabel("Source Identity (Original)", fontsize=14, labelpad=15)
    
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    save_or_show(save_flag, save_path)