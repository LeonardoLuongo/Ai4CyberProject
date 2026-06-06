# FILE: src/util/plot/utils_plot_generic.py

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


def plot_security_evaluation_curves(epsilons: list, 
                                    accuracies_dict: dict, 
                                    model_name: str = "NN1",
                                    save_flag: bool = False, 
                                    save_path: str = None):
    """
    Genera la Security Evaluation Curve ufficiale (Accuratezza vs Epsilon).
    """
    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    colors = sns.color_palette("Set1", n_colors=len(accuracies_dict))
    
    # Check per non far crasciare il moltiplicatore se la lista è vuota
    is_fraction = any(max(acc) <= 1.0 for acc in accuracies_dict.values() if len(acc) > 0)
    multiplier = 100.0 if is_fraction else 1.0

    for idx, (attack_name, acc_list) in enumerate(accuracies_dict.items()):
        if not acc_list: continue # Sicurezza se lista vuota
        
        acc_percent = [a * multiplier for a in acc_list]
        ax.plot(epsilons, acc_percent, marker='o', linewidth=2.5, markersize=8, 
                color=colors[idx], label=attack_name)
                 
        # Aggiunta dei numerini sopra ogni punto
        for x, y in zip(epsilons, acc_percent):
            ax.annotate(f"{y:.1f}%", (x, y), textcoords="offset points", xytext=(0, 10), 
                        ha='center', fontsize=9, fontweight='bold', color=colors[idx])

    # Linea tratteggiata per la baseline
    if list(accuracies_dict.values())[0]:
        baseline_acc = list(accuracies_dict.values())[0][0] * multiplier
        ax.axhline(y=baseline_acc, color='gray', linestyle='--', alpha=0.7, 
                   label=f'Clean Baseline ({baseline_acc:.1f}%)')

    plt.title(f"Security Evaluation Curves - {model_name}\nError-Generic (Untargeted) Attacks", fontsize=16, pad=15)
    plt.xlabel(r"Perturbation Budget ($L_\infty$ $\epsilon$)", fontsize=14)
    plt.ylabel("Robust Accuracy (%)", fontsize=14)
    
    plt.ylim(-5, 115) # Alzato a 115 per non tagliare i numeri in alto
    plt.xlim(min(epsilons) - 0.005, max(epsilons) + 0.005)
    
    # Legenda spostata esternamente a destra per non coprire mai la curva
    plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=12)
    plt.tight_layout()
    
    save_or_show(save_flag, save_path)



def plot_confidence_degradation(epsilons: list, 
                                confidence_data: list, 
                                attack_name: str = "FGSM",
                                save_flag: bool = False, 
                                save_path: str = None):
    """
    Genera un Boxplot che mostra come la confidenza della rete (sulla classe corretta)
    crolli all'aumentare di epsilon, indipendentemente dalla label predetta.
    
    :param confidence_data: Lista di array/liste. Ogni array contiene le confidenze 
                            [0.0, 1.0] per tutte le immagini del test set a quel dato epsilon.
    """
    plt.figure(figsize=(10, 6))
    
    # Seaborn boxplot accetta direttamente una lista di liste
    sns.boxplot(data=confidence_data, palette="YlOrRd")
    
    # Sovrascriviamo le etichette dell'asse X con i valori di Epsilon
    plt.xticks(ticks=range(len(epsilons)), labels=[f"{eps:.4f}" for eps in epsilons])
    
    plt.title(f"Confidence Degradation on True Class\nAttack: {attack_name}", fontsize=16, pad=15)
    plt.xlabel(r"Perturbation Budget ($L_\infty$ $\epsilon$)", fontsize=14)
    plt.ylabel("Softmax Probability of True Class", fontsize=14)
    plt.ylim(-0.05, 1.05)
    
    save_or_show(save_flag, save_path)


def plot_transferability_matrix(matrix_data: np.ndarray, 
                                source_models: list, 
                                target_models: list,
                                metric_name: str = "Attack Success Rate (%)",
                                save_flag: bool = False, 
                                save_path: str = None):
    """
    Genera una Heatmap per valutare la Trasferibilità (Punto 5 della traccia).
    """
    plt.figure(figsize=(8, 6))
    
    # cmap="Reds" perché un Attack Success Rate alto (rosso fuoco) è "cattivo" per il modello
    sns.heatmap(matrix_data, annot=True, fmt=".1f", cmap="Reds", 
                xticklabels=target_models, yticklabels=source_models,
                cbar_kws={'label': metric_name}, vmin=0, vmax=100)
    
    plt.title("Adversarial Transferability Matrix (Error-Generic)", fontsize=16, pad=15)
    plt.xlabel("Target Model (Defending)", fontsize=14, labelpad=10)
    plt.ylabel("Source Model (Attacking)", fontsize=14, labelpad=10)
    
    save_or_show(save_flag, save_path)