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

    ax = plt.gca()

    for idx, (attack_name, asr_list) in enumerate(asr_dict.items()):
        asr_percent = [a * multiplier for a in asr_list]
        plt.plot(epsilons, asr_percent, marker='s', linewidth=2.5, markersize=8, 
                 color=colors[idx], label=attack_name)
        
        # Aggiunta dei numerini sopra ogni punto
        for x, y in zip(epsilons, asr_percent):
            # Posizioniamo il testo leggermente sopra e a sinistra per non coprire il marker
            ax.annotate(f"{y:.1f}%", (x, y), textcoords="offset points", xytext=(0, 10), 
                        ha='center', fontsize=9, fontweight='bold', color=colors[idx])

    plt.title(f"Targeted Attack Success Rate (t-ASR) - {model_name}\nError-Specific Attacks", fontsize=16, pad=15)
    plt.xlabel(r"Perturbation Budget ($L_\infty$ $\epsilon$)", fontsize=14)
    plt.ylabel("Targeted Success Rate (%)", fontsize=14)
    
    # Asse Y da 0 a 100%
    plt.ylim(-5, 115)
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
    
    plt.xticks(ticks=range(len(epsilons)), labels=[f"{eps:.4f}" for eps in epsilons])
    
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
    
    ax = sns.heatmap(matrix_data, annot=True, fmt=".0f", cmap="Reds",
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


def plot_attack_outcome_distribution(epsilons: list, 
                                     outcome_data: dict, 
                                     attack_name: str = "Targeted Attack",
                                     save_flag: bool = False, 
                                     save_path: str = None):
    """
    Genera uno Stacked Bar Chart al 100% che mostra l'esito dell'attacco:
    - Rete Resiste (Verde)
    - Rete Ingannata ma Bersaglio Mancato (Giallo)
    - Bersaglio Raggiunto (Rosso)
    """
    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    
    # Estraiamo le liste (devono essere percentuali che sommano a 100 per ogni eps)
    resisted = np.array(outcome_data["Resisted"])
    untargeted = np.array(outcome_data["Untargeted"])
    targeted = np.array(outcome_data["Targeted"])
    
    # Creiamo le barre impilate
    bar_width = 0.5 if len(epsilons) < 10 else 0.8
    x_pos = np.arange(len(epsilons))
    
    # Plottiamo le barre e conserviamo l'oggetto "BarContainer" per estrarre le coordinate
    b1 = plt.bar(x_pos, resisted, color='forestgreen', edgecolor='white', width=bar_width, label='Model Resisted')
    b2 = plt.bar(x_pos, untargeted, bottom=resisted, color='gold', edgecolor='white', width=bar_width, label='Untargeted Success')
    b3 = plt.bar(x_pos, targeted, bottom=resisted+untargeted, color='firebrick', edgecolor='white', width=bar_width, label='Targeted Success')
    
    # Funzione interna per stampare il testo al centro del segmento
    def add_bar_labels(bars):
        for bar in bars:
            height = bar.get_height()
            if height > 2.0: # Stampiamo il numero solo se il segmento è abbastanza grande da contenerlo
                # Troviamo il centro verticale del segmento: Coordinata Y di base + mezza altezza
                y_center = bar.get_y() + (height / 2)
                ax.text(bar.get_x() + bar.get_width()/2, y_center, f"{height:.1f}%", 
                        ha='center', va='center', color='white' if bar.get_facecolor() != (1.0, 0.843, 0.0, 1.0) else 'black', # Nero sul giallo
                        fontweight='bold', fontsize=9)

    add_bar_labels(b1)
    add_bar_labels(b2)
    add_bar_labels(b3)
    
    plt.xticks(x_pos, [f"{eps:.3f}" for eps in epsilons])
    plt.title(f"Attack Outcome Distribution - {attack_name}", fontsize=16, pad=15)
    plt.xlabel(r"Perturbation Budget ($L_\infty$ $\epsilon$)", fontsize=14)
    plt.ylabel("Percentage of Test Set (%)", fontsize=14)
    plt.ylim(0, 105)
    
    # Spostiamo la legenda fuori per non coprire i dati
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12)
    plt.tight_layout() # Adatta i margini per la legenda esterna
    
    save_or_show(save_flag, save_path)


def plot_vulnerability_vs_epsilon_heatmap(matrix_data: np.ndarray, 
                                          epsilons_labels: list, 
                                          identity_labels: list,
                                          title: str,
                                          save_flag: bool = False, 
                                          save_path: str = None):
    """
    Genera una heatmap che mostra l'evoluzione dell'ASR per singole identità
    all'aumentare di Epsilon (Sulle righe le Identità, sulle colonne gli Epsilon).
    """
    plt.figure(figsize=(10, 8))
    
    sns.heatmap(matrix_data, annot=True, fmt=".0f", cmap="Reds",
                xticklabels=epsilons_labels, yticklabels=identity_labels,
                cbar_kws={'label': 'Targeted Success Rate (%)'}, vmin=0, vmax=100)
    
    plt.title(title, fontsize=16, pad=20)
    plt.xlabel("Perturbation Budget (Epsilon)", fontsize=14, labelpad=15)
    plt.ylabel("Source Identity", fontsize=14, labelpad=15)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    save_or_show(save_flag, save_path)
