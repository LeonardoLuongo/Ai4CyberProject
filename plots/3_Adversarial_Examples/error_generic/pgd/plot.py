import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.ticker import PercentFormatter

# I dati originali
results_accuracy = {
    0.001: {1.0: [0.8873, 0.8873, 0.8873], 1.5: [0.8567, 0.8567, 0.8567], 2.5: [0.8425, 0.8403, 0.8392]},
    0.005: {1.0: [0.3403, 0.3348, 0.3359], 1.5: [0.1969, 0.1937, 0.1893], 2.5: [0.1389, 0.1357, 0.1357]},
    0.01:  {1.0: [0.0317, 0.0306, 0.0317], 1.5: [0.0109, 0.0066, 0.0077], 2.5: [0.0022, 0.0011, 0.0011]},
    0.015: {1.0: [0.0022, 0.0022, 0.0022], 1.5: [0.0011, 0.0000, 0.0000], 2.5: [0.0000, 0.0000, 0.0000]}
}

# Parametri
total_valid = 914
BEST_MAX_ITER = 4
num_init_list = [1, 3, 5]

# Selezioniamo SOLO gli epsilon significativi
epsilons_to_plot = [0.001, 0.005, 0.01, 0.015]
mults = [1.0, 1.5, 2.5]
markers = ['o', 's', '^']
colors = ['tab:blue', 'tab:orange', 'tab:green']

# Creiamo una griglia 2x2
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for i, eps in enumerate(epsilons_to_plot):
    ax = axes[i]
    
    for j, mult in enumerate(mults):
        ax.plot(
            num_init_list, 
            results_accuracy[eps][mult], 
            marker=markers[j],
            color=colors[j],
            linestyle='-', 
            linewidth=2.5,
            label=f'Step mult = {mult}'
        )
        
    ax.set_title(f'$\epsilon$ = {eps}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Num Random Init', fontsize=12)
    ax.set_ylabel('Robust Accuracy', fontsize=12)
    ax.set_xticks(num_init_list)
    ax.grid(True, linestyle='--', alpha=0.7)

    # ====================================================================
    # LOGICA DI ZOOM DINAMICA (Senza scritte e con formattazione % sull'asse)
    # ====================================================================
    if eps == 0.01:
        # Zoom fino al 3.5%
        ax.set_ylim([-0.001, 0.035])
    elif eps == 0.015:
        # Zoom estremo fino allo 0.25%
        ax.set_ylim([-0.0001, 0.0025])
    else:
        # Default (0-100%) per 0.001 e 0.005
        ax.set_ylim([-0.05, 1.05])
        
    # Trasforma i valori decimali dell'asse Y in percentuale visiva (es. 0.03 -> 3%)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=2 if eps==0.015 else None))
    # ====================================================================

# Mettiamo una SINGOLA legenda globale
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.94), 
           ncol=3, fontsize=12, title="Step Multiplier (\u03B1)", title_fontsize=12)

# Titolo principale
plt.suptitle(f'PGD Hyperparameter Tuning by Epsilon\n(Evaluated on {total_valid} valid crops | max_iter={BEST_MAX_ITER})', 
             fontsize=18, fontweight='bold', y=1.02)

# Salvataggio
plots_dir = Path("plots/3_Adversarial_Examples/error_generic/pgd")
plots_dir.mkdir(parents=True, exist_ok=True)
save_path = plots_dir / "pgd_hyperparameter_tuning_subplots_zoomed.png"

plt.tight_layout(rect=[0, 0, 1, 0.88]) 
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"✅ Grafico a subplots con zoom dinamico salvato in: {save_path}")
plt.show()