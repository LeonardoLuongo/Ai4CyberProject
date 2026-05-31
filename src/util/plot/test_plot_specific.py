# FILE: src/util/plot/test_plot_specific.py

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
from utils_plot_specific import (
    plot_targeted_success_curve,
    plot_target_confidence_growth,
    plot_source_target_heatmap
)

print("Inizio Test: Metriche Error-Specific (Targeted)...\n")

out_dir = "test_output_plots"
os.makedirs(out_dir, exist_ok=True)

# ---------------------------------------------------------
# 1. TEST: Targeted Success Rate Curve
# ---------------------------------------------------------
print("1. Generazione Targeted Success Rate Curve (t-ASR)...")
epsilons = [0.0, 0.02, 0.05, 0.08, 0.1]
# A eps=0 l'ASR è 0% (la rete indovina chi sei). 
# Salendo con epsilon, l'attacco mirato inizia a vincere.
asr_data = {
    "Targeted FGSM": [0.0, 15.0, 45.0, 70.0, 85.0],
    "Targeted PGD (Iterative)": [0.0, 40.0, 95.0, 100.0, 100.0]
}
plot_targeted_success_curve(
    epsilons, asr_data, model_name="NN1 (FaceNet)", 
    save_flag=True, save_path=f"{out_dir}/8_specific_tasr_curve.png"
)

# ---------------------------------------------------------
# 2. TEST: Target Confidence Growth
# ---------------------------------------------------------
print("2. Generazione Target Confidence Growth Plot...")
np.random.seed(42)
# A eps=0, la probabilità che tu sia Brad Pitt è ~0% (0.01)
conf_eps0 = np.clip(np.random.normal(0.01, 0.01, 1000), 0, 1)
# A eps=0.02 la probabilità sale al 20%
conf_eps1 = np.clip(np.random.normal(0.20, 0.10, 1000), 0, 1)
# A eps=0.05 sale al 60%
conf_eps2 = np.clip(np.random.normal(0.60, 0.15, 1000), 0, 1)
# A eps=0.1 è vicina al 95% (Vittoria totale dell'attaccante)
conf_eps3 = np.clip(np.random.normal(0.85, 0.10, 1000), 0, 1)
conf_eps4 = np.clip(np.random.normal(0.98, 0.02, 1000), 0, 1)

target_confidence_data = [conf_eps0, conf_eps1, conf_eps2, conf_eps3, conf_eps4]

plot_target_confidence_growth(
    epsilons, target_confidence_data, attack_name="Targeted PGD", 
    save_flag=True, save_path=f"{out_dir}/9_specific_confidence.png"
)

# ---------------------------------------------------------
# 3. TEST: Source-to-Target Vulnerability Matrix
# ---------------------------------------------------------
print("3. Generazione Impersonation Matrix...")
# Simuliamo un sottoinsieme di 5 identità per leggibilità (es. i primi 5 del tuo CSV)
labels = ["ID_000", "ID_001", "ID_002", "ID_003", "ID_004"]

# Generiamo una matrice di successo casuale (0-100%).
# I valori sulla diagonale verranno automaticamente oscurati dalla nostra funzione!
vuln_matrix = np.random.uniform(10, 95, size=(5, 5))

plot_source_target_heatmap(
    vuln_matrix, source_labels=labels, target_labels=labels,
    save_flag=True, save_path=f"{out_dir}/10_specific_heatmap.png"
)

print(f"\nTest Completato! Controlla i nuovi file nella cartella '{out_dir}'.")