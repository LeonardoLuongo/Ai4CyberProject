# FILE: src/util/plot/test_plot_generic.py

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
from utils_plot_generic import (
    plot_security_evaluation_curves,
    plot_confidence_degradation,
    plot_transferability_matrix
)

print("Inizio Test: Metriche Error-Generic...\n")

out_dir = "test_output_plots"
os.makedirs(out_dir, exist_ok=True)

# ---------------------------------------------------------
# 1. TEST: Security Evaluation Curve
# ---------------------------------------------------------
print("1. Generazione Security Evaluation Curve...")
epsilons = [0.0, 0.02, 0.05, 0.08, 0.1]
accuracies = {
    "FGSM": [0.98, 0.75, 0.45, 0.20, 0.12],
    "PGD (Iterative)": [0.98, 0.40, 0.05, 0.00, 0.00]
}
plot_security_evaluation_curves(
    epsilons, accuracies, model_name="NN1 (FaceNet)", 
    save_flag=True, save_path=f"{out_dir}/5_generic_sec_curve.png"
)

# ---------------------------------------------------------
# 2. TEST: Confidence Degradation
# ---------------------------------------------------------
print("2. Generazione Confidence Degradation Plot...")
np.random.seed(42)
# Simuliamo 1000 immagini. A eps=0 la confidenza è alta (~0.95)
conf_eps0 = np.clip(np.random.normal(0.95, 0.05, 1000), 0, 1)
# A eps=0.02 inizia a scendere
conf_eps1 = np.clip(np.random.normal(0.70, 0.15, 1000), 0, 1)
# A eps=0.05 crolla
conf_eps2 = np.clip(np.random.normal(0.40, 0.20, 1000), 0, 1)
# A eps=0.1 è quasi nulla
conf_eps3 = np.clip(np.random.normal(0.15, 0.10, 1000), 0, 1)
# A eps=0.1 la rete è distrutta (confidenza bassissima sulla classe corretta)
conf_eps4 = np.clip(np.random.normal(0.05, 0.05, 1000), 0, 1)

confidence_data = [conf_eps0, conf_eps1, conf_eps2, conf_eps3, conf_eps4]

plot_confidence_degradation(
    epsilons, confidence_data, attack_name="FGSM", 
    save_flag=True, save_path=f"{out_dir}/6_generic_confidence.png"
)

# ---------------------------------------------------------
# 3. TEST: Transferability Matrix (Punto 5)
# ---------------------------------------------------------
print("3. Generazione Transferability Matrix...")
# Simuliamo l'Attack Success Rate (ASR) per un epsilon fisso (es. eps=0.05).
# L'ASR (in %) indica quante immagini sono diventate "avversarie".
# Di solito, un attacco su NN1 funziona al 95% su se stesso (NN1), ma scende al 40% su NN2.
transfer_matrix = np.array([
    [95.4, 42.1], # Attacchi generati su NN1 testati su [NN1, NN2]
    [38.5, 91.2]  # Attacchi generati su NN2 testati su [NN1, NN2]
])
models = ["NN1 (FaceNet)", "NN2 (GitHub Model)"]

plot_transferability_matrix(
    transfer_matrix, source_models=models, target_models=models, 
    metric_name="Attack Success Rate (ASR %)",
    save_flag=True, save_path=f"{out_dir}/7_generic_transferability.png"
)

print(f"\nTest Completato! Controlla i nuovi file nella cartella '{out_dir}'.")