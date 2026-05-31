import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

print("1. Importazione librerie...")
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print("2. Creazione dati fake...")
img = np.random.rand(224, 224, 3)

print("3. Inizializzazione Matplotlib...")
try:
    fig, axes = plt.subplots(1, 1)
    print("4. Subplot creato con successo.")
    
    axes.imshow(img)
    print("5. Immagine inserita nel grafico.")
    
    path = "test_debug.png"
    plt.savefig(path)
    print(f"6. SALVATAGGIO RIUSCITO IN: {path}")
except Exception as e:
    print(f"ERRORE TROVATO: {e}")