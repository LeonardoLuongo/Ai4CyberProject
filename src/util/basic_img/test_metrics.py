import numpy as np
from util.basic_img.metrics import calculate_linf, validate_linf_batch

print("--- Test Validatore Singolo ---")
# Immagine 2x2 finta [0, 255]
clean_img = np.array([[[100, 100, 100], [100, 100, 100]],
                      [[100, 100, 100], [100, 100, 100]]], dtype=np.uint8)

# Creiamo un'immagine avversaria aggiungendo esattamente 25 pixel (25/255 = 0.098)
adv_img = clean_img.copy()
adv_img[0, 0, 1] = 125 # Modifichiamo il canale verde del primo pixel

linf = calculate_linf(clean_img, adv_img)
print(f"L_inf calcolato: {linf:.4f} (Atteso: ~0.0980)")


print("\n--- Test Validatore Batch ---")
# Creiamo un batch di 3 immagini
clean_batch = np.stack([clean_img, clean_img, clean_img])
adv_batch = clean_batch.copy()

# Immagine 0: Modifica permessa (Epsilon ~0.05)
adv_batch[0] = np.clip(clean_batch[0] + 13, 0, 255) 

# Immagine 1: Modifica al limite esatto (Epsilon ~0.10) -> (25.5 / 255 = 0.1)
# Usiamo float per precisione
c_float = clean_batch[1].astype(np.float32) / 255.0
adv_batch_1_float = c_float + 0.10 
adv_batch[1] = (adv_batch_1_float * 255.0).astype(np.uint8)

# Immagine 2: Modifica ILLEGALE (Epsilon ~0.15)
c_float2 = clean_batch[2].astype(np.float32) / 255.0
adv_batch_2_float = c_float2 + 0.15
adv_batch[2] = (adv_batch_2_float * 255.0).astype(np.uint8)

# Validiamo con threshold 0.1
report = validate_linf_batch(clean_batch, adv_batch, threshold=0.1)

for r in report:
    status = "VALIDO (1)" if r['is_valid'] == 1 else "VIOLAZIONE (0)"
    print(f"Sample {r['sample_index']}: Epsilon calcolato = {r['actual_epsilon']:.4f} -> {status}")