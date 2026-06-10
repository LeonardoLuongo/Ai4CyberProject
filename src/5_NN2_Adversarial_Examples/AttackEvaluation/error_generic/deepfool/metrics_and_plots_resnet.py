import os
import pickle

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm

# Utilizziamo PYTHONPATH=src per gli import
from models.resnet import resnet50
from util.google_logger import GoogleSheetLogger
from util.identity_mapper import IdentityMapper
from util.plot.utils_plot_generic import (
    plot_security_evaluation_curves,
    plot_confidence_degradation,
)
from util.plot.utils_plot_shared import (
    plot_adversarial_showcase,
    plot_frequency_spectrum,
)


BATCH_SIZE = 64
IMAGE_SIZE_NN2 = 224
MEAN_BGR = np.array([91.4953, 103.8827, 131.0912], dtype=np.float32)


def load_caffe_weights(model, weights_path):
    """Carica i pesi Caffe convertiti per ResNet50-Scratch."""
    with open(weights_path, "rb") as handle:
        weights = pickle.load(handle, encoding="latin1")

    own_state = model.state_dict()
    for name, param in weights.items():
        if name in own_state:
            own_state[name].copy_(torch.from_numpy(param))


def preprocess_for_resnet(img_bgr_01):
    """Resize 160x160 -> 224x224 e preprocessing Caffe per ResNet50-Scratch."""
    img_bgr_224 = cv2.resize(
        img_bgr_01,
        (IMAGE_SIZE_NN2, IMAGE_SIZE_NN2),
        interpolation=cv2.INTER_LINEAR,
    )
    img_bgr_224 = img_bgr_224.astype(np.float32) * 255.0
    img_bgr_224 -= MEAN_BGR
    return img_bgr_224.transpose(2, 0, 1)


def main():
    print("======================================================")
    print(" METRICHE & PLOT: DEEPFOOL NN1 -> RESNET50-SCRATCH    ")
    print("======================================================\n")

    # =========================================================
    # BLOCCO 0: SETUP E CARICAMENTO CSV
    # =========================================================
    base_dir = Path.cwd()
    tracker_csv_path = base_dir / "dataset" / "attacks" / "NN1" / "error_generic" / "deepfool" / "tracker_deepfool.csv"
    output_eval_dir = base_dir / "plots" / "5_NN2_Adversarial_Examples" / "NN2_ResNet50_Scratch" / "error_generic" / "deepfool"

    progression_dir = output_eval_dir / "visual_progression"

    for d in [output_eval_dir, progression_dir]:
        d.mkdir(parents=True, exist_ok=True)

    if not tracker_csv_path.exists():
        raise FileNotFoundError(
            f"Errore: Tracker CSV non trovato in {tracker_csv_path}. "
            "Esegui prima samples_gen.py di DeepFool."
        )

    df = pd.read_csv(tracker_csv_path)
    original_total_images = len(df)
    print(f"-> Trovate {original_total_images} immagini generate da DeepFool.")

    # Conserviamo le predizioni NN1 presenti nel tracker per selezionare,
    # nel BLOCCO 3, gli stessi campioni mostrati durante la valutazione NN1.
    df["nn1_clean_pred_class"] = df["clean_pred_class"]
    df["nn1_adv_pred_class"] = df["adv_pred_class"]

    # --- INIZIALIZZAZIONE NN2 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"-> Caricamento NN2 ResNet50-Scratch su {device}...")

    nn2 = resnet50(num_classes=8631, include_top=True)
    weights_path = base_dir / "src" / "models" / "resnet50_scratch_weight.pkl"
    if not weights_path.exists():
        raise FileNotFoundError(f"Errore: pesi NN2 mancanti in {weights_path}")

    load_caffe_weights(nn2, weights_path)
    nn2 = nn2.eval().to(device)

    mapper = IdentityMapper(base_dir / "dataset" / "clean" / "splits" / "identity_meta.csv")

    # --- INIZIALIZZAZIONE LOGGER ---
    logger = GoogleSheetLogger()

    # =========================================================
    # BLOCCO 1: INFERENZA NN2 SU CLEAN E ADV
    # =========================================================
    print("\n[BLOCCO 1] Inferenza NN2 sui TIFF DeepFool trasferiti...")

    clean_predictions = []
    adv_predictions = []
    clean_confidences = []
    adv_confidences = []
    true_classes = []

    with torch.inference_mode():
        for start_idx in tqdm(
            range(0, len(df), BATCH_SIZE),
            desc="Inferenza ResNet50-Scratch",
        ):
            batch_df = df.iloc[start_idx : start_idx + BATCH_SIZE]
            x_clean_list = []
            x_adv_list = []
            batch_true_classes = []

            for _, row in batch_df.iterrows():
                clean_path = base_dir / str(row["source_image_path"])
                adv_path = base_dir / str(row["adversarial_image_path"])

                clean_bgr = cv2.imread(str(clean_path), cv2.IMREAD_UNCHANGED)
                adv_bgr = cv2.imread(str(adv_path), cv2.IMREAD_UNCHANGED)

                if clean_bgr is None or adv_bgr is None:
                    raise FileNotFoundError(
                        f"Impossibile leggere clean o adversarial: {clean_path}, {adv_path}"
                    )

                x_clean_list.append(preprocess_for_resnet(clean_bgr))
                x_adv_list.append(preprocess_for_resnet(adv_bgr))
                batch_true_classes.append(
                    mapper.get_facenet_id_by_class_id(str(row["identity_id"]))
                )

            clean_tensor = torch.from_numpy(np.stack(x_clean_list)).to(device)
            adv_tensor = torch.from_numpy(np.stack(x_adv_list)).to(device)

            clean_logits = nn2(clean_tensor)
            adv_logits = nn2(adv_tensor)
            clean_probs = F.softmax(clean_logits, dim=1)
            adv_probs = F.softmax(adv_logits, dim=1)

            clean_preds = torch.argmax(clean_logits, dim=1).cpu().numpy()
            adv_preds = torch.argmax(adv_logits, dim=1).cpu().numpy()
            batch_true_classes = np.asarray(batch_true_classes, dtype=int)

            clean_predictions.extend(clean_preds.tolist())
            adv_predictions.extend(adv_preds.tolist())
            true_classes.extend(batch_true_classes.tolist())
            clean_confidences.extend(
                clean_probs[
                    torch.arange(len(batch_true_classes), device=device),
                    torch.as_tensor(batch_true_classes, device=device),
                ].cpu().numpy().tolist()
            )
            adv_confidences.extend(
                adv_probs[
                    torch.arange(len(batch_true_classes), device=device),
                    torch.as_tensor(batch_true_classes, device=device),
                ].cpu().numpy().tolist()
            )

    # Sovrascriviamo in memoria le predizioni NN1 del tracker con quelle NN2:
    # da qui in poi la struttura rimane uguale al valutatore del punto 3.
    df["clean_pred_class"] = clean_predictions
    df["adv_pred_class"] = adv_predictions
    df["clean_confidence"] = clean_confidences
    df["adv_confidence"] = adv_confidences
    df["true_class"] = true_classes

    # Manteniamo una copia completa per generare showcase confrontabili con NN1.
    df_showcase = df.copy()

    # La trasferibilita si valuta solo sulle immagini che NN2 riconosce da pulite.
    df = df[df["clean_pred_class"] == df["true_class"]].copy()
    total_images = len(df)
    print(
        f"-> Immagini clean riconosciute da NN2: "
        f"{total_images}/{original_total_images}"
    )

    if total_images == 0:
        raise RuntimeError("NN2 non riconosce nessuna immagine clean.")

    # =========================================================
    # BLOCCO 2: GENERAZIONE CURVE DI VALUTAZIONE
    # =========================================================
    print("\n[BLOCCO 2] Generazione Grafici Globali (Robust Accuracy Curve)...")

    epsilons = [0.0, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20]
    asr_dict = {"DeepFool Transfer NN1->ResNet50-Scratch": []}
    confidence_data = []
    previous_eps = 0.0

    # Calcoliamo la robustezza in modo retroattivo, come nel punto 3.
    for eps in epsilons:
        # 1. Fallimenti per sforamento budget
        resisted_budget = df["linf"] > eps

        # 2. Fallimenti matematici: sotto budget ma la classe NN2 non cambia
        within_budget_mask = df["linf"] <= eps
        resisted_attack = within_budget_mask & (
            df["adv_pred_class"] == df["clean_pred_class"]
        )

        total_resisted = len(df[resisted_budget | resisted_attack])
        robust_accuracy = total_resisted / total_images
        successful_transfers = int(
            (
                within_budget_mask
                & (df["adv_pred_class"] != df["clean_pred_class"])
            ).sum()
        )
        newly_admitted_mask = (df["linf"] > previous_eps) & (df["linf"] <= eps)
        newly_transferred = int(
            (
                newly_admitted_mask
                & (df["adv_pred_class"] != df["clean_pred_class"])
            ).sum()
        )

        try:
            logger.log_attack_metrics(
                tester="Andrea",
                attack_type="DeepFool Transfer NN1->ResNet50-Scratch",
                strategy="Untargeted",
                epsilon=eps,
                defense_type="None",
                robust_accuracy=robust_accuracy,
                targeted_asr=0.0,
                untargeted_asr=1.0 - robust_accuracy,
                notes=(
                    f"Sorgente NN1 160x160; target NN2 ResNet50-Scratch resize 224x224; "
                    f"TIFF float32; Caffe BGR mean preprocessing; "
                    f"clean_correct_subset={total_images}/{original_total_images}"
                ),
            )
        except Exception as e:
            print(f"[WARNING] Errore Google Logger: {e}")

        asr_dict["DeepFool Transfer NN1->ResNet50-Scratch"].append(robust_accuracy)

        # Se sfora il budget usiamo la confidenza clean, come nel punto 3.
        confidences = np.where(
            df["linf"] > eps,
            df["clean_confidence"],
            df["adv_confidence"],
        )
        confidence_data.append(confidences)

        print(
            f"eps={eps:.3f} | Robust Accuracy={robust_accuracy*100:.2f}% | "
            f"Untargeted ASR={(1.0-robust_accuracy)*100:.2f}% | "
            f"Within budget={int(within_budget_mask.sum())}/{total_images} | "
            f"Trasferiti={successful_transfers}/{total_images} | "
            f"Nuovi nell'intervallo ({previous_eps:.3f}, {eps:.3f}]="
            f"{int(newly_admitted_mask.sum())}, nuovi trasferiti={newly_transferred}"
        )
        previous_eps = eps

    plot_security_evaluation_curves(
        epsilons,
        asr_dict,
        "NN2 (ResNet50-Scratch)",
        True,
        str(output_eval_dir / "robust_accuracy_curve.png"),
    )
    plot_confidence_degradation(
        epsilons,
        confidence_data,
        "DeepFool Transfer NN1->ResNet50-Scratch",
        True,
        str(output_eval_dir / "confidence_degradation.png"),
    )

    # =========================================================
    # BLOCCO 3: VISUAL SHOWCASE
    # =========================================================
    print("\n[BLOCCO 3] Generazione Visual Showcase per Epsilon...")

    for eps in epsilons:
        if eps == 0.0:
            continue

        # Stessa selezione del file NN1: successo sulla rete sorgente e
        # perturbazione vicina al budget. In questo modo i casi visualizzati
        # restano identici tra punto 3 e punto 5.
        suitable_samples = df_showcase[
            (df_showcase["nn1_adv_pred_class"] != df_showcase["nn1_clean_pred_class"])
            & (df_showcase["linf"] <= eps)
            & (df_showcase["linf"] > (eps - 0.02))
        ]

        if not suitable_samples.empty:
            sample = suitable_samples.iloc[0]

            clean_bgr = cv2.imread(
                str(base_dir / sample["source_image_path"]),
                cv2.IMREAD_UNCHANGED,
            )
            adv_bgr = cv2.imread(
                str(base_dir / sample["adversarial_image_path"]),
                cv2.IMREAD_UNCHANGED,
            )

            if clean_bgr is None or adv_bgr is None:
                continue

            clean_rgb = cv2.cvtColor(clean_bgr, cv2.COLOR_BGR2RGB)
            adv_rgb = cv2.cvtColor(adv_bgr, cv2.COLOR_BGR2RGB)
            eps_str_fmt = f"{eps:.3f}".replace(".", "_")

            plot_adversarial_showcase(
                clean_rgb,
                adv_rgb,
                f"Orig: {sample['identity_name']}",
                f"ResNet Pred: ID {sample['adv_pred_class']}",
                True,
                str(progression_dir / f"showcase_eps_limit_{eps_str_fmt}.png"),
            )
            plot_frequency_spectrum(
                clean_rgb,
                adv_rgb,
                True,
                str(progression_dir / f"spectrum_eps_limit_{eps_str_fmt}.png"),
            )
        else:
            print(
                f" -> [SKIP] Nessun campione rappresentativo trovato vicino "
                f"a L_inf = {eps:.3f}"
            )

    print("\n[OK] Pipeline DeepFool NN1 -> ResNet50-Scratch conclusa!")


if __name__ == "__main__":
    main()
