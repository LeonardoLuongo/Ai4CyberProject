import os
import pickle

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path

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
    print(" METRICHE & PLOT: C&W NN1 -> RESNET50-SCRATCH         ")
    print("======================================================\n")

    # =========================================================
    # BLOCCO 0: SETUP E CARICAMENTO CSV
    # =========================================================
    base_dir = Path.cwd()
    tracker_csv_path = base_dir / "dataset" / "attacks" / "NN1" / "error_generic" / "cw" / "tracker_cw_untargeted.csv"
    output_eval_dir = base_dir / "plots" / "5_NN2_Adversarial_Examples" / "NN2_ResNet50_Scratch" / "error_generic" / "cw"

    progression_dir = output_eval_dir / "visual_progression"

    for d in [output_eval_dir, progression_dir]:
        d.mkdir(parents=True, exist_ok=True)

    if not tracker_csv_path.exists():
        raise FileNotFoundError(
            f"Errore: Tracker CSV non trovato in {tracker_csv_path}. "
            "Esegui prima samples_gen.py di C&W."
        )

    df = pd.read_csv(tracker_csv_path)
    original_total_images = len(df)
    print(f"-> Trovate {original_total_images} immagini generate da Carlini-Wagner.")

    # Conserviamo le predizioni NN1 del tracker per preservare gli epsilon
    # dinamici e i campioni mostrati durante la valutazione del punto 3.
    df["nn1_clean_pred_class"] = df["clean_pred_class"]
    df["nn1_adv_pred_class"] = df["adv_pred_class"]

    successful_attacks_nn1 = df[
        df["nn1_adv_pred_class"] != df["nn1_clean_pred_class"]
    ]

    if not successful_attacks_nn1.empty:
        percentiles = np.linspace(0, 100, 6)
        epsilons_raw = np.percentile(
            successful_attacks_nn1["linf"],
            percentiles,
        ).tolist()
    else:
        print("Attenzione: nessun attacco NN1 riuscito. Uso epsilon default.")
        epsilons_raw = [0.0, 0.025, 0.05, 0.075, 0.1, 0.15, 0.20]

    epsilons = [0.0] + [round(e, 8) for e in epsilons_raw]
    epsilons += [round(epsilons_raw[-1] + 0.001, 8)]
    epsilons = sorted(list(set(epsilons)))
    print(f"-> Epsilon NN1 preservati (Percentili): {epsilons}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"-> Inizializzazione NN2 ResNet50-Scratch su {device}...")

    nn2 = resnet50(num_classes=8631, include_top=True)
    weights_path = base_dir / "src" / "models" / "resnet50_scratch_weight.pkl"
    if not weights_path.exists():
        raise FileNotFoundError(f"Errore: pesi NN2 mancanti in {weights_path}")
    load_caffe_weights(nn2, weights_path)
    nn2 = nn2.eval().to(device)

    mapper = IdentityMapper(base_dir / "dataset" / "clean" / "splits" / "identity_meta.csv")
    df["true_class"] = [
        mapper.get_facenet_id_by_class_id(str(identity_id))
        for identity_id in df["identity_id"].values
    ]

    # =========================================================
    # BLOCCO 1: INFERENZA BATCH (Valutazione Esito)
    # =========================================================
    print("\n[BLOCCO 1] Inferenza NN2 delle immagini avversarie e originali...")

    with torch.inference_mode():
        for i in tqdm(
            range(0, original_total_images, BATCH_SIZE),
            desc="Inferenza Batch",
        ):
            batch_df = df.iloc[i : i + BATCH_SIZE]
            x_adv_batch, x_clean_batch = [], []

            for _, row in batch_df.iterrows():
                clean_path = base_dir / row["source_image_path"]
                adv_path = base_dir / row["adversarial_image_path"]

                clean_bgr = cv2.imread(str(clean_path), cv2.IMREAD_UNCHANGED)
                adv_bgr = cv2.imread(str(adv_path), cv2.IMREAD_UNCHANGED)

                if clean_bgr is None or adv_bgr is None:
                    raise FileNotFoundError(
                        f"Impossibile leggere clean o adversarial: "
                        f"{clean_path}, {adv_path}"
                    )

                x_clean_batch.append(preprocess_for_resnet(clean_bgr))
                x_adv_batch.append(preprocess_for_resnet(adv_bgr))

            x_clean_tensor = torch.from_numpy(np.stack(x_clean_batch)).to(device)
            x_adv_tensor = torch.from_numpy(np.stack(x_adv_batch)).to(device)

            clean_logits = nn2(x_clean_tensor)
            adv_logits = nn2(x_adv_tensor)

            clean_preds = torch.argmax(clean_logits, dim=1).cpu().numpy()
            adv_preds = torch.argmax(adv_logits, dim=1).cpu().numpy()
            clean_probs = F.softmax(clean_logits, dim=1).cpu().numpy()
            adv_probs = F.softmax(adv_logits, dim=1).cpu().numpy()

            for j in range(len(adv_preds)):
                true_class = int(batch_df["true_class"].iloc[j])
                original_idx = batch_df.index[j]

                df.loc[original_idx, "clean_pred_class"] = int(clean_preds[j])
                df.loc[original_idx, "adv_pred_class"] = int(adv_preds[j])
                df.loc[original_idx, "clean_confidence"] = clean_probs[j, true_class]
                df.loc[original_idx, "adv_confidence"] = adv_probs[j, true_class]

    # Copia completa usata per preservare gli stessi showcase del punto 3.
    df_showcase = df.copy()

    # La trasferibilita si valuta sulle sole immagini clean riconosciute da NN2.
    df = df[df["clean_pred_class"] == df["true_class"]].copy()
    total_images = len(df)
    print(f"-> Immagini clean riconosciute da NN2: {total_images}/{original_total_images}")

    if total_images == 0:
        raise RuntimeError("NN2 non riconosce nessuna immagine clean.")

    evaluated_csv_path = output_eval_dir / "cw_transferability_nn1_to_nn2_evaluated.csv"
    df.to_csv(evaluated_csv_path, index=False)
    print(f"-> Master Data salvato in {evaluated_csv_path}")

    # =========================================================
    # BLOCCO 2: GENERAZIONE CURVE DI VALUTAZIONE E LOGGING
    # =========================================================
    print("\n[BLOCCO 2] Generazione Grafici Globali e Logging...")

    asr_dict = {"C&W Transfer NN1->ResNet50-Scratch": []}
    confidence_data = []

    logger = GoogleSheetLogger()
    previous_eps = 0.0

    for eps in epsilons:
        resisted_budget = df["linf"] > eps
        within_budget_mask = df["linf"] <= eps
        resisted_attack = within_budget_mask & (
            df["adv_pred_class"] == df["clean_pred_class"]
        )

        total_resisted = len(df[resisted_budget | resisted_attack])
        robust_accuracy = total_resisted / total_images

        success_mask = within_budget_mask & (
            df["adv_pred_class"] != df["clean_pred_class"]
        )
        total_untargeted_success = int(success_mask.sum())
        untargeted_asr = total_untargeted_success / total_images

        newly_admitted_mask = (df["linf"] > previous_eps) & (df["linf"] <= eps)
        newly_transferred = int(
            (
                newly_admitted_mask
                & (df["adv_pred_class"] != df["clean_pred_class"])
            ).sum()
        )

        asr_dict["C&W Transfer NN1->ResNet50-Scratch"].append(robust_accuracy)

        confidences = np.where(
            df["linf"] > eps,
            df["clean_confidence"],
            df["adv_confidence"],
        )
        confidence_data.append(confidences)

        if hasattr(logger, "log_attack_metrics"):
            logger.log_attack_metrics(
                tester="Andrea",
                attack_type="C&W Transfer NN1->ResNet50-Scratch",
                strategy="Untargeted",
                epsilon=eps,
                defense_type="None",
                robust_accuracy=robust_accuracy,
                targeted_asr=0.0,
                untargeted_asr=untargeted_asr,
                notes=(
                    f"Sorgente NN1 160x160; target NN2 ResNet50-Scratch resize 224x224; "
                    f"TIFF float32; Caffe BGR mean preprocessing; "
                    f"clean_correct_subset={total_images}/{original_total_images}"
                ),
            )

        print(
            f"eps={eps:.8f} | Robust Accuracy={robust_accuracy*100:.2f}% | "
            f"Untargeted ASR={untargeted_asr*100:.2f}% | "
            f"Within budget={int(within_budget_mask.sum())}/{total_images} | "
            f"Trasferiti={total_untargeted_success}/{total_images} | "
            f"Nuovi trasferiti={newly_transferred}"
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
        "C&W Transfer NN1->ResNet50-Scratch",
        True,
        str(output_eval_dir / "confidence_degradation.png"),
    )

    # =========================================================
    # BLOCCO 3: VISUAL SHOWCASE
    # =========================================================
    print("\n[BLOCCO 3] Generazione Visual Showcase per Epsilon...")

    for i, eps in enumerate(epsilons):
        if eps == 0.0:
            continue

        lower_bound = epsilons[i - 1]

        # Usiamo i successi NN1 per mostrare gli stessi campioni del punto 3.
        suitable_samples = df_showcase[
            (df_showcase["nn1_adv_pred_class"] != df_showcase["nn1_clean_pred_class"])
            & (df_showcase["linf"] <= eps)
            & (df_showcase["linf"] > lower_bound)
        ]

        if not suitable_samples.empty:
            sample = suitable_samples.sort_values(
                by="linf",
                ascending=False,
            ).iloc[0]

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
            eps_str_fmt = f"{eps:.4f}".replace(".", "_")

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
                f" -> [SKIP] Nessun campione rappresentativo C&W trovato "
                f"tra {lower_bound:.8f} e {eps:.8f}"
            )

    print("\n[OK] Pipeline C&W NN1 -> ResNet50-Scratch conclusa con successo!")


if __name__ == "__main__":
    main()
