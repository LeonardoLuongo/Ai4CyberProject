import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["NUMBA_NUM_THREADS"] = "1"
os.environ["TORCH_CUDNN_V8_API_ENABLED"] = "0"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps, UnidentifiedImageError
from tqdm import tqdm

torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


# =========================================================================
# RISOLUZIONE ROBUSTA DEI PATH
# =========================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[5]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.senet import senet50
from src.util.google_logger import GoogleSheetLogger
from src.util.identity_mapper import IdentityMapper
from util.plot.utils_plot_generic import (
    plot_confidence_degradation,
    plot_security_evaluation_curves,
    plot_transferability_matrix,
)
from util.plot.utils_plot_shared import (
    plot_adversarial_showcase,
    plot_frequency_spectrum,
)


IMAGE_SIZE_NN2 = 224
MEAN_BGR = np.array([91.4953, 103.8827, 131.0912], dtype=np.float32)


def load_caffe_weights(model, weights_path: Path) -> None:
    with weights_path.open("rb") as handle:
        weights = pickle.load(handle, encoding="latin1")

    own_state = model.state_dict()
    for name, param in weights.items():
        if name in own_state:
            own_state[name].copy_(torch.from_numpy(param))
        else:
            print(f"[WARNING] Chiave inattesa nei pesi SENet: {name}")


def resolve_project_path(base_dir: Path, path_value) -> Path:
    path = Path(str(path_value))
    if path.is_absolute():
        return path
    return base_dir / path


def load_rgb_image(path: Path, image_size: int = IMAGE_SIZE_NN2) -> np.ndarray:
    try:
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            if image.size != (image_size, image_size):
                image = image.resize((image_size, image_size), Image.Resampling.BILINEAR)
            return np.asarray(image, dtype=np.uint8)
    except (OSError, UnidentifiedImageError) as exc:
        raise FileNotFoundError(f"Immagine non leggibile: {path}") from exc


def preprocess_rgb_for_senet(image_rgb: np.ndarray) -> np.ndarray:
    image_bgr = image_rgb[:, :, ::-1].astype(np.float32)
    image_bgr -= MEAN_BGR
    return np.transpose(image_bgr, (2, 0, 1)).astype(np.float32)


def main():
    print("================================================================")
    print(" METRICHE & PLOT: BIM TRANSFERABILITY NN1 -> NN2 (ERROR-GENERIC) ")
    print("================================================================\n")

    base_dir = PROJECT_ROOT
    print(f"-> Project Root impostata a: {base_dir}")

    attacks_dir = (
        base_dir
        / "dataset"
        / "attacks"
        / "NN2"
        / "error_generic"
        / "bim"
    )
    output_eval_dir = (
        base_dir
        / "plots"
        / "5_NN2_Adversarial_Examples"
        / "NN2"
        / "error_generic"
        / "bim"
    )
    progression_dir = output_eval_dir / "visual_progression"

    for directory in [output_eval_dir, progression_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    tracker_files = sorted(attacks_dir.glob("eps_*/tracker_eps_*.csv"))
    if not tracker_files:
        raise FileNotFoundError(
            f"Nessun tracker trovato in {attacks_dir}. "
            "Esegui prima il generatore BIM transferability NN1 -> NN2."
        )

    print(f"-> Trovati {len(tracker_files)} tracker. Unione in corso...")
    df = pd.concat([pd.read_csv(path) for path in tracker_files], ignore_index=True)
    df["eps"] = pd.to_numeric(df["eps"], errors="raise").astype(float)

    meta_csv_path = base_dir / "dataset" / "clean" / "splits" / "identity_meta.csv"
    if "true_facenet_id" not in df.columns:
        mapper = IdentityMapper(meta_csv_path)
        df["true_facenet_id"] = [
            mapper.get_facenet_id_by_class_id(str(identity_id))
            for identity_id in df["identity_id"].values
        ]
    df["true_facenet_id"] = pd.to_numeric(df["true_facenet_id"], errors="raise").astype(int)

    epsilons = sorted(df["eps"].unique())
    print(f"-> Epsilon rilevati: {epsilons}")
    print("-> Le immagini adversarial sono generate da NN1 a 160x160.")
    print("-> In questa valutazione vengono ridimensionate a 224x224 e preprocessate per SENet.\n")

    for column, default in [
        ("clean_pred_class", -1),
        ("adv_pred_class", -1),
        ("clean_true_class_confidence", 0.0),
        ("adv_true_class_confidence", 0.0),
        ("clean_correct_nn2", 0),
        ("adv_correct_nn2", 0),
        ("prediction_changed_nn2", 0),
        ("transfer_success_nn2", 0),
    ]:
        if column not in df.columns:
            df[column] = default

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"-> Inizializzazione NN2 SENet50 su {device}...")

    nn2 = senet50(num_classes=8631, include_top=True)
    weights_path = base_dir / "src" / "models" / "senet50_ft_weight.pkl"
    if not weights_path.exists():
        raise FileNotFoundError(f"Pesi SENet mancanti: {weights_path}")

    load_caffe_weights(nn2, weights_path)
    nn2 = nn2.eval().to(device)
    print("-> Pesi SENet caricati correttamente.")

    try:
        logger = GoogleSheetLogger()
    except Exception as exc:
        logger = None
        print(f"[WARNING] Logger Google non inizializzato: {exc}")

    # =========================================================
    # BLOCCO 1: INFERENZA NN2 SULLE CLEAN E SULLE ADV TRASFERITE
    # =========================================================
    batch_size = 64
    print("\n[BLOCCO 1] Inferenza NN2 su clean e adversarial trasferite...")

    with torch.no_grad():
        for eps in epsilons:
            df_eps = df[df["eps"] == eps]

            for start in tqdm(range(0, len(df_eps), batch_size), desc=f"Inferenza eps={eps:.3f}"):
                batch_df = df_eps.iloc[start : start + batch_size]

                clean_batch = []
                adv_batch = []
                for _, row in batch_df.iterrows():
                    clean_path = resolve_project_path(base_dir, row["source_image_path"])
                    adv_path = resolve_project_path(base_dir, row["adversarial_image_path"])

                    clean_rgb = load_rgb_image(clean_path, IMAGE_SIZE_NN2)
                    adv_rgb = load_rgb_image(adv_path, IMAGE_SIZE_NN2)

                    clean_batch.append(preprocess_rgb_for_senet(clean_rgb))
                    adv_batch.append(preprocess_rgb_for_senet(adv_rgb))

                clean_tensor = torch.from_numpy(np.stack(clean_batch)).to(device)
                adv_tensor = torch.from_numpy(np.stack(adv_batch)).to(device)

                clean_logits = nn2(clean_tensor)
                adv_logits = nn2(adv_tensor)

                clean_probs = F.softmax(clean_logits, dim=1).detach().cpu().numpy()
                adv_probs = F.softmax(adv_logits, dim=1).detach().cpu().numpy()
                clean_preds = torch.argmax(clean_logits, dim=1).detach().cpu().numpy()
                adv_preds = torch.argmax(adv_logits, dim=1).detach().cpu().numpy()
                true_labels = batch_df["true_facenet_id"].to_numpy(dtype=int)

                for j, row_index in enumerate(batch_df.index):
                    true_class = int(true_labels[j])
                    clean_pred = int(clean_preds[j])
                    adv_pred = int(adv_preds[j])

                    clean_correct = int(clean_pred == true_class)
                    adv_correct = int(adv_pred == true_class)
                    prediction_changed = int(adv_pred != clean_pred)
                    transfer_success = int(clean_correct == 1 and adv_correct == 0)

                    df.loc[row_index, "clean_pred_class"] = clean_pred
                    df.loc[row_index, "adv_pred_class"] = adv_pred
                    df.loc[row_index, "clean_true_class_confidence"] = float(clean_probs[j, true_class])
                    df.loc[row_index, "adv_true_class_confidence"] = float(adv_probs[j, true_class])
                    df.loc[row_index, "clean_correct_nn2"] = clean_correct
                    df.loc[row_index, "adv_correct_nn2"] = adv_correct
                    df.loc[row_index, "prediction_changed_nn2"] = prediction_changed
                    df.loc[row_index, "transfer_success_nn2"] = transfer_success

    evaluated_csv_path = output_eval_dir / "bim_transferability_nn1_to_nn2_evaluated.csv"
    df.to_csv(evaluated_csv_path, index=False)
    print(f"-> Master CSV salvato in: {evaluated_csv_path}")

    # =========================================================
    # BLOCCO 2: METRICHE, GRAFICI, LOGGER
    # =========================================================
    print("\n[BLOCCO 2] Calcolo metriche e generazione grafici...")

    robust_accuracy_curve = {"BIM Transfer NN1->NN2": []}
    confidence_data = []
    transfer_asr_values = []
    metric_rows = []

    for eps in epsilons:
        df_eps = df[df["eps"] == eps].copy()
        total = len(df_eps)
        clean_correct_df = df_eps[df_eps["clean_correct_nn2"] == 1]
        clean_correct_total = len(clean_correct_df)

        clean_accuracy = float(df_eps["clean_correct_nn2"].mean()) if total else 0.0
        robust_accuracy_all = float(df_eps["adv_correct_nn2"].mean()) if total else 0.0

        if clean_correct_total > 0:
            robust_accuracy = float(clean_correct_df["adv_correct_nn2"].mean())
            transfer_asr = 1.0 - robust_accuracy
            prediction_change_rate = float(clean_correct_df["prediction_changed_nn2"].mean())
            confidence_values = clean_correct_df["adv_true_class_confidence"].values
        else:
            robust_accuracy = 0.0
            transfer_asr = 0.0
            prediction_change_rate = 0.0
            confidence_values = df_eps["adv_true_class_confidence"].values

        robust_accuracy_curve["BIM Transfer NN1->NN2"].append(robust_accuracy)
        confidence_data.append(confidence_values)
        transfer_asr_values.append(transfer_asr)

        metric_rows.append(
            {
                "eps": eps,
                "total_images": total,
                "nn2_clean_correct_images": clean_correct_total,
                "nn2_clean_accuracy_all": clean_accuracy,
                "nn2_robust_accuracy_clean_correct": robust_accuracy,
                "nn2_robust_accuracy_all": robust_accuracy_all,
                "transfer_asr_clean_correct": transfer_asr,
                "prediction_change_rate_clean_correct": prediction_change_rate,
            }
        )

        print(
            f"eps={eps:.3f} | clean acc NN2={clean_accuracy*100:.2f}% | "
            f"robust acc transfer={robust_accuracy*100:.2f}% | "
            f"transfer ASR={transfer_asr*100:.2f}%"
        )

        if logger is not None:
            logger.log_attack_metrics(
                tester="Andrea",
                attack_type="BIM Transfer NN1->NN2",
                strategy="Error-Generic Untargeted",
                epsilon=eps,
                defense_type="None",
                robust_accuracy=robust_accuracy,
                targeted_asr=0.0,
                untargeted_asr=transfer_asr,
                notes=(
                    f"Sorgente NN1 160x160; target NN2 SENet50 resize 224x224; "
                    f"Caffe BGR mean preprocessing; clean_acc_all={clean_accuracy:.4f}; "
                    f"robust_acc_all={robust_accuracy_all:.4f}; "
                    f"clean_correct_subset={clean_correct_total}/{total}"
                ),
            )

    metrics_csv_path = output_eval_dir / "bim_transferability_nn1_to_nn2_metrics.csv"
    pd.DataFrame(metric_rows).to_csv(metrics_csv_path, index=False)
    print(f"-> CSV metriche salvato in: {metrics_csv_path}")

    plot_security_evaluation_curves(
        epsilons,
        robust_accuracy_curve,
        "Transferability NN1 -> NN2 (SENet50)",
        True,
        str(output_eval_dir / "robust_accuracy_curve_transfer_nn1_to_nn2.png"),
    )
    plot_confidence_degradation(
        epsilons,
        confidence_data,
        "BIM Transfer NN1->NN2",
        True,
        str(output_eval_dir / "confidence_degradation_transfer_nn1_to_nn2.png"),
    )

    pivot_eps = 0.10 if any(abs(float(eps) - 0.10) < 1e-9 for eps in epsilons) else epsilons[-1]
    pivot_index = epsilons.index(pivot_eps)
    transfer_matrix = np.array([[transfer_asr_values[pivot_index] * 100.0]])
    plot_transferability_matrix(
        transfer_matrix,
        source_models=["NN1"],
        target_models=["NN2"],
        metric_name=f"Transfer ASR at eps={pivot_eps:.3f} (%)",
        save_flag=True,
        save_path=str(output_eval_dir / "transferability_matrix_nn1_to_nn2.png"),
    )

    # =========================================================
    # BLOCCO 3: VISUAL SHOWCASE
    # =========================================================
    print("\n[BLOCCO 3] Generazione visual showcase...")

    sample_source_path = df["source_image_path"].iloc[0]
    for eps in epsilons:
        sample_candidates = df[(df["eps"] == eps) & (df["source_image_path"] == sample_source_path)]
        if sample_candidates.empty:
            continue

        sample = sample_candidates.iloc[0]
        clean_rgb = load_rgb_image(resolve_project_path(base_dir, sample["source_image_path"]), IMAGE_SIZE_NN2)
        adv_rgb = load_rgb_image(resolve_project_path(base_dir, sample["adversarial_image_path"]), IMAGE_SIZE_NN2)

        status_text = "RESISTED" if int(sample["adv_correct_nn2"]) == 1 else "TRANSFERRED"
        eps_str = f"{eps:.3f}".replace(".", "_")

        plot_adversarial_showcase(
            clean_rgb,
            adv_rgb,
            f"True ID {int(sample['true_facenet_id'])} | Clean pred {int(sample['clean_pred_class'])}",
            f"Adv pred {int(sample['adv_pred_class'])} ({status_text})",
            True,
            str(progression_dir / f"showcase_eps_{eps_str}.png"),
        )
        plot_frequency_spectrum(
            clean_rgb,
            adv_rgb,
            True,
            str(progression_dir / f"spectrum_eps_{eps_str}.png"),
        )

    print("\n[OK] Valutazione transferability BIM NN1 -> NN2 completata.")


if __name__ == "__main__":
    main()
