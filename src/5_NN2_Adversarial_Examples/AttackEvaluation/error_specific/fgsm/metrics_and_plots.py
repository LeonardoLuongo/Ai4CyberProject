import pickle
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[5]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.senet import senet50
from src.util.google_logger import GoogleSheetLogger
from src.util.identity_mapper import IdentityMapper
from src.util.plot.utils_plot_specific import (
    plot_attack_outcome_distribution,
    plot_target_confidence_growth,
    plot_targeted_success_curve,
    plot_vulnerability_vs_epsilon_heatmap,
    save_or_show,
)


IMAGE_SIZE_NN2 = 224
BATCH_SIZE = 64
STRATEGIES = [
    "next_best",
    "random",
    "rr_lookalikes",
    "rr_extremes",
    "rr_diversity",
    "least-likely",
]
MEAN_BGR = np.array([91.4953, 103.8827, 131.0912], dtype=np.float32)


def load_caffe_weights(model: torch.nn.Module, weights_path: Path) -> None:
    """Load the converted Caffe weights used by SENet50-FT."""
    with weights_path.open("rb") as handle:
        weights = pickle.load(handle, encoding="latin1")

    own_state = model.state_dict()
    for name, param in weights.items():
        if name in own_state:
            own_state[name].copy_(torch.from_numpy(param))
        else:
            print(f"[WARNING] Chiave inattesa nei pesi SENet: {name}")


def resolve_project_path(path_value) -> Path:
    path = Path(str(path_value))
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_rgb_tiff_01(path: Path, image_size: int = IMAGE_SIZE_NN2) -> np.ndarray:
    """Load an NN1 float32 TIFF in [0, 1] and resize it without quantization."""
    image_bgr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image_bgr is None:
        raise FileNotFoundError(f"TIFF non leggibile: {path}")
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError(f"TIFF RGB non valido: {path}, shape={image_bgr.shape}")
    if not np.issubdtype(image_bgr.dtype, np.floating):
        raise TypeError(
            f"Il dataset di trasferibilita deve contenere TIFF float32 [0, 1]: "
            f"{path}, dtype={image_bgr.dtype}"
        )

    image_bgr = image_bgr.astype(np.float32, copy=False)
    if not np.isfinite(image_bgr).all():
        raise ValueError(f"TIFF contenente valori non finiti: {path}")

    min_value = float(image_bgr.min())
    max_value = float(image_bgr.max())
    if min_value < -1e-6 or max_value > 1.0 + 1e-6:
        raise ValueError(
            f"TIFF fuori dall'intervallo [0, 1]: {path}, "
            f"min={min_value:.6f}, max={max_value:.6f}"
        )

    if image_bgr.shape[:2] != (image_size, image_size):
        image_bgr = cv2.resize(
            image_bgr,
            (image_size, image_size),
            interpolation=cv2.INTER_LINEAR,
        )

    return cv2.cvtColor(np.clip(image_bgr, 0.0, 1.0), cv2.COLOR_BGR2RGB)


def preprocess_rgb_for_senet(image_rgb: np.ndarray) -> np.ndarray:
    """Convert RGB [0, 1] to Caffe BGR [0, 255] - mean in CHW format."""
    image_bgr = image_rgb[:, :, ::-1].astype(np.float32) * 255.0
    image_bgr -= MEAN_BGR
    return np.transpose(image_bgr, (2, 0, 1)).astype(np.float32)


def plot_adversarial_showcase(
    clean_rgb: np.ndarray,
    adv_rgb: np.ndarray,
    clean_label: str,
    adv_label: str,
    save_path: Path,
) -> None:
    """Lightweight showcase that does not import the UMAP plotting module."""
    noise = np.abs(adv_rgb.astype(np.float32) - clean_rgb.astype(np.float32))
    noise_visual = np.clip(noise * 30.0, 0.0, 1.0)

    _, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(clean_rgb)
    axes[0].set_title(f"Clean Image\n{clean_label}", color="green")
    axes[1].imshow(noise_visual)
    axes[1].set_title(f"Perturbation (noise x 30)\nMax L_inf: {noise.max():.4f}")
    axes[2].imshow(adv_rgb)
    axes[2].set_title(f"Adversarial Image\n{adv_label}", color="red")
    for axis in axes:
        axis.axis("off")
    plt.suptitle("Targeted Transfer Attack Showcase", fontsize=16)
    save_or_show(True, str(save_path))


def plot_frequency_spectrum(
    clean_rgb: np.ndarray,
    adv_rgb: np.ndarray,
    save_path: Path,
) -> None:
    def magnitude_spectrum(image: np.ndarray) -> np.ndarray:
        gray = np.mean(image, axis=2)
        shifted = np.fft.fftshift(np.fft.fft2(gray))
        return 20.0 * np.log(np.abs(shifted) + 1.0)

    clean_spectrum = magnitude_spectrum(clean_rgb)
    adv_spectrum = magnitude_spectrum(adv_rgb)

    _, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(clean_spectrum, cmap="magma")
    axes[0].set_title("Clean Spectrum")
    axes[1].imshow(adv_spectrum, cmap="magma")
    axes[1].set_title("Adversarial Spectrum")
    axes[2].imshow(np.abs(adv_spectrum - clean_spectrum), cmap="inferno")
    axes[2].set_title("Spectrum Difference")
    for axis in axes:
        axis.axis("off")
    plt.suptitle("Frequency-Domain Comparison", fontsize=16)
    save_or_show(True, str(save_path))


def evaluate_strategy(
    df: pd.DataFrame,
    strategy: str,
    nn2: torch.nn.Module,
    device: torch.device,
) -> pd.DataFrame:
    print(f"\n[BLOCCO 1 - {strategy.upper()}] Inferenza SENet su clean e adversarial...")

    with torch.inference_mode():
        for eps in sorted(df["eps"].unique()):
            df_eps = df[df["eps"] == eps]

            for start in tqdm(
                range(0, len(df_eps), BATCH_SIZE),
                desc=f"{strategy} eps={eps:.3f}",
            ):
                batch_df = df_eps.iloc[start : start + BATCH_SIZE]
                clean_batch = []
                adv_batch = []

                for _, row in batch_df.iterrows():
                    clean_rgb = load_rgb_tiff_01(
                        resolve_project_path(row["source_image_path"])
                    )
                    adv_rgb = load_rgb_tiff_01(
                        resolve_project_path(row["adversarial_image_path"])
                    )
                    clean_batch.append(preprocess_rgb_for_senet(clean_rgb))
                    adv_batch.append(preprocess_rgb_for_senet(adv_rgb))

                clean_tensor = torch.from_numpy(np.stack(clean_batch)).to(device)
                adv_tensor = torch.from_numpy(np.stack(adv_batch)).to(device)

                clean_logits = nn2(clean_tensor)
                adv_logits = nn2(adv_tensor)
                clean_preds = torch.argmax(clean_logits, dim=1).cpu().numpy()
                adv_preds = torch.argmax(adv_logits, dim=1).cpu().numpy()

                true_classes = batch_df["true_facenet_id"].to_numpy(dtype=int)
                target_classes = batch_df["target_class"].to_numpy(dtype=int)
                target_tensor = torch.from_numpy(target_classes).to(device)
                target_confidences = (
                    F.softmax(adv_logits, dim=1)
                    .gather(1, target_tensor.unsqueeze(1))
                    .squeeze(1)
                    .cpu()
                    .numpy()
                )

                clean_correct = clean_preds == true_classes
                adv_correct = adv_preds == true_classes
                target_success = clean_correct & (adv_preds == target_classes)
                untargeted_success = (
                    clean_correct & ~adv_correct & (adv_preds != target_classes)
                )

                indices = batch_df.index
                df.loc[indices, "clean_pred_class"] = clean_preds
                df.loc[indices, "adv_pred_class"] = adv_preds
                df.loc[indices, "adv_target_class_confidence"] = target_confidences
                df.loc[indices, "clean_correct_nn2"] = clean_correct.astype(int)
                df.loc[indices, "adv_correct_nn2"] = adv_correct.astype(int)
                df.loc[indices, "target_success_nn2"] = target_success.astype(int)
                df.loc[indices, "untargeted_success_nn2"] = untargeted_success.astype(
                    int
                )

    integer_columns = [
        "clean_pred_class",
        "adv_pred_class",
        "clean_correct_nn2",
        "adv_correct_nn2",
        "target_success_nn2",
        "untargeted_success_nn2",
    ]
    df[integer_columns] = df[integer_columns].astype(int)
    return df


def create_strategy_plots_and_metrics(
    df: pd.DataFrame,
    strategy: str,
    output_dir: Path,
    logger,
) -> list[dict]:
    print(f"\n[BLOCCO 2 - {strategy.upper()}] Metriche, grafici e logging...")
    epsilons = sorted(df["eps"].unique())
    clean_correct_df = df[df["clean_correct_nn2"] == 1].copy()

    if clean_correct_df.empty:
        print(f"[WARNING] Nessuna clean corretta su SENet per {strategy}.")
        return []

    targeted_curve = {f"FGSM Targeted Transfer ({strategy})": []}
    target_confidence_data = []
    outcome_data = {"Resisted": [], "Untargeted": [], "Targeted": []}
    metric_rows = []

    for eps in epsilons:
        df_eps_all = df[df["eps"] == eps]
        df_eps = clean_correct_df[clean_correct_df["eps"] == eps]
        total = len(df_eps)

        resisted = int(df_eps["adv_correct_nn2"].sum())
        targeted = int(df_eps["target_success_nn2"].sum())
        untargeted = int(df_eps["untargeted_success_nn2"].sum())

        robust_accuracy = resisted / total if total else 0.0
        targeted_asr = targeted / total if total else 0.0
        untargeted_asr = untargeted / total if total else 0.0
        clean_accuracy_all = float(df_eps_all["clean_correct_nn2"].mean())

        # These three outcomes must partition the clean-correct subset.
        if resisted + targeted + untargeted != total:
            raise RuntimeError(
                f"Esiti non mutuamente esclusivi per {strategy}, eps={eps}: "
                f"{resisted}+{targeted}+{untargeted}!={total}"
            )

        targeted_curve[f"FGSM Targeted Transfer ({strategy})"].append(targeted_asr)
        target_confidence_data.append(df_eps["adv_target_class_confidence"].values)
        outcome_data["Resisted"].append(robust_accuracy * 100.0)
        outcome_data["Untargeted"].append(untargeted_asr * 100.0)
        outcome_data["Targeted"].append(targeted_asr * 100.0)

        metric_rows.append(
            {
                "strategy": strategy,
                "eps": eps,
                "total_images": len(df_eps_all),
                "nn2_clean_correct_images": total,
                "nn2_clean_accuracy_all": clean_accuracy_all,
                "robust_accuracy_clean_correct": robust_accuracy,
                "targeted_asr_clean_correct": targeted_asr,
                "untargeted_asr_clean_correct": untargeted_asr,
                "targeted_successes": targeted,
                "untargeted_successes": untargeted,
                "resisted": resisted,
            }
        )

        print(
            f"eps={eps:.3f} | clean correct={total}/{len(df_eps_all)} | "
            f"Robust Accuracy={robust_accuracy*100:.2f}% | "
            f"Targeted ASR={targeted_asr*100:.2f}% | "
            f"Untargeted ASR={untargeted_asr*100:.2f}%"
        )

        if logger is not None:
            logger.log_attack_metrics(
                tester="Andrea",
                attack_type="FGSM Targeted Transfer NN1->NN2",
                strategy=strategy,
                epsilon=eps,
                defense_type="None",
                robust_accuracy=robust_accuracy,
                targeted_asr=targeted_asr,
                untargeted_asr=untargeted_asr,
                notes=(
                    f"Sorgente NN1 160x160; target NN2 SENet50 resize 224x224; "
                    f"TIFF float32; Caffe BGR mean preprocessing; "
                    f"bersaglio preservato da NN1; "
                    f"clean_correct_subset={total}/{len(df_eps_all)}"
                ),
            )

    plot_targeted_success_curve(
        epsilons,
        targeted_curve,
        "NN2 (SENet50-FT)",
        True,
        str(output_dir / "targeted_asr_curve_transfer_nn1_to_nn2.png"),
    )
    plot_target_confidence_growth(
        epsilons,
        target_confidence_data,
        f"FGSM Targeted Transfer ({strategy})",
        True,
        str(output_dir / "target_confidence_transfer_nn1_to_nn2.png"),
    )
    plot_attack_outcome_distribution(
        epsilons,
        outcome_data,
        f"FGSM Targeted Transfer ({strategy})",
        True,
        str(output_dir / "outcome_distribution_transfer_nn1_to_nn2.png"),
    )

    vulnerability = (
        clean_correct_df.assign(success=clean_correct_df["target_success_nn2"])
        .pivot_table(
            index="identity_name",
            columns="eps",
            values="success",
            aggfunc="mean",
            fill_value=0.0,
        )
        * 100.0
    )
    if not vulnerability.empty:
        vulnerability = vulnerability.sort_values(
            by=vulnerability.columns[-1],
            ascending=False,
        ).head(20)
        plot_vulnerability_vs_epsilon_heatmap(
            vulnerability.values,
            [f"{eps:.3f}" for eps in vulnerability.columns],
            vulnerability.index.tolist(),
            f"Top-20 Targeted Transfer Vulnerability ({strategy})",
            True,
            str(output_dir / "targeted_vulnerability_vs_epsilon.png"),
        )

    return metric_rows


def create_showcases(df: pd.DataFrame, strategy: str, progression_dir: Path) -> None:
    print(f"\n[BLOCCO 3 - {strategy.upper()}] Visual showcase...")
    clean_correct_df = df[df["clean_correct_nn2"] == 1]
    if clean_correct_df.empty:
        return

    source_path = clean_correct_df["source_image_path"].iloc[0]
    for eps in sorted(df["eps"].unique()):
        candidates = clean_correct_df[
            (clean_correct_df["eps"] == eps)
            & (clean_correct_df["source_image_path"] == source_path)
        ]
        if candidates.empty:
            continue

        sample = candidates.iloc[0]
        clean_rgb = load_rgb_tiff_01(resolve_project_path(sample["source_image_path"]))
        adv_rgb = load_rgb_tiff_01(resolve_project_path(sample["adversarial_image_path"]))

        if int(sample["target_success_nn2"]) == 1:
            status = "TARGET REACHED"
        elif int(sample["untargeted_success_nn2"]) == 1:
            status = "WRONG NON-TARGET"
        else:
            status = "RESISTED"

        eps_str = f"{eps:.3f}".replace(".", "_")
        plot_adversarial_showcase(
            clean_rgb,
            adv_rgb,
            (
                f"True {int(sample['true_facenet_id'])} | "
                f"Clean pred {int(sample['clean_pred_class'])}"
            ),
            (
                f"Target {int(sample['target_class'])} | "
                f"Adv pred {int(sample['adv_pred_class'])} ({status})"
            ),
            progression_dir / f"showcase_eps_{eps_str}.png",
        )
        plot_frequency_spectrum(
            clean_rgb,
            adv_rgb,
            progression_dir / f"spectrum_eps_{eps_str}.png",
        )


def main() -> None:
    print("====================================================================")
    print(" METRICHE & PLOT: FGSM TARGETED TRANSFERABILITY NN1 -> NN2 SENET50 ")
    print("====================================================================\n")
    print(f"-> Project Root: {PROJECT_ROOT}")

    attacks_root = (
        PROJECT_ROOT
        / "dataset"
        / "attacks"
        / "NN1"
        / "error_specific"
        / "fgsm"
    )
    plots_root = (
        PROJECT_ROOT
        / "plots"
        / "5_NN2_Adversarial_Examples"
        / "NN2"
        / "error_specific"
        / "fgsm"
    )
    plots_root.mkdir(parents=True, exist_ok=True)

    meta_csv = PROJECT_ROOT / "dataset" / "clean" / "splits" / "identity_meta.csv"
    mapper = IdentityMapper(meta_csv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"-> Inizializzazione NN2 SENet50-FT su {device}...")
    nn2 = senet50(num_classes=8631, include_top=True)
    weights_path = PROJECT_ROOT / "src" / "models" / "senet50_ft_weight.pkl"
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

    all_metric_rows = []

    for strategy in STRATEGIES:
        print("\n======================================================")
        print(f" STRATEGIA: {strategy.upper()}")
        print("======================================================")

        strategy_attacks_dir = attacks_root / strategy
        tracker_files = sorted(strategy_attacks_dir.glob("eps_*/tracker_eps_*.csv"))
        if not tracker_files:
            print(f"[WARNING] Nessun tracker trovato in {strategy_attacks_dir}.")
            continue

        df = pd.concat([pd.read_csv(path) for path in tracker_files], ignore_index=True)
        df["eps"] = pd.to_numeric(df["eps"], errors="raise").astype(float)
        df["target_class"] = pd.to_numeric(
            df["target_class"], errors="raise"
        ).astype(int)
        df["true_facenet_id"] = [
            mapper.get_facenet_id_by_class_id(str(identity_id))
            for identity_id in df["identity_id"].values
        ]
        df["true_facenet_id"] = pd.to_numeric(
            df["true_facenet_id"], errors="raise"
        ).astype(int)

        invalid_targets = int((df["target_class"] == df["true_facenet_id"]).sum())
        if invalid_targets:
            raise ValueError(
                f"{strategy}: trovati {invalid_targets} target uguali alla classe vera."
            )

        print(
            f"-> Tracker: {len(tracker_files)} | Righe: {len(df)} | "
            f"Epsilon: {sorted(df['eps'].unique())}"
        )
        print(
            "-> Il target scelto su NN1 viene preservato e verificato "
            "sullo stesso indice VGGFace2 della SENet."
        )

        df = evaluate_strategy(df, strategy, nn2, device)

        output_dir = plots_root / strategy
        progression_dir = output_dir / "visual_progression"
        output_dir.mkdir(parents=True, exist_ok=True)
        progression_dir.mkdir(parents=True, exist_ok=True)

        evaluated_csv = output_dir / f"fgsm_targeted_transfer_evaluated_{strategy}.csv"
        df.to_csv(evaluated_csv, index=False)
        print(f"-> Master CSV salvato in: {evaluated_csv}")

        metric_rows = create_strategy_plots_and_metrics(
            df,
            strategy,
            output_dir,
            logger,
        )
        all_metric_rows.extend(metric_rows)
        pd.DataFrame(metric_rows).to_csv(
            output_dir / f"fgsm_targeted_transfer_metrics_{strategy}.csv",
            index=False,
        )
        create_showcases(df, strategy, progression_dir)

    if all_metric_rows:
        pd.DataFrame(all_metric_rows).to_csv(
            plots_root / "fgsm_targeted_transfer_metrics_all_strategies.csv",
            index=False,
        )

    print("\n[OK] Valutazione targeted FGSM NN1 -> SENet50-FT completata.")


if __name__ == "__main__":
    main()
