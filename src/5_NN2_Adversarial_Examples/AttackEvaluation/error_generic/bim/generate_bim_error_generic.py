import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image, ImageOps, UnidentifiedImageError
from tqdm import tqdm


# =========================================================================
# RISOLUZIONE ROBUSTA DEI PATH
# =========================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[5]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from art.attacks.evasion import BasicIterativeMethod
from art.estimators.classification import PyTorchClassifier
from facenet_pytorch import InceptionResnetV1, MTCNN

from src.util.attack_error_specific_utils import get_one_hot_target
from src.util.basic_img.metrics import calculate_linf
from src.util.identity_mapper import IdentityMapper


IMAGE_SIZE_NN1 = 160
IMAGE_SIZE_NN2 = 224


def load_rgb_chw_01(image_path: Path, image_size: int) -> np.ndarray:
    try:
        with Image.open(image_path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            if image.size != (image_size, image_size):
                image = image.resize((image_size, image_size), Image.Resampling.BILINEAR)
            array = np.asarray(image, dtype=np.float32) / 255.0
    except (OSError, UnidentifiedImageError) as exc:
        raise RuntimeError(f"Immagine non leggibile: {image_path}") from exc

    return np.transpose(np.clip(array, 0.0, 1.0), (2, 0, 1)).astype(np.float32)


def save_rgb_hwc_01(image_path: Path, image_hwc_01: np.ndarray) -> None:
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_uint8 = np.rint(np.clip(image_hwc_01, 0.0, 1.0) * 255.0).astype(np.uint8)
    Image.fromarray(image_uint8, mode="RGB").save(image_path)


def main():
    print("==============================================================")
    print(" GENERATORE TRANSFERABILITY: BIM ERROR-GENERIC NN1 -> NN2     ")
    print("==============================================================\n")

    base_dir = PROJECT_ROOT
    print(f"-> Project Root impostata a: {base_dir}")
    print("-> L'attacco BIM viene generato su NN1 a 160x160.")
    print("-> NN2 NON viene usata qui: la resize a 224x224 avverra nel file di valutazione.\n")

    csv_path = base_dir / "dataset" / "clean" / "splits" / "manifest.csv"
    meta_csv_path = base_dir / "dataset" / "clean" / "splits" / "identity_meta.csv"

    cropped_nn1_dir = base_dir / "dataset" / "clean_cropped" / "NN1"
    output_base_dir = (
        base_dir
        / "dataset"
        / "attacks"
        / "NN2"
        / "error_generic"
        / "bim"
    )

    epsilons = [0.025, 0.050, 0.075, 0.100, 0.15, 0.20]
    bim_max_iter = 4
    bim_eps_step_divisor = 24.0
    batch_size = 64

    if not csv_path.exists() or not meta_csv_path.exists():
        raise FileNotFoundError(f"manifest.csv o identity_meta.csv mancanti in {base_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"-> Inizializzazione NN1 su {device}...")

    mapper = IdentityMapper(meta_csv_path)
    mtcnn = MTCNN(
        image_size=IMAGE_SIZE_NN1,
        margin=0,
        keep_all=True,
        post_process=True,
        device=device,
    )

    nn1 = InceptionResnetV1(pretrained="vggface2", classify=True).eval().to(device)

    classifier = PyTorchClassifier(
        model=nn1,
        clip_values=(0.0, 1.0),
        loss=nn.CrossEntropyLoss(),
        optimizer=None,
        input_shape=(3, IMAGE_SIZE_NN1, IMAGE_SIZE_NN1),
        nb_classes=mapper.get_num_training_classes(),
        preprocessing=(0.5, 0.5),
        device_type="gpu" if torch.cuda.is_available() else "cpu",
    )

    df_clean = pd.read_csv(csv_path)
    print(f"-> Trovate {len(df_clean)} immagini nel manifest.")

    # =========================================================================
    # FASE 1: TEST SET VALIDATO DA NN1
    # =========================================================================
    print("\n[FASE 1] Caricamento/generazione crop NN1 validati a 160x160...")
    valid_records = []

    with torch.no_grad():
        for _, row in tqdm(df_clean.iterrows(), total=len(df_clean), desc="Validazione NN1"):
            class_id = str(row["identity_id"])
            true_facenet_id = mapper.get_facenet_id_by_class_id(class_id)
            if true_facenet_id == -1:
                continue

            source_img_path = base_dir / str(row["image_path"])
            identity_dir_name = source_img_path.parent.name
            img_filename = source_img_path.name
            crop_path = cropped_nn1_dir / identity_dir_name / img_filename

            if crop_path.exists():
                try:
                    x_clean_chw = load_rgb_chw_01(crop_path, IMAGE_SIZE_NN1)
                except RuntimeError:
                    continue
            else:
                try:
                    with Image.open(source_img_path) as image:
                        img_pil = ImageOps.exif_transpose(image).convert("RGB")
                except (OSError, UnidentifiedImageError):
                    continue

                faces = mtcnn(img_pil)
                if faces is None:
                    continue

                faces = faces.to(device)
                logits_all = nn1(faces)
                preds_all = torch.argmax(logits_all, dim=1).detach().cpu().numpy()
                if true_facenet_id not in preds_all:
                    continue

                match_idx = np.where(preds_all == true_facenet_id)[0][0]
                best_face = faces[match_idx].detach().cpu().numpy()
                x_clean_chw = np.clip((best_face + 1.0) / 2.0, 0.0, 1.0).astype(np.float32)

                save_rgb_hwc_01(crop_path, np.transpose(x_clean_chw, (1, 2, 0)))

            row_dict = row.to_dict()
            row_dict["true_facenet_id"] = true_facenet_id
            row_dict["x_clean"] = np.expand_dims(x_clean_chw, axis=0)
            row_dict["cropped_image_path"] = crop_path.relative_to(base_dir).as_posix()
            valid_records.append(row_dict)

    total_valid = len(valid_records)
    print(f"-> Immagini valide per trasferibilita NN1 -> NN2: {total_valid}")
    if total_valid == 0:
        raise RuntimeError("Nessuna immagine valida trovata per generare gli attacchi.")

    # =========================================================================
    # FASE 2: BIM SU NN1, SALVATAGGIO A 160x160
    # =========================================================================
    print("\n==============================================================")
    print(" AVVIO GENERAZIONE BIM ERROR-GENERIC SU NN1")
    print("==============================================================")

    for eps in epsilons:
        eps_str = f"eps_{eps:.3f}".replace(".", "_")
        eps_dir = output_base_dir / eps_str
        eps_dir.mkdir(parents=True, exist_ok=True)

        eps_step = eps / bim_eps_step_divisor
        print(
            f"\n[>>>] Generazione BIM NN1, epsilon={eps:.3f} | "
            f"eps_step={eps_step:.6f} | max_iter={bim_max_iter}"
        )

        attack = BasicIterativeMethod(
            estimator=classifier,
            eps=eps,
            eps_step=eps_step,
            max_iter=bim_max_iter,
            targeted=False,
            batch_size=batch_size,
        )

        eps_tracker_records = []

        for start_idx in tqdm(range(0, total_valid, batch_size), desc=f"Batch {eps_str}"):
            end_idx = min(start_idx + batch_size, total_valid)
            batch_records = valid_records[start_idx:end_idx]

            batch_x = np.stack([row["x_clean"][0] for row in batch_records]).astype(np.float32)
            batch_y = np.stack(
                [
                    get_one_hot_target(
                        row["true_facenet_id"],
                        num_classes=mapper.get_num_training_classes(),
                    )[0]
                    for row in batch_records
                ]
            ).astype(np.float32)

            x_adv_batch = attack.generate(x=batch_x, y=batch_y).astype(np.float32)
            x_adv_batch = np.clip(x_adv_batch, 0.0, 1.0)

            for i, row in enumerate(batch_records):
                x_clean_single = batch_x[i]
                x_adv_single = x_adv_batch[i]

                clean_hwc = np.transpose(x_clean_single, (1, 2, 0))
                adv_hwc = np.transpose(x_adv_single, (1, 2, 0))

                actual_linf = calculate_linf(clean_hwc, adv_hwc)
                mean_abs_perturbation = float(np.mean(np.abs(adv_hwc - clean_hwc)))

                source_img_path = base_dir / str(row["image_path"])
                identity_dir_name = source_img_path.parent.name
                orig_filename = source_img_path.stem

                out_img_dir = eps_dir / identity_dir_name
                adv_save_path = out_img_dir / f"{orig_filename}.png"
                save_rgb_hwc_01(adv_save_path, adv_hwc)

                eps_tracker_records.append(
                    {
                        "attack_type": "bim",
                        "eps": eps,
                        "eps_step": eps_step,
                        "max_iter": bim_max_iter,
                        "targeted": False,
                        "target_strategy": "none",
                        "target_class": -1,
                        "source_model": "NN1_InceptionResnetV1_vggface2",
                        "target_model": "NN2_SENet50",
                        "transferability_setting": "NN1_to_NN2",
                        "generated_image_size": IMAGE_SIZE_NN1,
                        "nn2_eval_image_size": IMAGE_SIZE_NN2,
                        "dataset_label": row["dataset_label"],
                        "identity_id": row["identity_id"],
                        "identity_name": row["identity_name"],
                        "identity_dir": identity_dir_name,
                        "true_facenet_id": row["true_facenet_id"],
                        "source_image_path": row["cropped_image_path"],
                        "adversarial_image_path": adv_save_path.relative_to(base_dir).as_posix(),
                        "linf": round(actual_linf, 6),
                        "mean_abs_perturbation": round(mean_abs_perturbation, 6),
                    }
                )

        tracker_path = eps_dir / f"tracker_{eps_str}.csv"
        pd.DataFrame(eps_tracker_records).to_csv(tracker_path, index=False)
        print(f"-> Tracker salvato in: {tracker_path}")

    print("\n[OK] Generazione BIM transferability NN1 -> NN2 completata.")
    print(f"-> Output: {output_base_dir}")
    print("-> Nel file di valutazione NN2 carica queste immagini, ridimensionale a 224x224 e applica il preprocessing SENet.")


if __name__ == "__main__":
    main()
