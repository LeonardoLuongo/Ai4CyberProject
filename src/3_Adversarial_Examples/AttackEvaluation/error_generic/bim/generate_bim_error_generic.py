"""Generate error-generic BIM adversarial images for NN1.

This attack uses the Basic Iterative Method (BIM) to make NN1 misclassify 
each input image without forcing a specific target class. 
It saves adversarial PNG images and a manifest CSV.
"""

from __future__ import annotations
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[5]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.util.attack_common import (
    batched, 
    build_nn1_art_classifier,
    default_output_dir,
    discover_test_images,
    ensure_clean_output_dir,
    load_batch_for_nn1,
    output_path_for_sample,
    perturbation_stats,
    progress_line,
    save_adv_image,
    validate_eps,
    write_json,
    write_manifest,
)


def main() -> int:
    # =========================================================================
    # PARAMETRI DI CONFIGURAZIONE
    # =========================================================================
    input_dir = Path("dataset/clean/test")
    output_dir = None       
    
    # Parametri specifici dell'attacco
    eps = 0.025             
    eps_step = 0.005        # Dimensione del passo (Step size)
    max_iter = 10           # Numero massimo di iterazioni
    
    batch_size = 64          
    image_size = 160        
    max_images = None       
    overwrite = True        
    # =========================================================================

    validate_eps(eps)

    if output_dir is None:
        base_dir = Path("dataset/attacks/error_generic")
        # Genererà ad esempio: dataset/attacks/error_generic/bim/eps_0_025
        output_dir = default_output_dir(base_dir, "bim", eps)
    
    ensure_clean_output_dir(output_dir, overwrite)

    samples = discover_test_images(input_dir)
    if max_images is not None:
        samples = samples[:max_images]
    print(f"Found {len(samples)} test images.")

    print("Loading NN1 and wrapping it with ART PyTorchClassifier...")
    classifier, num_classes, device_type = build_nn1_art_classifier()

    # Importiamo BIM (Basic Iterative Method) invece di FGSM
    from art.attacks.evasion import BasicIterativeMethod

    attack = BasicIterativeMethod(
        estimator=classifier, 
        eps=eps, 
        eps_step=eps_step, 
        max_iter=max_iter, 
        targeted=False
    )

    rows: list[dict[str, object]] = []
    done = 0
    for batch_index, batch_samples in enumerate(batched(samples, batch_size), start=1):
        x = load_batch_for_nn1(batch_samples, image_size)
        x_adv = attack.generate(x=x)

        for sample, original, adversarial in zip(batch_samples, x, x_adv):
            adv_path = output_path_for_sample(output_dir, sample)
            
            save_adv_image(adversarial, adv_path)
            
            linf, mean_abs = perturbation_stats(original, adversarial)
            
            rel_source_path = f"dataset/clean/test/{sample.relative_path.as_posix()}"
            rel_adv_path = adv_path.as_posix()
            
            # STRUTTURA DEL MANIFEST AGGIORNATA CON EPS_STEP E MAX_ITER
            rows.append(
                {
                    "attack_type": "bim",
                    "eps": eps,
                    "eps_step": eps_step,
                    "max_iter": max_iter,
                    "targeted": False,
                    "target_strategy": -1,
                    "target_class": -1,
                    "dataset_label": sample.dataset_label if sample.dataset_label is not None else -1,
                    "identity_id": sample.identity_id if sample.identity_id else -1,
                    "identity_name": sample.identity_name if sample.identity_name else -1,
                    "identity_dir": sample.identity_dir if sample.identity_dir else -1,
                    "source_image_path": rel_source_path,
                    "adversarial_image_path": rel_adv_path,
                    "linf": round(linf, 6),
                    "mean_abs_perturbation": round(mean_abs, 6)
                }
            )

        done += len(batch_samples)
        print(progress_line(done, len(samples), batch_index))

    write_manifest(output_dir / "manifest.csv", rows)
    
    write_json(
        output_dir / "config.json",
        {
            "attack_type": "bim",
            "classifier": "facenet_pytorch.InceptionResnetV1(pretrained='vggface2', classify=True)",
            "art_estimator": "PyTorchClassifier",
            "eps": eps,
            "eps_step": eps_step,
            "max_iter": max_iter,
            "max_allowed_eps": 0.1,
            "targeted": False,
            "input_dir": input_dir.as_posix(),
            "output_dir": output_dir.as_posix(),
            "num_images": len(samples),
            "max_images": max_images,
            "batch_size": batch_size,
            "image_size": image_size,
            "num_classes": num_classes,
            "device_type": device_type,
            "saved_format": "PNG",
        },
    )

    print(f"Saved adversarial images to: {output_dir}")
    print(f"Saved manifest to: {output_dir / 'manifest.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())