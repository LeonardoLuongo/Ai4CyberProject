"""Generate error-generic FGSM adversarial images for NN1.

This attack tries to make NN1 misclassify each input image without forcing a
specific target class. It saves adversarial PNG images and a manifest CSV.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from attack_common import (
    batched,
    build_nn1_art_classifier,
    clean_reference_path_for_sample,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate error-generic FGSM adversarial images against NN1."
    )
    parser.add_argument("--input-dir", type=Path, default=Path("vggface2_subset/test"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--eps", type=float, default=0.025)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=160)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    validate_eps(args.eps)

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = default_output_dir(Path("attacks"), "fgsm_error_generic", args.eps)
    ensure_clean_output_dir(output_dir, args.overwrite)

    samples = discover_test_images(args.input_dir)
    if args.max_images is not None:
        samples = samples[: args.max_images]
    print(f"Found {len(samples)} test images.")

    print("Loading NN1 and wrapping it with ART PyTorchClassifier...")
    classifier, num_classes, device_type = build_nn1_art_classifier()

    from art.attacks.evasion import FastGradientMethod

    attack = FastGradientMethod(estimator=classifier, eps=args.eps, targeted=False)

    rows: list[dict[str, object]] = []
    done = 0
    for batch_index, batch_samples in enumerate(batched(samples, args.batch_size), start=1):
        x = load_batch_for_nn1(batch_samples, args.image_size)
        x_adv = attack.generate(x=x)

        for sample, original, adversarial in zip(batch_samples, x, x_adv):
            adv_path = output_path_for_sample(output_dir, sample)
            clean_reference_path = clean_reference_path_for_sample(output_dir, sample)
            save_adv_image(original, clean_reference_path)
            save_adv_image(adversarial, adv_path)
            linf, mean_abs = perturbation_stats(original, adversarial)
            rows.append(
                {
                    "attack_type": "fgsm_error_generic",
                    "eps": args.eps,
                    "targeted": False,
                    "target_class": "",
                    "dataset_label": sample.dataset_label if sample.dataset_label is not None else "",
                    "identity_id": sample.identity_id or "",
                    "identity_name": sample.identity_name or "",
                    "identity_dir": sample.identity_dir,
                    "source_image_path": str(sample.image_path),
                    "clean_reference_image_path": str(clean_reference_path),
                    "adversarial_image_path": str(adv_path),
                    "linf": linf,
                    "mean_abs_perturbation": mean_abs,
                }
            )

        done += len(batch_samples)
        print(progress_line(done, len(samples), batch_index))

    write_manifest(output_dir / "manifest.csv", rows)
    write_json(
        output_dir / "config.json",
        {
            "attack_type": "fgsm_error_generic",
            "classifier": "facenet_pytorch.InceptionResnetV1(pretrained='vggface2', classify=True)",
            "art_estimator": "PyTorchClassifier",
            "eps": args.eps,
            "max_allowed_eps": 0.1,
            "targeted": False,
            "input_dir": str(args.input_dir),
            "output_dir": str(output_dir),
            "num_images": len(samples),
            "max_images": args.max_images,
            "batch_size": args.batch_size,
            "image_size": args.image_size,
            "num_classes": num_classes,
            "device_type": device_type,
            "saved_format": "PNG",
            "clean_reference": "clean image after NN1 preprocessing, saved for paired comparisons",
        },
    )

    print(f"Saved adversarial images to: {output_dir}")
    print(f"Saved manifest to: {output_dir / 'manifest.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
