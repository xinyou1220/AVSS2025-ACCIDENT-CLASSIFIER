"""Saliency maps and Grad-CAM for trained accident CNN checkpoints.

Pass one checkpoint, or a directory of per-epoch checkpoints to see how attention evolves.

Example:
    python explain_cnn.py --checkpoint runs/cnn/best.pth --data-dir data/cnn/validation --indices 0 10 20
    python explain_cnn.py --checkpoint runs/cnn/every_epoch --data-dir data/cnn/training --method gradcam
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from accident.cnn import load_cnn  # noqa: E402
from accident.data import denormalize, load_image_folder  # noqa: E402
from accident.explain import grad_cam, overlay_heatmap, saliency_maps  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, help=".pth file or a directory of .pth files")
    p.add_argument("--data-dir", required=True, help="image folder <label>/<image>")
    p.add_argument("--indices", type=int, nargs="+", default=[0, 1, 2, 3, 4], help="images to visualize")
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--layer", default="down4.conv1", help="Grad-CAM target layer")
    p.add_argument("--method", choices=("saliency", "gradcam", "both"), default="both")
    p.add_argument("--output-dir", default="runs/explain")
    return p.parse_args()


def save_figure(images, overlays, titles, suptitle, path, cmap=None):
    n = len(images)
    fig, axes = plt.subplots(2, n, figsize=(3 * n, 6), squeeze=False)
    for i in range(n):
        axes[0][i].imshow(images[i])
        axes[0][i].set_title(titles[i])
        axes[1][i].imshow(overlays[i], cmap=cmap)
        for ax in (axes[0][i], axes[1][i]):
            ax.axis("off")
    fig.suptitle(suptitle)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    if os.path.isdir(args.checkpoint):
        checkpoints = sorted(os.path.join(args.checkpoint, f)
                             for f in os.listdir(args.checkpoint) if f.endswith(".pth"))
    else:
        checkpoints = [args.checkpoint]

    x_all, y_all = load_image_folder(args.data_dir, (args.image_size, args.image_size))
    images, labels = x_all[args.indices], y_all[args.indices]
    display = [denormalize(img) for img in images]
    titles = [f"#{i} (label {int(y)})" for i, y in zip(args.indices, labels)]

    for path in checkpoints:
        name = os.path.splitext(os.path.basename(path))[0]
        model = load_cnn(path, depth=args.depth, device=device)
        print(f"Explaining {path}")

        if args.method in ("saliency", "both"):
            maps = saliency_maps(model, images.clone(), labels, device)
            save_figure(display, list(maps.numpy()), titles, f"Saliency - {name}",
                        os.path.join(args.output_dir, f"{name}_saliency.png"), cmap="hot")

        if args.method in ("gradcam", "both"):
            cams = grad_cam(model, images, model.get_submodule(args.layer), device)
            overlays = [overlay_heatmap(img, heatmap) for img, (heatmap, _) in zip(display, cams)]
            cam_titles = [f"{t}\npred {pred}" for t, (_, pred) in zip(titles, cams)]
            save_figure(display, overlays, cam_titles, f"Grad-CAM ({args.layer}) - {name}",
                        os.path.join(args.output_dir, f"{name}_gradcam.png"))

    print(f"Figures saved to {args.output_dir}")


if __name__ == "__main__":
    main()
