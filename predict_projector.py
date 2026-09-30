"""Predict accident risk with Phi-4-multimodal + a trained CNN-feature projection.

``risk`` is the probability of answer "1" after normalizing over the "0"/"1" tokens.

Example:
    python predict_projector.py --model-path models/phi4 --cnn-weights runs/cnn/best.pth \
        --projection runs/projector/cnn_projection_best.pth --clips-dir data/freeway/test --output pred.csv
"""
import argparse
import csv
import os

import torch
from torch.utils.data import DataLoader

from accident.cnn import load_cnn
from accident.projector import ClipDataset, next_token_logits
from accident.vlm import CNN_FEATURE_TOKEN, CNNProjection, binary_answer_token_ids, load_phi4


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-path", required=True)
    p.add_argument("--cnn-weights", required=True)
    p.add_argument("--cnn-depth", type=int, default=4)
    p.add_argument("--projection", required=True, help="checkpoint from train_projector.py")
    p.add_argument("--clips-dir", required=True)
    p.add_argument("--output", required=True, help="CSV with columns file_name,risk,prediction")
    p.add_argument("--num-frames", type=int, default=5)
    p.add_argument("--image-size", type=int, default=216)
    p.add_argument("--hidden-dim", type=int, default=1024)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--attn-implementation", default="flash_attention_2")
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor, model = load_phi4(args.model_path, device, args.attn_implementation, add_cnn_token=True)
    cnn = load_cnn(args.cnn_weights, depth=args.cnn_depth, device=device)
    projection = CNNProjection(cnn.out_channels, args.hidden_dim, model.config.hidden_size)
    projection = projection.load_checkpoint(args.projection, map_location=device).to(device).eval()

    cnn_token_id = processor.tokenizer.convert_tokens_to_ids(CNN_FEATURE_TOKEN)
    answer_ids = binary_answer_token_ids(processor.tokenizer)
    loader = DataLoader(ClipDataset(args.clips_dir, processor, cnn, device,
                                    num_frames=args.num_frames, image_size=args.image_size), batch_size=1)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file_name", "risk", "prediction"])
        for clips, feature, inputs, _ in loader:
            logits = next_token_logits(model, projection, feature, inputs, cnn_token_id, device)
            risk = logits[:, answer_ids].softmax(dim=-1)[0, 1].item()
            writer.writerow([clips[0], f"{risk:.6f}", int(risk >= args.threshold)])
            print(f"{clips[0]}: {risk:.4f}")


if __name__ == "__main__":
    main()
