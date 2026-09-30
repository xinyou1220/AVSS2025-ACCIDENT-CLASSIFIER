"""Train the CNN-feature projection that is injected into a frozen Phi-4-multimodal.

Only the projection MLP is trained; Phi-4 and the CNN stay frozen. The target is the next token
"0" (no accident) or "1" (accident), optimized with cross-entropy over the vocabulary.

Example:
    python train_projector.py --model-path models/phi4 --cnn-weights runs/cnn/best.pth \
        --data data/freeway/train data/freeway_train.csv --data data/road/train data/road_train.csv
"""
import argparse
import os

import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, random_split

from accident.cnn import load_cnn
from accident.data import read_labels
from accident.projector import ClipDataset, next_token_logits
from accident.vlm import CNN_FEATURE_TOKEN, CNNProjection, binary_answer_token_ids, load_phi4


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-path", required=True, help="Phi-4-multimodal-instruct path or hub id")
    p.add_argument("--cnn-weights", required=True, help="AccidentCNN checkpoint")
    p.add_argument("--cnn-depth", type=int, default=4)
    p.add_argument("--data", nargs=2, action="append", required=True, metavar=("CLIPS_DIR", "LABEL_CSV"),
                   help="clip folder and its file_name,risk CSV; repeat to combine datasets")
    p.add_argument("--output-dir", default="runs/projector")
    p.add_argument("--num-frames", type=int, default=5)
    p.add_argument("--image-size", type=int, default=216)
    p.add_argument("--hidden-dim", type=int, default=1024)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--attn-implementation", default="flash_attention_2")
    return p.parse_args()


def run_epoch(loader, model, projection, cnn_token_id, answer_ids, device, optimizer=None):
    training = optimizer is not None
    projection.train(training)
    total_loss, correct = 0.0, 0
    with torch.set_grad_enabled(training):
        for _, feature, inputs, label in loader:
            target = answer_ids[label.to(device)]
            logits = next_token_logits(model, projection, feature, inputs, cnn_token_id, device)
            loss = F.cross_entropy(logits, target)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item()
            correct += (logits.argmax(dim=-1) == target).sum().item()
    return total_loss / len(loader), correct / len(loader.dataset)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    processor, model = load_phi4(args.model_path, device, args.attn_implementation, add_cnn_token=True)
    for p in model.parameters():
        p.requires_grad = False
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    cnn = load_cnn(args.cnn_weights, depth=args.cnn_depth, device=device)

    dataset = ConcatDataset([
        ClipDataset(clips_dir, processor, cnn, device, read_labels(csv_path), args.num_frames, args.image_size)
        for clips_dir, csv_path in args.data
    ])
    val_size = int(len(dataset) * args.val_ratio)
    train_set, val_set = random_split(dataset, [len(dataset) - val_size, val_size])
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=1)
    print(f"train={len(train_set)} val={len(val_set)}")

    projection = CNNProjection(cnn.out_channels, args.hidden_dim, model.config.hidden_size).to(device)
    optimizer = torch.optim.AdamW(projection.parameters(), lr=args.lr)
    cnn_token_id = processor.tokenizer.convert_tokens_to_ids(CNN_FEATURE_TOKEN)
    answer_ids = torch.tensor(binary_answer_token_ids(processor.tokenizer), device=device)

    best_val_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(train_loader, model, projection, cnn_token_id, answer_ids, device,
                                          optimizer)
        val_loss, val_acc = run_epoch(val_loader, model, projection, cnn_token_id, answer_ids, device)
        print(f"[epoch {epoch:02d}] train loss {train_loss:.4f} acc {train_acc:.4f} | "
              f"val loss {val_loss:.4f} acc {val_acc:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(projection.state_dict(), os.path.join(args.output_dir, "cnn_projection_best.pth"))

    print(f"Best validation loss {best_val_loss:.4f}; projection saved in {args.output_dir}")


if __name__ == "__main__":
    main()
