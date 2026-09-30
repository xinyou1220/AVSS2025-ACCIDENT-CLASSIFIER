# AVSS 2025 Accident Classification

Code for our experiments on **traffic-accident anticipation from dash-cam video** for the AVSS 2025 accident classification task. Given the last frames of a clip, we predict whether an accident will happen shortly afterwards.

The repository covers four approaches:

| Approach | Script(s) | Idea |
|---|---|---|
| Frame CNN | `train_cnn.py`, `explain_cnn.py` | Small CNN on single frames, with Saliency / Grad-CAM for interpretation |
| VLM zero-shot | `phi4_zero_shot.py` | Phi-4-multimodal reads 10 consecutive frames and outputs an accident probability |
| VLM + captions | `generate_captions.py` → `phi4_zero_shot.py --caption-dir` | Qwen2.5-VL writes a traffic-scene description that is added to the Phi-4 prompt |
| CNN → VLM fusion | `train_projector.py`, `predict_projector.py` | The CNN feature is projected into Phi-4's embedding space through a special `<|cnn_feature|>` token |

`llama_vision_baseline.py` is a small qualitative baseline that asks Llama-3.2-Vision about hazards in individual frames.

## Installation

```bash
pip install -r requirements.txt
```

The CNN scripts only need the core packages. The VLM scripts need a CUDA GPU and access to the Hugging Face checkpoints: `microsoft/Phi-4-multimodal-instruct`, `Qwen/Qwen2.5-VL-7B-Instruct`, and `meta-llama/Llama-3.2-11B-Vision-Instruct` (gated). Phi-4 uses FlashAttention 2 by default. If you do not have `flash-attn`, pass `--attn-implementation eager`.

## Data layout

The dataset is not included. Obtain it from the challenge organizers.

**Clips** (VLM and projector scripts): one folder of frames per clip. Frames are sorted by file name, and the *last* `N` frames are used.

```
data/freeway/train/
  freeway_0000/00001.jpg 00002.jpg ...
  freeway_0001/...
data/freeway_train.csv          # columns: file_name,risk   (risk = 0 or 1)
```

**Frame classification** (CNN): an image folder whose sub-folder names are integer labels, for example built from the last frames of each clip.

```
data/cnn/training/0/*.jpg   data/cnn/training/1/*.jpg
data/cnn/validation/0/*.jpg data/cnn/validation/1/*.jpg
```

## Usage

### 1. Frame CNN

```bash
python train_cnn.py --train-dir data/cnn/training --val-dir data/cnn/validation --depth 4 --image-size 512
```

`AccidentCNN(depth)` stacks `depth` blocks of Conv3×3–BN–ReLU–MaxPool (32→64→128→256→512 channels) with dropout, followed by global average pooling and a 3-layer MLP head. We used `depth=4` (256-d features) for the fusion experiments. The script saves weights for every epoch to `runs/cnn/every_epoch/`, the checkpoint with the lowest validation loss to `runs/cnn/best.pth`, and a loss/accuracy plot.

```bash
# Saliency + Grad-CAM for one checkpoint
python explain_cnn.py --checkpoint runs/cnn/best.pth --data-dir data/cnn/validation --indices 0 10 20 30 40
# Grad-CAM for every epoch, to see how attention evolves during training
python explain_cnn.py --checkpoint runs/cnn/every_epoch --data-dir data/cnn/training --method gradcam
```

### 2. Phi-4 zero-shot (optionally with captions)

```bash
python phi4_zero_shot.py --model-path microsoft/Phi-4-multimodal-instruct \
    --clips-dir data/freeway/test --output results/phi4_freeway_test.csv

# add Qwen2.5-VL scene descriptions to the prompt
python generate_captions.py --clips-dir data/freeway/test --output-dir captions/freeway_test
python phi4_zero_shot.py ... --caption-dir captions/freeway_test
```

The output CSV contains `file_name,risk`, where `risk` is the model's raw text answer (expected to be a number between 0 and 1).

### 3. CNN feature → Phi-4 fusion

The prompt contains the last 5 frames as images and a special token `<|cnn_feature|>`. The trained CNN's pooled feature of the last frame (`[256]`) passes through a 2-layer MLP (`256 → 1024 → hidden_size`). The result **replaces the embedding of `<|cnn_feature|>`**. Phi-4 and the CNN stay frozen. Only the projection is trained, with cross-entropy on the next token (`"0"` = no accident, `"1"` = accident).

```bash
python train_projector.py --model-path microsoft/Phi-4-multimodal-instruct --cnn-weights runs/cnn/best.pth \
    --data data/freeway/train data/freeway_train.csv \
    --data data/road/train    data/road_train.csv

python predict_projector.py --model-path microsoft/Phi-4-multimodal-instruct --cnn-weights runs/cnn/best.pth \
    --projection runs/projector/cnn_projection_best.pth --clips-dir data/freeway/test --output results/fusion.csv
```

`predict_projector.py` writes `file_name,risk,prediction`, where `risk = P("1") / (P("0") + P("1"))` at the answer position.

## Repository layout

```
accident/
  cnn.py        AccidentCNN (checkpoint-compatible parameter names) and loader
  data.py       image-folder loading, clip/frame listing, label CSV reader
  explain.py    saliency maps, Grad-CAM, heat-map overlay
  vlm.py        Phi-4 loading, prompts, CNN projection, token-embedding injection
  projector.py  clip dataset and forward pass for the CNN→Phi-4 fusion
train_cnn.py  explain_cnn.py  generate_captions.py  phi4_zero_shot.py
train_projector.py  predict_projector.py  llama_vision_baseline.py
```

## Status

This project is archived and no longer maintained. Issues and pull requests may not receive a response.

## License

The code is released under the [MIT License](LICENSE). Datasets and pretrained models used by the scripts are subject to their own licenses.
