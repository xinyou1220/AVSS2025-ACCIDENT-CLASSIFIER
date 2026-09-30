"""Ask Llama-3.2-Vision about hazards in dash-cam frames (qualitative baseline).

Example:
    python llama_vision_baseline.py --images data/freeway/train/freeway_0179/00065.jpg
"""
import argparse

import torch
from PIL import Image

DEFAULT_QUESTION = "Is there any hazard for this dashcam owner?"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--images", nargs="+", required=True)
    p.add_argument("--model", default="meta-llama/Llama-3.2-11B-Vision-Instruct")
    p.add_argument("--question", default=DEFAULT_QUESTION)
    p.add_argument("--max-new-tokens", type=int, default=100)
    return p.parse_args()


def main():
    from transformers import AutoProcessor, MllamaForConditionalGeneration

    args = parse_args()
    model = MllamaForConditionalGeneration.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="auto")
    processor = AutoProcessor.from_pretrained(args.model)
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": args.question}]}]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)

    for path in args.images:
        inputs = processor(Image.open(path), prompt, add_special_tokens=False, return_tensors="pt").to(model.device)
        output = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
        answer = processor.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        print(f"=== {path}\n{answer.strip()}\n")


if __name__ == "__main__":
    main()
