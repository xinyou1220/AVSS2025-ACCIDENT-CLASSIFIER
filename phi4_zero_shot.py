"""Zero-shot accident-risk estimation with Phi-4-multimodal on the last frames of each clip.

The model is asked for the probability (0-1) that an accident happens within the next 2 seconds.
With ``--caption-dir`` the Qwen2.5-VL caption of each clip (see generate_captions.py) is added to
the prompt as extra scene context.

Example:
    python phi4_zero_shot.py --model-path models/phi4 --clips-dir data/freeway/test --output results.csv
"""
import argparse
import csv
import json
import os

import torch
from PIL import Image
from transformers import GenerationConfig

from accident.data import last_frames, list_clips
from accident.vlm import chat_prompt, image_placeholders, load_phi4

QUESTION = (
    "The following {n} images are consecutive frames from a continuous video sequence taken from a moving "
    "vehicle. Please analyze them as a temporally correlated sequence. Based on this visual input, what is "
    "the probability (from 0 to 1) that a car accident will happen within the next 2 seconds? "
    "Respond only with a numeric value between 0 and 1."
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-path", required=True, help="local path or hub id of Phi-4-multimodal-instruct")
    p.add_argument("--clips-dir", required=True)
    p.add_argument("--output", required=True, help="CSV with columns file_name,risk")
    p.add_argument("--num-frames", type=int, default=10)
    p.add_argument("--caption-dir", help="optional folder of <clip>.json captions")
    p.add_argument("--max-new-tokens", type=int, default=10)
    p.add_argument("--attn-implementation", default="flash_attention_2")
    return p.parse_args()


def load_caption(caption_dir, clip):
    path = os.path.join(caption_dir, f"{clip}.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)["caption"]


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor, model = load_phi4(args.model_path, device, args.attn_implementation)
    generation_config = GenerationConfig.from_pretrained(args.model_path)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file_name", "risk"])
        for clip in list_clips(args.clips_dir):
            try:
                frames = last_frames(os.path.join(args.clips_dir, clip), args.num_frames)
            except ValueError as e:
                print(f"Skipping {clip}: {e}")
                continue
            images = [Image.open(p) for p in frames]

            user = image_placeholders(len(images)) + QUESTION.format(n=len(images))
            caption = load_caption(args.caption_dir, clip) if args.caption_dir else None
            if caption:
                user += f"\nScene description of the last frames:\n{caption}"
            prompt = chat_prompt(processor, "", user)

            inputs = processor(prompt, images, return_tensors="pt").to(device)
            with torch.no_grad():
                output_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
                                            generation_config=generation_config)
            response = processor.batch_decode(output_ids[:, inputs["input_ids"].shape[1]:],
                                              skip_special_tokens=True,
                                              clean_up_tokenization_spaces=False)[0].strip()
            print(f"{clip}: {response}")
            writer.writerow([clip, response])


if __name__ == "__main__":
    main()
