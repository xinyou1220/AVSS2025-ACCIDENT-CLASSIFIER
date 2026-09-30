"""Describe the last frames of every clip with Qwen2.5-VL and save one JSON per clip.

Output: ``<output-dir>/<clip>.json`` = ``{"subdir": "<clip>", "caption": "..."}``.

Example:
    python generate_captions.py --clips-dir data/freeway/test --output-dir captions/freeway_test
"""
import argparse
import json
import os

import torch
from PIL import Image

from accident.data import last_frames, list_clips

CAPTION_PROMPT = """
For each frame, describe the relative positions, travel directions, and estimated speeds (including acceleration/deceleration) of all vehicles.

Pay attention to changes in inter-vehicle distances, any lane departures, sudden steering maneuvers, or emergency braking events.

Incorporate road markings, traffic signs, pedestrians, or other obstacles to analyze how the environment influences vehicle behavior.

Provide at least three concise bullet-point observations to justify your judgment, for example:

"In frame 3, the vehicle on the left rapidly cuts into our lane at a distance of just 1 meter."

"Frame 4 shows high-rate deceleration with no corresponding braking by the lead car."

"The intersection lacks traffic signals, and pedestrian crossings increase collision risk."
"""


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--clips-dir", required=True, help="directory with one sub-folder of frames per clip")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--num-frames", type=int, default=5)
    p.add_argument("--max-new-tokens", type=int, default=500)
    return p.parse_args()


def main():
    args = parse_args()
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    processor = AutoProcessor.from_pretrained(args.model, use_fast=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model, torch_dtype="auto").to(device).eval()

    for clip in list_clips(args.clips_dir):
        frames = last_frames(os.path.join(args.clips_dir, clip), args.num_frames)
        content = [{"type": "image", "image": Image.open(f).convert("RGB")} for f in frames]
        content.append({"type": "text", "text": CAPTION_PROMPT})
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": content},
        ]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                           padding=True, return_tensors="pt").to(device)

        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_k=50,
                top_p=0.9,
                num_beams=5,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
                early_stopping=True,
            )
        new_tokens = generated[:, inputs.input_ids.shape[1]:]
        caption = processor.batch_decode(new_tokens, skip_special_tokens=True,
                                         clean_up_tokenization_spaces=False)[0].strip()

        save_path = os.path.join(args.output_dir, f"{clip}.json")
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump({"subdir": clip, "caption": caption}, f, ensure_ascii=False, indent=2)
        print(f"[{clip}] caption saved to {save_path}")


if __name__ == "__main__":
    main()
