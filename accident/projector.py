"""CNN-feature injection into Phi-4-multimodal via a learned projection of a special token.

For each clip the last ``num_frames`` frames go to Phi-4 as images, and the CNN feature of the last
frame (global-average-pooled ``[C]`` vector) is projected to the LM hidden size and written into the
embedding of the ``<|cnn_feature|>`` token. The answer is read from the next-token logits of "0"/"1".
"""
import os

import torch
from PIL import Image
from torch.utils.data import Dataset

from .data import cnn_transform, last_frames
from .vlm import CNN_FEATURE_TOKEN, chat_prompt, embed_with_cnn_token, image_placeholders

SYSTEM_PROMPT = (
    "The following {n} frames and their CNN feature map (from the last image) are consecutive frames from "
    "a video. Please analyze them as temporally correlated inputs. Respond only with either 0 or 1 to "
    "indicate whether a car accident will happen within the next 50 frames."
)


class ClipDataset(Dataset):
    """Yields ``(clip_name, cnn_feature [C], processor_inputs, label)`` for every clip folder.

    ``labels`` maps clip name to 0/1; clips without a label get ``-1`` (prediction only).
    """

    def __init__(self, clips_dir, processor, cnn, device, labels=None, num_frames=5, image_size=216):
        self.clips_dir = clips_dir
        self.clips = sorted(d for d in os.listdir(clips_dir) if os.path.isdir(os.path.join(clips_dir, d)))
        if labels is not None:
            self.clips = [c for c in self.clips if c in labels]
        self.labels = labels or {}
        self.processor = processor
        self.cnn = cnn
        self.device = device
        self.num_frames = num_frames
        self.image_size = (image_size, image_size)
        self.cnn_transform = cnn_transform(self.image_size)

    def __len__(self):
        return len(self.clips)

    @torch.no_grad()
    def _cnn_feature(self, image):
        x = self.cnn_transform(image).unsqueeze(0).to(self.device)
        return self.cnn.features(x).mean(dim=(2, 3)).squeeze(0).cpu()

    def __getitem__(self, idx):
        clip = self.clips[idx]
        frames = [Image.open(p).convert("RGB").resize(self.image_size, Image.BILINEAR)
                  for p in last_frames(os.path.join(self.clips_dir, clip), self.num_frames)]

        user = image_placeholders(len(frames)) + f"{CNN_FEATURE_TOKEN} This corresponds to the last image only\n"
        prompt = chat_prompt(self.processor, SYSTEM_PROMPT.format(n=len(frames)), user)
        inputs = self.processor(prompt, images=frames, return_tensors="pt")
        inputs = {k: v.squeeze(0) for k, v in inputs.items() if v is not None}

        label = int(self.labels.get(clip, -1))
        return clip, self._cnn_feature(frames[-1]), inputs, torch.tensor(label)


def next_token_logits(model, projection, cnn_feature, inputs, cnn_token_id, device):
    """Vocabulary logits for the token after the prompt, shape ``[B, vocab]``."""
    inputs = {k: v.to(device) for k, v in inputs.items()}
    input_ids = inputs.pop("input_ids")
    projected = projection(cnn_feature.to(device))
    inputs["inputs_embeds"] = embed_with_cnn_token(model, input_ids, projected, cnn_token_id)
    return model(**inputs).logits[:, -1, :].float()
