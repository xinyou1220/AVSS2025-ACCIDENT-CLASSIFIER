"""Helpers for Phi-4-multimodal: loading, prompts and CNN-feature token injection."""
import torch
import torch.nn as nn

CNN_FEATURE_TOKEN = "<|cnn_feature|>"


def load_phi4(model_path, device, attn_implementation="flash_attention_2", add_cnn_token=False):
    """Loads the processor and model (bf16). Optionally registers the ``<|cnn_feature|>`` token."""
    from transformers import AutoModelForCausalLM, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    if add_cnn_token:
        processor.tokenizer.add_special_tokens({"additional_special_tokens": [CNN_FEATURE_TOKEN]})
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        _attn_implementation=attn_implementation,
    )
    if add_cnn_token:
        model.resize_token_embeddings(len(processor.tokenizer))
    return processor, model.to(device).eval()


def image_placeholders(num_images):
    return "".join(f"<|image_{i}|>\n" for i in range(1, num_images + 1))


def chat_prompt(processor, system, user):
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    return processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


class CNNProjection(nn.Module):
    """Two-layer MLP mapping pooled CNN features to the language model's hidden size."""

    def __init__(self, cnn_dim, hidden_dim, output_dim):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(cnn_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        return self.mlp(x)

    def load_checkpoint(self, path, map_location="cpu"):
        state = torch.load(path, map_location=map_location)
        # Checkpoints saved from nn.DataParallel prefix every key with "module.".
        self.load_state_dict({k.removeprefix("module."): v for k, v in state.items()})
        return self


def embed_with_cnn_token(model, input_ids, projected, cnn_token_id):
    """Token embeddings where every ``<|cnn_feature|>`` position is replaced by ``projected[b]``."""
    embeddings = model.get_input_embeddings()(input_ids)
    mask = (input_ids == cnn_token_id).unsqueeze(-1)
    projected = projected.to(embeddings.dtype).unsqueeze(1).expand_as(embeddings)
    return torch.where(mask, projected, embeddings)


def binary_answer_token_ids(tokenizer):
    """Token ids of the answers "0" and "1"."""
    return [tokenizer.convert_tokens_to_ids(str(c)) for c in (0, 1)]
