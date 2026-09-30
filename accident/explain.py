"""Saliency maps and Grad-CAM for the accident CNN."""
import matplotlib.cm as cm
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def _min_max(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-8)


def saliency_maps(model, images, labels, device):
    """|d loss / d input|, max over RGB, min-max normalized per image. Returns ``[N, H, W]``."""
    model.eval()
    images = images.to(device).requires_grad_()
    loss = F.cross_entropy(model(images), labels.to(device))
    loss.backward()
    grads = images.grad.abs().max(dim=1).values.detach().cpu()
    return torch.stack([_min_max(g) for g in grads])


def grad_cam(model, images, target_layer, device):
    """Grad-CAM for the predicted class of each image.

    Returns a list of ``(heatmap [h, w] in [0, 1], predicted_class)``.
    """
    model.eval()
    activations, gradients = [], []
    fwd = target_layer.register_forward_hook(lambda m, i, o: activations.append(o.detach()))
    bwd = target_layer.register_full_backward_hook(lambda m, gi, go: gradients.append(go[0].detach()))
    results = []
    try:
        for image in images:
            activations.clear()
            gradients.clear()
            output = model(image.unsqueeze(0).to(device).requires_grad_())
            pred = output.argmax(dim=1).item()
            model.zero_grad()
            output[0, pred].backward()

            fmap, grad = activations[0][0], gradients[0][0]
            channel_weights = grad.mean(dim=(1, 2))
            heatmap = F.relu((fmap * channel_weights[:, None, None]).sum(dim=0))
            results.append((_min_max(heatmap).cpu().numpy(), pred))
    finally:
        fwd.remove()
        bwd.remove()
    return results


def overlay_heatmap(image, heatmap, alpha=0.4):
    """Blends a jet-colored ``heatmap`` onto ``image`` (``[H, W, 3]`` in [0, 1])."""
    h, w = image.shape[:2]
    resized = np.asarray(Image.fromarray(np.uint8(255 * heatmap)).resize((w, h))) / 255.0
    colored = cm.jet(resized)[..., :3]
    return np.clip((1 - alpha) * image + alpha * colored, 0, 1)
