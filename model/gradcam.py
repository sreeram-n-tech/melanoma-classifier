"""
model/gradcam.py

Grad-CAM for the trained checkpoint -- answers "is the model looking at the
lesion, or at background/artifacts?" Saves before/after (original | heatmap
overlay) image pairs.

Usage (from the project root):
    python model/gradcam.py --checkpoint model_output/mobilenet_v2_weighted_ce_on_main/best_model.pt \
        --image_dir data/preprocessed --n_samples 8
"""
import argparse
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

from model import build_model, get_target_layer

NORMALIZE = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_idx):
        self.model.zero_grad()
        logits = self.model(input_tensor)
        score = logits[0, class_idx]
        score.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()
        cam = cv2.resize(cam, (input_tensor.shape[-1], input_tensor.shape[-2]))
        cam -= cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam


def overlay_heatmap(orig_img_rgb, cam, alpha=0.4):
    heatmap = (cam * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = (alpha * heatmap + (1 - alpha) * orig_img_rgb).astype(np.uint8)
    return overlay


def run_gradcam(checkpoint_path, image_dir, out_dir, n_samples=8, img_size=224):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=device)
    arch = ckpt["arch"]
    net, _ = build_model(arch, num_classes=2, pretrained=False)
    net.load_state_dict(ckpt["model_state"])
    net.to(device).eval()

    target_layer = get_target_layer(net, arch)
    cam_gen = GradCAM(net, target_layer)

    os.makedirs(out_dir, exist_ok=True)
    files = sorted(f for f in os.listdir(image_dir) if f.endswith(".png"))[:n_samples]

    for fname in files:
        path = os.path.join(image_dir, fname)
        pil_img = Image.open(path).convert("RGB").resize((img_size, img_size))
        np_img = np.array(pil_img)
        tensor = NORMALIZE(T.ToTensor()(pil_img)).unsqueeze(0).to(device)
        tensor.requires_grad_()

        probs = torch.softmax(net(tensor), dim=1)
        pred_class = int(probs.argmax(dim=1).item())

        cam = cam_gen.generate(tensor, pred_class)
        overlay = overlay_heatmap(np_img, cam)

        side_by_side = np.concatenate([np_img, overlay], axis=1)
        out_path = os.path.join(out_dir, f"gradcam_{fname}")
        cv2.imwrite(out_path, cv2.cvtColor(side_by_side, cv2.COLOR_RGB2BGR))

    print(f"Saved {len(files)} Grad-CAM overlays to {out_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--out_dir", default="model_output/gradcam")
    parser.add_argument("--n_samples", type=int, default=8)
    args = parser.parse_args()
    run_gradcam(args.checkpoint, args.image_dir, args.out_dir, args.n_samples)
