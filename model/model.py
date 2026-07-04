"""
model/model.py -- transfer-learning architectures and the freeze/unfreeze
helpers used for two-stage training (frozen head, then fine-tune).
"""
import torch.nn as nn
import torchvision.models as models


def build_model(arch="mobilenet_v2", num_classes=2, pretrained=True):
    if arch == "mobilenet_v2":
        weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        net = models.mobilenet_v2(weights=weights)
        in_features = net.classifier[1].in_features
        net.classifier[1] = nn.Linear(in_features, num_classes)
        backbone = net.features
    elif arch == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        net = models.efficientnet_b0(weights=weights)
        in_features = net.classifier[1].in_features
        net.classifier[1] = nn.Linear(in_features, num_classes)
        backbone = net.features
    elif arch == "efficientnet_b3":
        weights = models.EfficientNet_B3_Weights.DEFAULT if pretrained else None
        net = models.efficientnet_b3(weights=weights)
        in_features = net.classifier[1].in_features
        net.classifier[1] = nn.Linear(in_features, num_classes)
        backbone = net.features
    else:
        raise ValueError(f"Unknown arch: {arch}")
    return net, backbone


def freeze_backbone(backbone):
    for p in backbone.parameters():
        p.requires_grad = False


def unfreeze_last_n_blocks(backbone, n=3):
    """Unfreezes the last n top-level blocks of the feature extractor for
    Stage-2 fine-tuning, leaving the earlier (more generic) layers frozen."""
    children = list(backbone.children())
    for block in children[-n:]:
        for p in block.parameters():
            p.requires_grad = True


def get_target_layer(net, arch):
    """Returns the last conv layer -- used as the Grad-CAM target layer."""
    if arch in ("mobilenet_v2", "efficientnet_b0", "efficientnet_b3"):
        return net.features[-1]
    raise ValueError(f"Unknown arch: {arch}")
