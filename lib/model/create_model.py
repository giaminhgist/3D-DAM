import os
import numpy as np
from .DuoAttention import Duo_Attention
from .vit import ViT
import torch
from .AttentionNet import AttentionNet

model_dict = {
    'DuoAttention':Duo_Attention,
    'VIT': ViT,
    'PDAM': AttentionNet,
    'NET': AttentionNet,
}


def create_model(
        model_name: str,
        num_classes: int,
        pretrained_path: str = None,
        **kwargs,
):
    model = model_dict[model_name](
        num_classes=num_classes,
        **kwargs,
    )

    if pretrained_path is not None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        print('Load pretrained...')
        model.module.load_state_dict(
            torch.load(
                pretrained_path,
                map_location=str(device))
        )

    return model

