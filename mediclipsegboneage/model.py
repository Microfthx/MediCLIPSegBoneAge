import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


def _ensure_medclipseg_importable(medclipseg_root):
    root = Path(medclipseg_root).resolve()
    if not root.exists():
        raise FileNotFoundError(f"MedCLIPSeg root not found: {root}")
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


class PromptEncoder(nn.Module):
    def __init__(self, clip_module, clip_model):
        super().__init__()
        self.clip_module = clip_module
        self.clip_model = clip_model

    def forward(self, prompts, device):
        tokenized_prompts = self.clip_module.tokenize(list(prompts)).to(device)
        x = self.clip_model.token_embedding(tokenized_prompts).type(self.clip_model.dtype)
        x = x + self.clip_model.positional_embedding.type(self.clip_model.dtype)
        x = x.permute(1, 0, 2)
        x = self.clip_model.transformer([x])[0]
        x = x.permute(1, 0, 2)
        x = self.clip_model.ln_final(x).type(self.clip_model.dtype)
        x = x[torch.arange(x.shape[0], device=device), tokenized_prompts.argmax(dim=-1)] @ self.clip_model.text_projection
        return x.float()


def _load_medclipseg_clip_model(clip_module, backbone, device):
    if backbone not in clip_module._MODELS:
        raise ValueError(f"Unsupported backbone: {backbone}")

    model_path = clip_module._download(clip_module._MODELS[backbone])
    try:
        state_dict = torch.jit.load(model_path, map_location="cpu").state_dict()
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    design_details = {
        "trainer": "MedCLIPSeg",
        "vision_depth": 0,
        "language_depth": 0,
        "vision_ctx": 0,
        "language_ctx": 0,
    }
    model = clip_module.build_model(state_dict, design_details)
    model = model.to(device)
    model.float()
    return model


def _encode_image_features(clip_model, image):
    visual = clip_model.visual
    x = visual.conv1(image.type(clip_model.dtype))
    x = x.reshape(x.shape[0], x.shape[1], -1)
    x = x.permute(0, 2, 1)
    x = torch.cat(
        [
            visual.class_embedding.to(x.dtype)
            + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
            x,
        ],
        dim=1,
    )
    x = x + visual.positional_embedding.to(x.dtype)

    if getattr(visual, "VPT_shallow", False):
        visual_ctx = visual.VPT.expand(x.shape[0], -1, -1).to(dtype=x.dtype, device=x.device)
        x = torch.cat([x, visual_ctx], dim=1)

    x = visual.ln_pre(x)
    x = x.permute(1, 0, 2)
    x = visual.transformer([x, []])[0]
    x = x.permute(1, 0, 2)
    x = visual.ln_post(x[:, 0, :])

    if visual.proj is not None:
        x = x @ visual.proj

    return x.float()


class MedCLIPBoneAgeRegressor(nn.Module):
    def __init__(
        self,
        medclipseg_root,
        backbone="ViT-B/16",
        freeze_image_encoder=False,
        gender_dim=32,
        hidden_dim=512,
        dropout=0.2,
    ):
        super().__init__()
        _ensure_medclipseg_importable(medclipseg_root)

        from clip import clip as medclip_clip

        self.clip_module = medclip_clip
        self.clip_model = _load_medclipseg_clip_model(medclip_clip, backbone, device="cpu")
        self.prompt_encoder = PromptEncoder(medclip_clip, self.clip_model)

        if freeze_image_encoder:
            for param in self.clip_model.visual.parameters():
                param.requires_grad_(False)

        self.image_feature_dim = self.clip_model.text_projection.shape[1]
        self.text_feature_dim = self.clip_model.text_projection.shape[1]

        self.gender_branch = nn.Sequential(
            nn.Linear(1, gender_dim),
            nn.ReLU(inplace=True),
        )
        self.fusion = nn.Sequential(
            nn.Linear(self.image_feature_dim + self.text_feature_dim + gender_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

        for module in self.fusion:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, image, prompts, gender):
        device = image.device
        self.clip_model = self.clip_model.to(device)

        image_features = _encode_image_features(self.clip_model, image)
        text_features = self.prompt_encoder(prompts, device)

        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        gender_features = self.gender_branch(gender)

        fused = torch.cat([image_features, text_features, gender_features], dim=1)
        return self.fusion(fused).squeeze(1)
