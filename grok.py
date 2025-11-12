#!/usr/bin/env python3
# corrected_biomedclip_test.py
import os
import random
import argparse
import json
from math import sqrt
from typing import List, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

# open_clip
from huggingface_hub import hf_hub_download
from open_clip import create_model_and_transforms, get_tokenizer
from open_clip.factory import HF_HUB_PREFIX, _MODEL_CONFIGS

REAL_NAME = {'Brain': 'Brain', 'Liver':'Liver', 'Retina_RESC':'retinal OCT', 'Chest':'Chest X-ray film', 'Retina_OCT2017':'retinal OCT', 'Histopathology':'histopathological image'}
# -------------------------------------------------------------------------
# 1. Constants

# Biomed datasets indexing (you can keep as is if relevant for your task)
CLASS_INDEX = {'Brain': 3, 'Liver': 2, 'Retina_RESC': 1, 'Retina_OCT2017': -1, 'Chest': -2, 'Histopathology': -3}
CLASS_NAMES = {'Brain': 3, 'Liver': 2, 'Retina_RESC': 1, 'Retina_OCT2017': -1, 'Chest': -2, 'Histopathology': -3}
CLASS_INDEX_INV = {v: k for k, v in CLASS_INDEX.items()}


# -------------------------------------------------------------------------
# 2. Losses (kept for completeness – not used in test)
# -------------------------------------------------------------------------
class FocalLoss(nn.Module):
    def __init__(self, apply_nonlin=None, alpha=None, gamma=2,
                 balance_index=0, smooth=1e-5, size_average=True):
        super().__init__()
        self.apply_nonlin = apply_nonlin
        self.alpha = alpha
        self.gamma = gamma
        self.balance_index = balance_index
        self.smooth = smooth
        self.size_average = size_average

    def forward(self, logit, target):
        if self.apply_nonlin is not None:
            logit = self.apply_nonlin(logit)
        num_class = logit.shape[1]

        if logit.dim() > 2:
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.permute(0, 2, 1).contiguous()
            logit = logit.view(-1, logit.size(-1))

        target = target.view(-1, 1)
        alpha = self.alpha
        if alpha is None:
            alpha = torch.ones(num_class, 1)
        elif isinstance(alpha, (list, np.ndarray)):
            alpha = torch.FloatTensor(alpha).view(num_class, 1)
            alpha = alpha / alpha.sum()
        elif isinstance(alpha, float):
            alpha = torch.ones(num_class, 1) * (1 - self.alpha)
            alpha[self.balance_index] = self.alpha
        else:
            raise TypeError('Unsupported alpha type')

        if alpha.device != logit.device:
            alpha = alpha.to(logit.device)

        idx = target.cpu().long()
        one_hot = torch.zeros(target.size(0), num_class, device=logit.device)
        one_hot = one_hot.scatter_(1, idx, 1)

        if self.smooth:
            one_hot = torch.clamp(one_hot, self.smooth / (num_class - 1),
                                  1.0 - self.smooth)

        pt = (one_hot * logit).sum(1) + self.smooth
        logpt = pt.log()
        alpha = alpha[idx].squeeze()
        loss = -alpha * torch.pow(1 - pt, self.gamma) * logpt

        return loss.mean() if self.size_average else loss


class BinaryDiceLoss(nn.Module):
    def forward(self, input, targets):
        smooth = 1.0
        input_flat = input.view(input.size(0), -1)
        target_flat = targets.view(targets.size(0), -1)
        intersection = input_flat * target_flat
        score = (2. * intersection.sum(1) + smooth) / \
                (input_flat.sum(1) + target_flat.sum(1) + smooth)
        return 1 - score.mean()


loss_focal = FocalLoss()
loss_dice = BinaryDiceLoss()
loss_bce = nn.BCEWithLogitsLoss()


# -------------------------------------------------------------------------
# 3. Dataset
# -------------------------------------------------------------------------
class MedTestDataset(Dataset):
    def __init__(self, dataset_path: str = './data/', class_name: str = 'Liver',
                 resize: int = 224):
        assert class_name in CLASS_NAMES, f"{class_name} not in {CLASS_NAMES}"
        self.dataset_path = os.path.join(dataset_path, f'{class_name}_AD')
        self.class_name = class_name
        self.seg_flag = CLASS_INDEX[class_name]   # >0 → masks exist
        self.resize = resize

        self.x, self.y, self.mask = self._load_dataset_folder()
        self.transform_img = transforms.Compose([
            transforms.Resize((resize, resize), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
        ])
        self.transform_mask = transforms.Compose([
            transforms.Resize((resize, resize), interpolation=Image.NEAREST),
            transforms.ToTensor(),
        ])

    # ---------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.x)

    # ---------------------------------------------------------------------
    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        img_path, label, mask_path = self.x[idx], self.y[idx], self.mask[idx]

        # ----- image -----
        img = Image.open(img_path).convert('RGB')
        img_tensor = self.transform_img(img)                     # (C, H, W)

        # ----- label -----
        label_tensor = torch.tensor(float(label), dtype=torch.float32)

        # ----- mask -----
        if self.seg_flag <= 0 or mask_path is None:
            mask_tensor = torch.zeros((1, self.resize, self.resize), dtype=torch.float32)
        else:
            mask = Image.open(mask_path).convert('L')
            mask_tensor = self.transform_mask(mask)              # (1, H, W)

        return img_tensor, label_tensor, mask_tensor

    # ---------------------------------------------------------------------
    def _load_dataset_folder(self) -> Tuple[List[str], List[int], List[str]]:
        x, y, mask = [], [], []

        # normal (good) images
        normal_dir = os.path.join(self.dataset_path, 'test/good/img')
        if os.path.isdir(normal_dir):
            imgs = sorted([os.path.join(normal_dir, f) for f in os.listdir(normal_dir)])
            x.extend(imgs)
            y.extend([0] * len(imgs))
            mask.extend([None] * len(imgs))

        # abnormal (Ungood) images
        abnorm_dir = os.path.join(self.dataset_path, 'test/Ungood/img')
        if os.path.isdir(abnorm_dir):
            imgs = sorted([os.path.join(abnorm_dir, f) for f in os.listdir(abnorm_dir)])
            x.extend(imgs)
            y.extend([1] * len(imgs))

        # masks (only when segmentation is available)
        if self.seg_flag > 0:
            mask_dir = os.path.join(self.dataset_path, 'test/Ungood/anomaly_mask')
            if os.path.isdir(mask_dir):
                masks = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir)])
                # pad with None if fewer masks than abnormal images
                needed = len([p for p in x if '/Ungood/' in p])
                if len(masks) < needed:
                    masks += [None] * (needed - len(masks))
                mask.extend(masks[:needed])
            else:
                mask.extend([None] * len([p for p in x if '/Ungood/' in p]))

        assert len(x) == len(y) == len(mask), "Lengths must match"
        return x, y, mask


# -------------------------------------------------------------------------
# 4. Utility
# -------------------------------------------------------------------------
def setup_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='BiomedCLIP Testing')
    parser.add_argument('--model_name', type=str,
                        default='BiomedCLIP-PubMedBERT-ViT-B-16')
    parser.add_argument('--text_encoder', type=str,
                        default='microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext')
    parser.add_argument('--pretrain', type=str, default='microsoft')
    parser.add_argument('--obj', type=str, default='Liver')
    parser.add_argument('--data_path', type=str, default='./data/')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--seed', type=int, default=111)
    parser.add_argument('--save_model', type=int, default=0)
    return parser.parse_args()


# -------------------------------------------------------------------------
# 5. Prompt ensemble
# -------------------------------------------------------------------------
def encode_text_with_prompt_ensemble(model, tokenizer, obj: str,
                                    device: torch.device) -> torch.Tensor:
    prompt_normal = ['{}', 'flawless {}', 'perfect {}', 'unblemished {}',
                     '{} without flaw', '{} without defect', '{} without damage']
    prompt_abnormal = ['damaged {}', 'broken {}', '{} with flaw',
                       '{} with defect', '{} with damage']
    prompt_state = [prompt_normal, prompt_abnormal]

    templates = [
        'a bad photo of a {}.', 'a low resolution photo of the {}.',
        'a cropped photo of the {}.', 'a bright photo of a {}.',
        'a dark photo of the {}.', 'a photo of my {}.', 'a photo of the cool {}.',
        'a close-up photo of a {}.', 'a black and white photo of the {}.',
        'a jpeg corrupted photo of a {}.', 'a blurry photo of the {}.',
        'a good photo of the {}.', 'a photo of one {}.', 'a photo of a {}.',
        'a low resolution photo of a {}.', 'a photo of a large {}.',
        'a blurry photo of a {}.', 'a jpeg corrupted photo of a {}.',
        'a good photo of a {}.', 'a photo of the small {}.',
        'a photo of the large {}.', 'a black and white photo of a {}.',
        'a dark photo of a {}.', 'a photo of a cool {}.', 'a photo of a small {}.',
        'there is a {} in the scene.', 'there is the {} in the scene.',
        'this is a {} in the scene.', 'this is the {} in the scene.',
        'this is one {} in the scene.'
    ]

    sentences: List[str] = []
    for state_list in prompt_state:
        for s in state_list:
            for t in templates:
                sentences.append(t.format(s.format(obj)))

    batch_size = 128
    feats = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]
            tokens = tokenizer(batch).to(device)
            text_feat = model.encode_text(tokens)          # (B, C)
            feats.append(text_feat)
    all_feats = torch.cat(feats, dim=0)                    # (N, C)
    mean_feat = all_feats.mean(dim=0, keepdim=True)
    mean_feat = mean_feat / mean_feat.norm(dim=-1, keepdim=True)
    return mean_feat                                        # (1, C)


# -------------------------------------------------------------------------
# 6. Test loop
# -------------------------------------------------------------------------
def test(args, clip_model, preprocess, test_loader, text_feature, device):
    gt_list = []
    image_scores = []
    seg_present = CLASS_INDEX[args.obj] > 0
    gt_mask_list = []
    seg_scores_list = []

    clip_model.eval()
    with torch.no_grad():
        for images, labels, masks in tqdm(test_loader, desc="Testing"):
            images = images.to(device)                     # (B,3,H,W)

            # ----- image features -----
            img_feats = clip_model.encode_image(images)
            img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)

            # ----- cosine similarity -----
            sim = (img_feats @ text_feature.t()).squeeze(1)   # (B,)
            image_scores.extend(sim.cpu().numpy().tolist())
            gt_list.extend(labels.numpy().tolist())

            # ----- placeholder segmentation maps -----
            if seg_present:
                B = images.size(0)
                for _ in range(B):
                    seg_scores_list.append(np.zeros((args.img_size, args.img_size)))
                    gt_mask_list.append(masks[_].cpu().numpy().squeeze())

    # ----- image-level AUC -----
    gt_arr = np.array(gt_list)
    scores_arr = np.array(image_scores)
    if scores_arr.max() > scores_arr.min():
        scores_norm = (scores_arr - scores_arr.min()) / (scores_arr.max() - scores_arr.min())
    else:
        scores_norm = scores_arr
    img_auc = roc_auc_score(gt_arr, scores_norm)
    print(f'{args.obj} Image-level AUC : {img_auc:.4f}')

    # ----- pixel-level AUC (if masks exist) -----
    if seg_present:
        gt_masks = np.concatenate([m.ravel() for m in gt_mask_list])
        pred_masks = np.concatenate([s.ravel() for s in seg_scores_list])
        if pred_masks.max() > pred_masks.min():
            pred_norm = (pred_masks - pred_masks.min()) / (pred_masks.max() - pred_masks.min())
        else:
            pred_norm = pred_masks
        seg_auc = roc_auc_score(gt_masks, pred_norm)
        print(f'{args.obj} Pixel-level AUC : {seg_auc:.4f}')
        return img_auc + seg_auc
    return img_auc


# -------------------------------------------------------------------------
# 7. Main
# -------------------------------------------------------------------------
def main():
    args = parse_args()
    setup_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Running on device:", device)
    print("Arguments:", args)

    # ----- download checkpoints (if not present) -----
    ckpt_dir = "checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)

    repo_id = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    try:
        hf_hub_download(repo_id=repo_id,
                        filename="open_clip_pytorch_model.bin",
                        local_dir=ckpt_dir)
        hf_hub_download(repo_id=repo_id,
                        filename="open_clip_config.json",
                        local_dir=ckpt_dir)
    except Exception as e:
        print("Warning: could not download from HF hub (maybe already cached):", e)

    # ----- load optional config -----
    model_name_local = "biomedclip_local"
    cfg_path = os.path.join(ckpt_dir, "open_clip_config.json")
    model_cfg = preprocess_cfg = None
    if os.path.isfile(cfg_path):
        with open(cfg_path) as f:
            cfg = json.load(f)
            model_cfg = cfg.get("model_cfg")
            preprocess_cfg = cfg.get("preprocess_cfg")
    if model_name_local not in _MODEL_CONFIGS and model_cfg is not None:
        _MODEL_CONFIGS[model_name_local] = model_cfg

    # ----- build model & tokenizer -----
    tokenizer = get_tokenizer(model_name_local)
    clip_model, _, preprocess = create_model_and_transforms(
        model_name=model_name_local,
        pretrained=os.path.join(ckpt_dir, "open_clip_pytorch_model.bin"),
        **({f"image_{k}": v for k, v in (preprocess_cfg or {}).items()})
    )
    clip_model.to(device)

    # ----- dataset & loader -----
    test_set = MedTestDataset(args.data_path, args.obj, resize=args.img_size)
    kwargs = {'num_workers': 4, 'pin_memory': True} if torch.cuda.is_available() else {}
    test_loader = DataLoader(test_set, batch_size=args.batch_size,
                             shuffle=False, **kwargs)
    print(f"Test dataset size: {len(test_set)}")

    # ----- text prompt embedding -----
    obj_name = REAL_NAME[CLASS_INDEX[args.obj]]
    text_feature = encode_text_with_prompt_ensemble(clip_model, tokenizer,
                                                    obj_name, device)

    # ----- run test -----
    _ = test(args, clip_model, preprocess, test_loader, text_feature, device)


if __name__ == '__main__':
    main()