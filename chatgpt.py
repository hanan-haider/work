#!/usr/bin/env python3
# corrected_biomedclip_test.py

import os
import random
import argparse
import json
from math import sqrt
from typing import List

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

# open_clip related
from huggingface_hub import hf_hub_download
from open_clip import create_model_and_transforms, get_tokenizer
from open_clip.factory import HF_HUB_PREFIX, _MODEL_CONFIGS

# ---------------------------
# Constants / small mappings
# ---------------------------
# Replace/extend these with the full lists from BMAD if you have them
CLASS_NAMES = ['Liver']                # example; extend as needed
CLASS_INDEX = {'Liver': 1}             # seg_flag > 0 indicates segmentation is available
CLASS_INDEX_INV = {v: k for k, v in CLASS_INDEX.items()}
REAL_NAME = {1: 'liver'}               # human-readable object names if needed

# ---------------------------
# Losses
# ---------------------------
class FocalLoss(nn.Module):
    def __init__(self, apply_nonlin=None, alpha=None, gamma=2, balance_index=0, smooth=1e-5, size_average=True):
        super(FocalLoss, self).__init__()
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

        target = torch.squeeze(target, 1)
        target = target.view(-1, 1)
        alpha = self.alpha

        if alpha is None:
            alpha = torch.ones(num_class, 1)
        elif isinstance(alpha, (list, np.ndarray)):
            assert len(alpha) == num_class
            alpha = torch.FloatTensor(alpha).view(num_class, 1)
            alpha = alpha / alpha.sum()
        elif isinstance(alpha, float):
            alpha = torch.ones(num_class, 1)
            alpha = alpha * (1 - self.alpha)
            alpha[self.balance_index] = self.alpha
        else:
            raise TypeError('Not support alpha type')

        if alpha.device != logit.device:
            alpha = alpha.to(logit.device)

        idx = target.cpu().long()
        one_hot_key = torch.FloatTensor(target.size(0), num_class).zero_()
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        if one_hot_key.device != logit.device:
            one_hot_key = one_hot_key.to(logit.device)

        if self.smooth:
            one_hot_key = torch.clamp(one_hot_key, self.smooth / (num_class - 1), 1.0 - self.smooth)
        pt = (one_hot_key * logit).sum(1) + self.smooth
        logpt = pt.log()
        gamma = self.gamma

        alpha = alpha[idx]
        alpha = torch.squeeze(alpha)
        loss = -1 * alpha * torch.pow((1 - pt), gamma) * logpt

        if self.size_average:
            loss = loss.mean()
        return loss


class BinaryDiceLoss(nn.Module):
    def __init__(self):
        super(BinaryDiceLoss, self).__init__()

    def forward(self, input, targets):
        N = targets.size()[0]
        smooth = 1
        input_flat = input.view(N, -1)
        targets_flat = targets.view(N, -1)
        intersection = input_flat * targets_flat
        N_dice_eff = (2 * intersection.sum(1) + smooth) / (input_flat.sum(1) + targets_flat.sum(1) + smooth)
        loss = 1 - N_dice_eff.sum() / N
        return loss


# loss instances (if you need)
loss_focal = FocalLoss()
loss_dice = BinaryDiceLoss()
loss_bce = torch.nn.BCEWithLogitsLoss()

# ---------------------------
# Dataset
# ---------------------------
class MedTestDataset(Dataset):
    def __init__(self, dataset_path: str = './data/', class_name: str = 'Liver', resize: int = 224):
        assert class_name in CLASS_NAMES, f'class_name: {class_name}, should be in {CLASS_NAMES}'
        self.dataset_path = os.path.join(dataset_path, f'{class_name}_AD')
        self.class_name = class_name
        self.seg_flag = CLASS_INDEX[class_name]
        self.resize = resize

        # load dataset lists
        self.x, self.y, self.mask = self.load_dataset_folder(self.seg_flag)

        # transforms
        self.transform_x = transforms.Compose([
            transforms.Resize((resize, resize), Image.BICUBIC),
            transforms.ToTensor(),
        ])
        self.transform_mask = transforms.Compose([
            transforms.Resize((resize, resize), Image.NEAREST),
            transforms.ToTensor(),
        ])

    def __getitem__(self, idx):
        x_path, y_label, mask_path = self.x[idx], self.y[idx], self.mask[idx]
        x = Image.open(x_path).convert('RGB')
        x_img = self.transform_x(x).float()
        x_img = self.transform_x(x)

        if self.seg_flag <= 0:
            # segmentation not available
            return x_img, torch.tensor(y, dtype=torch.float32), torch.zeros([1, self.resize, self.resize], dtype=torch.float32)

        if mask_path is None:
            mask = torch.zeros([1, self.resize, self.resize], dtype=torch.float32)
            y = torch.tensor(0.0, dtype=torch.float32)
        
        else:
            mask = Image.open(mask_path).convert('L')
            mask = self.transform_mask(mask).float()
            y = torch.tensor(1.0, dtype=torch.float32)
            mask = self.transform_mask(mask)
            y = 1
        return x_img, y, mask

    def __len__(self):
        return len(self.x)

    def load_dataset_folder(self, seg_flag):
        x, y, mask = [], [], []
        # normal images
        normal_img_dir = os.path.join(self.dataset_path, 'test/good/img')
        if os.path.isdir(normal_img_dir):
            img_fpath_list = sorted([os.path.join(normal_img_dir, f) for f in os.listdir(normal_img_dir)])
            x.extend(img_fpath_list)
            y.extend([0] * len(img_fpath_list))
            mask.extend([None] * len(img_fpath_list))
        # abnormal images
        abnorm_img_dir = os.path.join(self.dataset_path, 'test/Ungood/img')
        if os.path.isdir(abnorm_img_dir):
            img_fpath_list = sorted([os.path.join(abnorm_img_dir, f) for f in os.listdir(abnorm_img_dir)])
            x.extend(img_fpath_list)
            y.extend([1] * len(img_fpath_list))
        # masks (if segmentation available)
        if seg_flag > 0:
            gt_type_dir = os.path.join(self.dataset_path, 'test/Ungood/anomaly_mask')
            if os.path.isdir(gt_type_dir):
                gt_fpath_list = sorted([os.path.join(gt_type_dir, f) for f in os.listdir(gt_type_dir)])
                # if there are fewer masks than images, pad with None
                if len(gt_fpath_list) < len([p for p in x if '/Ungood/' in p]):
                    gt_fpath_list = (gt_fpath_list + [None] * len(x))[:len(x)]
                mask.extend(gt_fpath_list)
            else:
                mask.extend([None] * len(x))
        else:
            # for datasets without segmentation
            mask.extend([None] * len(x))

        assert len(x) == len(y) == len(mask), "x,y,mask must be same length"
        return x, y, mask

# ---------------------------
# Utility functions
# ---------------------------
def setup_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser(description='BiomedCLIP Testing')
    parser.add_argument('--model_name', type=str, default='BiomedCLIP-PubMedBERT-ViT-B-16')
    parser.add_argument('--text_encoder', type=str, default='microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext')
    parser.add_argument('--pretrain', type=str, default='microsoft')
    parser.add_argument('--obj', type=str, default='Liver')
    parser.add_argument('--data_path', type=str, default='./data/')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--seed', type=int, default=111)
    parser.add_argument('--save_model', type=int, default=0)
    args = parser.parse_args()
    return args

# ---------------------------
# Text prompt ensemble encoding
# ---------------------------
def encode_text_with_prompt_ensemble(model, tokenizer, obj: str, device: torch.device) -> torch.Tensor:
    """
    Build many prompt permutations and encode them into a single averaged (and normalized) embedding.
    Returns a 1-D torch tensor on `device`.
    """
    prompt_normal = ['{}', 'flawless {}', 'perfect {}', 'unblemished {}', '{} without flaw', '{} without defect', '{} without damage']
    prompt_abnormal = ['damaged {}', 'broken {}', '{} with flaw', '{} with defect', '{} with damage']
    prompt_state = [prompt_normal, prompt_abnormal]

    prompt_templates = [
        'a bad photo of a {}.', 'a low resolution photo of the {}.', 'a bad photo of the {}.',
        'a cropped photo of the {}.', 'a bright photo of a {}.', 'a dark photo of the {}.',
        'a photo of my {}.', 'a photo of the cool {}.', 'a close-up photo of a {}.',
        'a black and white photo of the {}.', 'a bright photo of the {}.', 'a cropped photo of a {}.',
        'a jpeg corrupted photo of a {}.', 'a blurry photo of the {}.', 'a photo of the {}.',
        'a good photo of the {}.', 'a photo of one {}.', 'a close-up photo of the {}.',
        'a photo of a {}.', 'a low resolution photo of a {}.', 'a photo of a large {}.',
        'a blurry photo of a {}.', 'a jpeg corrupted photo of the {}.', 'a good photo of a {}.',
        'a photo of the small {}.', 'a photo of the large {}.', 'a black and white photo of a {}.',
        'a dark photo of a {}.', 'a photo of a cool {}.', 'a photo of a small {}.',
        'there is a {} in the scene.', 'there is the {} in the scene.', 'this is a {} in the scene.',
        'this is the {} in the scene.', 'this is one {} in the scene.'
    ]

    all_sentences: List[str] = []
    for state_list in prompt_state:
        prompted_state = [s.format(obj) for s in state_list]
        for s in prompted_state:
            for t in prompt_templates:
                all_sentences.append(t.format(s))

    # Tokenize in batches to avoid giant tensors
    batch_size = 128
    encoded_feats = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(all_sentences), batch_size):
            batch_sentences = all_sentences[i:i+batch_size]
            tokens = tokenizer(batch_sentences).to(device)  # open_clip tokenizer returns token tensor
            text_feats = model.encode_text(tokens)          # (B, C)
            encoded_feats.append(text_feats)

    encoded_feats = torch.cat(encoded_feats, dim=0)      # (N_prompts, C)
    # average and normalize
    text_feature = encoded_feats.mean(dim=0, keepdim=True)   # (1, C)
    text_feature = text_feature / text_feature.norm(dim=-1, keepdim=True)
    return text_feature  # (1, C) on device


# ---------------------------
# Test / evaluation (simplified)
# ---------------------------
def test(args, clip_model, preprocess, test_loader, text_feature, device):
    """
    Simplified test loop using global image embedding vs text embedding cosine similarity.
    If you have patch-level features available from your CLIP variant, replace the scoring
    below with patch-level anomaly-map generation.
    """
    gt_list = []
    image_scores = []
    seg_present = CLASS_INDEX[args.obj] > 0
    gt_mask_list = []
    segment_scores = []

    clip_model.eval()
    with torch.no_grad():
        for images, y_labels, masks in tqdm(test_loader, desc="Testing"):
            images = images.to(device)                   # (B, 3, H, W)
            # get pooled image features
            image_feats = clip_model.encode_image(images)   # shape: (B, C)
            image_feats = image_feats / image_feats.norm(dim=-1, keepdim=True)

            # compute similarity between image and prompt text_feature
            # text_feature shape: (1, C)
            # compute cosine: (B, 1)
            sims = (image_feats @ text_feature.t()).squeeze(1)  # (B,)
            # convert to score range 0-1 using sigmoid or min-max later
            image_scores.extend(sims.cpu().numpy().tolist())
            gt_list.extend(y_labels.numpy().tolist())

            # For segmentation-level map: placeholder - we do not produce per-pixel maps here
            if seg_present:
                # If you have per-patch tokens you can build an anomaly map per image here.
                # We append a dummy zero mask (or use upsampled similarity map if available).
                batch_size = images.shape[0]
                for _ in range(batch_size):
                    segment_scores.append(np.zeros((args.img_size, args.img_size)))
                    gt_mask_list.append(np.zeros((args.img_size, args.img_size)))

    # convert to numpy arrays
    gt_arr = np.array(gt_list)
    image_scores = np.array(image_scores)
    # normalize image scores to [0,1]
    if image_scores.max() > image_scores.min():
        image_scores_norm = (image_scores - image_scores.min()) / (image_scores.max() - image_scores.min())
    else:
        image_scores_norm = image_scores

    img_roc_auc_det = roc_auc_score(gt_arr, image_scores_norm)
    print(f'{args.obj} Image-level AUC : {round(img_roc_auc_det, 4)}')

    if seg_present:
        # seg ROC requires flattening real maps and predicted maps
        gt_masks = np.asarray(gt_mask_list).astype(int).flatten()
        seg_scores = np.asarray(segment_scores).flatten()
        if seg_scores.max() > seg_scores.min():
            seg_scores_norm = (seg_scores - seg_scores.min()) / (seg_scores.max() - seg_scores.min())
        else:
            seg_scores_norm = seg_scores
        seg_roc_auc = roc_auc_score(gt_masks, seg_scores_norm)
        print(f'{args.obj} Pixel-level AUC : {round(seg_roc_auc, 4)}')
        return img_roc_auc_det + seg_roc_auc
    else:
        return img_roc_auc_det


# ---------------------------
# Main
# ---------------------------
def main():
    args = parse_args()
    setup_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_cuda = torch.cuda.is_available()

    print("Running on device:", device)
    print("Arguments:", args)

    # ---------------------------
    # Ensure checkpoints dir and download config/weights if needed (example)
    # ---------------------------
    ckpt_dir = "checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)

    # Example of downloading a repo asset from HF hub that you used previously.
    # You can skip these downloads if you already have them locally.
    # NOTE: Provide correct repo_id and filenames for your model. Adjust as necessary.
    repo_id = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    try:
        hf_hub_download(repo_id=repo_id, filename="open_clip_pytorch_model.bin", local_dir=ckpt_dir)
        hf_hub_download(repo_id=repo_id, filename="open_clip_config.json", local_dir=ckpt_dir)
    except Exception as e:
        print("Warning: problem downloading assets from HF hub (maybe already present).", e)

    # load config if present (optional)
    model_name_local = "biomedclip_local"
    config_path = os.path.join(ckpt_dir, "open_clip_config.json")
    model_cfg = None
    preprocess_cfg = None
    if os.path.isfile(config_path):
        with open(config_path, "r") as f:
            cfg = json.load(f)
            model_cfg = cfg.get("model_cfg", None)
            preprocess_cfg = cfg.get("preprocess_cfg", None)

    if (not model_name_local.startswith(HF_HUB_PREFIX)
        and model_name_local not in _MODEL_CONFIGS
        and model_cfg is not None):
        _MODEL_CONFIGS[model_name_local] = model_cfg

    # get tokenizer and create model
    tokenizer = get_tokenizer(model_name_local)
    clip_model, _, preprocess = create_model_and_transforms(
        model_name=model_name_local,
        pretrained=os.path.join(ckpt_dir, "open_clip_pytorch_model.bin"),
        **({f"image_{k}": v for k, v in (preprocess_cfg or {}).items()})
    )
    clip_model.to(device)

    # dataset and loader
    test_dataset = MedTestDataset(args.data_path, args.obj, resize=args.img_size)
    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, **kwargs)
    print(f"Test dataset size: {len(test_dataset)}")

    # prepare text prompt embedding for the object
    text_feature = encode_text_with_prompt_ensemble(clip_model, tokenizer, REAL_NAME[CLASS_INDEX[args.obj]], device)

    # run test
    _ = test(args, clip_model, preprocess, test_loader, text_feature, device)


if __name__ == '__main__':
    main()
