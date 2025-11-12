import os
import warnings
import random
import argparse
from loss import FocalLoss, BinaryDiceLoss
from medical_zero import MedTestDataset


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def parse_args():
    parser = argparse.ArgumentParser(description='BiomedCLIP Testing')
    # General defaults
    parser.add_argument('--model_name', type=str, default='BiomedCLIP-PubMedBERT-ViT-B-16',
                        help="BiomedCLIP model version")    
    parser.add_argument('--text_encoder', type=str, default='microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext',
                        help="Text encoder used for BiomedCLIP" )

    parser.add_argument('--pretrain', type=str, default='microsoft',
                            help="pretrained checkpoint source")
    parser.add_argument('--obj', type=str, default='Liver')
    parser.add_argument('--data_path', type=str, default='./data/',
                        help="path to dataset"  )
    #parser.add_argument('--data_path', type=str, default='/kaggle/input/preprocessed/Liver')

    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--save_model', type=int, default=1)
    #parser.add_argument('--save_path', type=str, default='./ckpt/few-shot/')
    parser.add_argument('--img_size', type=int, default=224, 
                        help="BiomedCLIP trained with 224x224 resolution")

    parser.add_argument("--epoch", type=int, default=50)
    #parser.add_argument("--learning_rate", type=float, default=0.001)
    #parser.add_argument("--features_list", type=int, nargs="+", default=[3, 6, 9, 12],
                        help="layer features used for adapters")    
    parser.add_argument('--seed', type=int, default=111)
    #parser.add_argument('--shot', type=int, default=4)
    parser.add_argument('--iterate', type=int, default=0)
    args, _ = parser.parse_known_args()

    return args


def main():
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    os.environ["TRANSFORMERS_NO_ADDITIONAL_CHAT_TEMPLATES"] = "1"
    args = parse_args()

    # Save args in global_vars and globals for convenience
    global_vars.update(vars(args))
    global_vars['args'] = args
    for k, v in global_vars.items():
        globals()[k] = v

    # Print device and parsed args (optional)
    print(f"Using device: {device}")
    print("Parsed arguments:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")

    # set seed
    setup_seed(args.seed)



class MedTestDataset(Dataset):
    def __init__(self,
                 dataset_path='/data/',
                 class_name='Brain',
                 resize=240
                 ):
        assert class_name in CLASS_NAMES, 'class_name: {}, should be in {}'.format(class_name, CLASS_NAMES)

        self.dataset_path = os.path.join(dataset_path, f'{class_name}_AD')
        self.class_name = class_name
        self.seg_flag = CLASS_INDEX[class_name]
        self.resize = resize

        # load dataset
        self.x, self.y, self.mask = self.load_dataset_folder(self.seg_flag)

        # set transforms
        self.transform_x = transforms.Compose([
            transforms.Resize((resize,resize), Image.BICUBIC),
            transforms.ToTensor(),
        ])

        self.transform_mask = transforms.Compose(
            [transforms.Resize((resize,resize), Image.NEAREST),
             transforms.ToTensor()])


    def __getitem__(self, idx):
        x, y, mask = self.x[idx], self.y[idx], self.mask[idx]
        x = Image.open(x).convert('RGB')
        x_img = self.transform_x(x)

        if self.seg_flag < 0:
            return x_img, y, torch.zeros([1, self.resize, self.resize])

        if mask is None:
            mask = torch.zeros([1, self.resize, self.resize])
            y = 0
        else:
            mask = Image.open(mask).convert('L')
            mask = self.transform_mask(mask)
            y = 1
        return x_img, y, mask

    def __len__(self):
        return len(self.x)

    def load_dataset_folder(self, seg_flag):
        x, y, mask = [], [], []

        normal_img_dir = os.path.join(self.dataset_path, 'test/good/img')
        img_fpath_list = sorted([os.path.join(normal_img_dir, f) for f in os.listdir(normal_img_dir)])
        x.extend(img_fpath_list)
        y.extend([0] * len(img_fpath_list))
        mask.extend([None] * len(img_fpath_list))

        abnorm_img_dir = os.path.join(self.dataset_path, 'test/Ungood/img')
        img_fpath_list = sorted([os.path.join(abnorm_img_dir, f) for f in os.listdir(abnorm_img_dir)])
        x.extend(img_fpath_list)
        y.extend([1] * len(img_fpath_list))

        if seg_flag > 0:
            gt_type_dir = os.path.join(self.dataset_path, 'test/Ungood/anomaly_mask')
            gt_fpath_list = sorted([os.path.join(gt_type_dir, f) for f in os.listdir(gt_type_dir)])
            mask.extend(gt_fpath_list)
        else:
            mask.extend([None] * len(img_fpath_list))

        assert len(x) == len(y), 'number of x and y should be same'
        return list(x), list(y), list(mask)

    # load dataset and loader
    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
    test_dataset = MedTestDataset(args.data_path, args.obj, args.img_size)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, **kwargs)
    print(f"Test dataset size: {len(test_dataset)}")



import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import exp

class FocalLoss(nn.Module):
    """
    copy from: https://github.com/Hsuxu/Loss_ToolBox-PyTorch/blob/master/FocalLoss/FocalLoss.py
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)*log(pt)
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param smooth: (float,double) smooth value when cross entropy
    :param balance_index: (int) balance class index, should be specific when alpha is float
    :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
    """

    def __init__(self, apply_nonlin=None, alpha=None, gamma=2, balance_index=0, smooth=1e-5, size_average=True):
        super(FocalLoss, self).__init__()
        self.apply_nonlin = apply_nonlin
        self.alpha = alpha
        self.gamma = gamma
        self.balance_index = balance_index
        self.smooth = smooth
        self.size_average = size_average

        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError('smooth value should be in [0,1]')

    def forward(self, logit, target):
        if self.apply_nonlin is not None:
            logit = self.apply_nonlin(logit)
        num_class = logit.shape[1]

        if logit.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
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
            one_hot_key = torch.clamp(
                one_hot_key, self.smooth / (num_class - 1), 1.0 - self.smooth)
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

        # losses
    loss_focal = FocalLoss()
    loss_dice = BinaryDiceLoss()
    loss_bce = torch.nn.BCEWithLogitsLoss()


import json

from urllib.request import urlopen
from PIL import Image
import torch
from huggingface_hub import hf_hub_download
from open_clip import create_model_and_transforms, get_tokenizer
from open_clip.factory import HF_HUB_PREFIX, _MODEL_CONFIGS


# Download the model and config files
hf_hub_download(
    repo_id="microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
    filename="open_clip_pytorch_model.bin",
    local_dir="checkpoints"
)
hf_hub_download(
    repo_id="microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
    filename="open_clip_config.json",
    local_dir="checkpoints"
)


# Load the model and config files
model_name = "biomedclip_local"

with open("checkpoints/open_clip_config.json", "r") as f:
    config = json.load(f)
    model_cfg = config["model_cfg"]
    preprocess_cfg = config["preprocess_cfg"]


if (not model_name.startswith(HF_HUB_PREFIX)
    and model_name not in _MODEL_CONFIGS
    and config is not None):
    _MODEL_CONFIGS[model_name] = model_cfg

tokenizer = get_tokenizer(model_name)

model, _, preprocess = create_model_and_transforms(
    model_name=model_name,
    pretrained="checkpoints/open_clip_pytorch_model.bin",
    **{f"image_{k}": v for k, v in preprocess_cfg.items()},
)


def encode_text_with_prompt_ensemble(model, obj, device):
    print("\n=== Starting encode_text_with_prompt_ensemble ===")
    print(f"Object to encode: {obj}")
    print(f"Using device: {device}")

    prompt_normal = ['{}', 'flawless {}', 'perfect {}', 'unblemished {}', 
                     '{} without flaw', '{} without defect', '{} without damage']
    prompt_abnormal = ['damaged {}', 'broken {}', '{} with flaw', 
                       '{} with defect', '{} with damage']
    prompt_state = [prompt_normal, prompt_abnormal]
    prompt_templates = ['a bad photo of a {}.', 'a low resolution photo of the {}.', 
                        'a bad photo of the {}.', 'a cropped photo of the {}.', 
                        'a bright photo of a {}.', 'a dark photo of the {}.', 
                        'a photo of my {}.', 'a photo of the cool {}.', 
                        'a close-up photo of a {}.', 'a black and white photo of the {}.', 
                        'a bright photo of the {}.', 'a cropped photo of a {}.', 
                        'a jpeg corrupted photo of a {}.', 'a blurry photo of the {}.', 
                        'a photo of the {}.', 'a good photo of the {}.', 
                        'a photo of one {}.', 'a close-up photo of the {}.', 
                        'a photo of a {}.', 'a low resolution photo of a {}.', 
                        'a photo of a large {}.', 'a blurry photo of a {}.', 
                        'a jpeg corrupted photo of the {}.', 'a good photo of a {}.', 
                        'a photo of the small {}.', 'a photo of the large {}.', 
                        'a black and white photo of a {}.', 'a dark photo of a {}.', 
                        'a photo of a cool {}.', 'a photo of a small {}.', 
                        'there is a {} in the scene.', 'there is the {} in the scene.', 
                        'this is a {} in the scene.', 'this is the {} in the scene.', 
                        'this is one {} in the scene.']
     
    text_features = []
    for i in range(len(prompt_state)):
        print(f"\n--- Processing prompt set {i + 1}/{len(prompt_state)} ---")

        prompted_state = [state.format(obj) for state in prompt_state[i]]
        print(f"Prompted state examples ({len(prompted_state)}): {prompted_state[:3]} ...")

        prompted_sentence = []
        for s in prompted_state:
            for template in prompt_templates:
                prompted_sentence.append(template.format(s))
        print(f"Total prompted sentences generated: {len(prompted_sentence)}")
        print(f"Example sentences: {prompted_sentence[:3]} ...")


text_feature_list = [0]
    # text prompt
    with torch.cuda.amp.autocast(), torch.no_grad():
      
            text_feature = encode_text_with_prompt_ensemble(clip_model, REAL_NAME[CLASS_INDEX_INV[i]], device)
            text_feature_list.append(text_feature)

    score = test(args, model, test_loader, text_feature_list[CLASS_INDEX[args.obj]])
        


def test(args, seg_model, test_loader, text_features):
    gt_list = []
    gt_mask_list = []
    image_scores = []
    segment_scores = []
    
    for (image, y, mask) in tqdm(test_loader):
        image = image.to(device)
        mask[mask > 0.5], mask[mask <= 0.5] = 1, 0

        with torch.no_grad():
            image_features, text_features, logit_scale = model(images, texts)
            
            # image
            anomaly_score = 0
            patch_tokens = image_features.copy()
            for layer in range(len(patch_tokens)):
                patch_tokens[layer] /= patch_tokens[layer].norm(dim=-1, keepdim=True)
                anomaly_map = (100.0 * patch_tokens[layer] @ text_features).unsqueeze(0)
                anomaly_map = torch.softmax(anomaly_map, dim=-1)[:, :, 1]
                anomaly_score += anomaly_map.mean()
            image_scores.append(anomaly_score.cpu())

            # pixel
            patch_tokens = ori_seg_patch_tokens
            anomaly_maps = []
            for layer in range(len(patch_tokens)):
                patch_tokens[layer] /= patch_tokens[layer].norm(dim=-1, keepdim=True)
                anomaly_map = (100.0 * patch_tokens[layer] @ text_features).unsqueeze(0)
                B, L, C = anomaly_map.shape
                H = int(np.sqrt(L))
                anomaly_map = F.interpolate(anomaly_map.permute(0, 2, 1).view(B, 2, H, H),
                                            size=args.img_size, mode='bilinear', align_corners=True)
                anomaly_map = torch.softmax(anomaly_map, dim=1)[:, 1, :, :]
                anomaly_maps.append(anomaly_map.cpu().numpy())
            final_score_map = np.sum(anomaly_maps, axis=0)
            
            gt_mask_list.append(mask.squeeze().cpu().detach().numpy())
            gt_list.extend(y.cpu().detach().numpy())
            segment_scores.append(final_score_map)
        
        

    gt_list = np.array(gt_list)
    gt_mask_list = np.asarray(gt_mask_list)
    gt_mask_list = (gt_mask_list>0).astype(np.int_)

    segment_scores = np.array(segment_scores)
    image_scores = np.array(image_scores)

    segment_scores = (segment_scores - segment_scores.min()) / (segment_scores.max() - segment_scores.min())
    image_scores = (image_scores - image_scores.min()) / (image_scores.max() - image_scores.min())

    img_roc_auc_det = roc_auc_score(gt_list, image_scores)
    print(f'{args.obj} AUC : {round(img_roc_auc_det,4)}')

    if CLASS_INDEX[args.obj] > 0:
        seg_roc_auc = roc_auc_score(gt_mask_list.flatten(), segment_scores.flatten())
        print(f'{args.obj} pAUC : {round(seg_roc_auc,4)}')
        return seg_roc_auc + img_roc_auc_det
    else:
        return img_roc_auc_det

if __name__ == '__main__':
    main()


