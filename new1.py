import os
import warnings
import random
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, auc
from scipy.ndimage import label, generate_binary_structure
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from huggingface_hub import hf_hub_download
from open_clip import create_model_and_transforms, get_tokenizer
from open_clip.factory import HF_HUB_PREFIX, _MODEL_CONFIGS

warnings.filterwarnings('ignore')

# ==================== Global Variables ====================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
use_cuda = torch.cuda.is_available()

CLASS_NAMES = ['Brain', 'Liver', 'Retina_RESC', 'Retina_OCT2017', 'Chest', 'Histopathology']
CLASS_INDEX = {
    'Brain': 1,
    'Liver': 1,
    'Retina_RESC': 1,
    'Retina_OCT2017': 1,
    'Chest': 1,
    'Histopathology': 1
}
CLASS_INDEX_INV = {v: k for k, v in CLASS_INDEX.items()}
REAL_NAME = {
    'Brain': 'brain',
    'Liver': 'liver',
    'Retina_RESC': 'retina',
    'Retina_OCT2017': 'retina',
    'Chest': 'chest',
    'Histopathology': 'tissue'
}


# ==================== Loss Functions ====================
class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
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
    """Dice Loss for segmentation"""
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


# ==================== Dataset Class ====================
class MedTestDataset(Dataset):
    """Medical Image Test Dataset"""
    def __init__(self, dataset_path='./data/', class_name='Brain', resize=224):
        assert class_name in CLASS_NAMES, f'class_name: {class_name}, should be in {CLASS_NAMES}'

        self.dataset_path = os.path.join(dataset_path, f'{class_name}_AD')
        self.class_name = class_name
        self.seg_flag = CLASS_INDEX[class_name]
        self.resize = resize

        # Load dataset
        self.x, self.y, self.mask = self.load_dataset_folder(self.seg_flag)

        # Set transforms
        self.transform_x = transforms.Compose([
            transforms.Resize((resize, resize), Image.BICUBIC),
            transforms.ToTensor(),
        ])

        self.transform_mask = transforms.Compose([
            transforms.Resize((resize, resize), Image.NEAREST),
            transforms.ToTensor()
        ])

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

        # Load normal images
        normal_img_dir = os.path.join(self.dataset_path, 'test/good/img')
        if os.path.exists(normal_img_dir):
            img_fpath_list = sorted([os.path.join(normal_img_dir, f) 
                                    for f in os.listdir(normal_img_dir) 
                                    if f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
            x.extend(img_fpath_list)
            y.extend([0] * len(img_fpath_list))
            mask.extend([None] * len(img_fpath_list))

        # Load abnormal images
        abnorm_img_dir = os.path.join(self.dataset_path, 'test/Ungood/img')
        if os.path.exists(abnorm_img_dir):
            img_fpath_list = sorted([os.path.join(abnorm_img_dir, f) 
                                    for f in os.listdir(abnorm_img_dir)
                                    if f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
            x.extend(img_fpath_list)
            y.extend([1] * len(img_fpath_list))

            # Load masks if available
            if seg_flag > 0:
                gt_type_dir = os.path.join(self.dataset_path, 'test/Ungood/anomaly_mask')
                if os.path.exists(gt_type_dir):
                    gt_fpath_list = sorted([os.path.join(gt_type_dir, f) 
                                           for f in os.listdir(gt_type_dir)
                                           if f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
                    mask.extend(gt_fpath_list)
                else:
                    mask.extend([None] * len(img_fpath_list))
            else:
                mask.extend([None] * len(img_fpath_list))

        assert len(x) == len(y), 'number of x and y should be same'
        print(f"Loaded {len(x)} images: {y.count(0)} normal, {y.count(1)} abnormal")
        return list(x), list(y), list(mask)


# ==================== Utility Functions ====================
def setup_seed(seed):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='BiomedCLIP Testing')
    parser.add_argument('--model_name', type=str, default='BiomedCLIP-PubMedBERT-ViT-B-16',
                        help="BiomedCLIP model version")
    parser.add_argument('--text_encoder', type=str, 
                        default='microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext',
                        help="Text encoder used for BiomedCLIP")
    parser.add_argument('--pretrain', type=str, default='microsoft',
                        help="Pretrained checkpoint source")
    parser.add_argument('--obj', type=str, default='Liver',
                        help="Object/organ to test")
    parser.add_argument('--data_path', type=str, default='./data/',
                        help="Path to dataset")
    parser.add_argument('--batch_size', type=int, default=1,
                        help="Batch size")
    parser.add_argument('--save_model', type=int, default=1)
    parser.add_argument('--img_size', type=int, default=224,
                        help="Image size (BiomedCLIP uses 224x224)")
    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument('--features_list', type=int, nargs="+", default=[3, 6, 9, 12],
                        help="Layer features for multi-scale")
    parser.add_argument('--seed', type=int, default=111)
    parser.add_argument('--iterate', type=int, default=0)
    parser.add_argument('--save_results', type=str, default='./results/',
                        help="Path to save results")
    
    args = parser.parse_args()
    return args


def load_biomedclip_model():
    """Load BiomedCLIP model"""
    print("\n=== Loading BiomedCLIP Model ===")
    
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Download model files
    hf_hub_download(
        repo_id="microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        filename="open_clip_pytorch_model.bin",
        local_dir=checkpoint_dir
    )
    hf_hub_download(
        repo_id="microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        filename="open_clip_config.json",
        local_dir=checkpoint_dir
    )
    
    # Load config
    model_name = "biomedclip_local"
    with open(os.path.join(checkpoint_dir, "open_clip_config.json"), "r") as f:
        config = json.load(f)
        model_cfg = config["model_cfg"]
        preprocess_cfg = config["preprocess_cfg"]
    
    # Register config
    if (not model_name.startswith(HF_HUB_PREFIX)
        and model_name not in _MODEL_CONFIGS
        and config is not None):
        _MODEL_CONFIGS[model_name] = model_cfg
    
    tokenizer = get_tokenizer(model_name)
    
    model, _, preprocess = create_model_and_transforms(
        model_name=model_name,
        pretrained=os.path.join(checkpoint_dir, "open_clip_pytorch_model.bin"),
        **{f"image_{k}": v for k, v in preprocess_cfg.items()},
    )
    
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded on {device}")
    return model, tokenizer, preprocess


def encode_text_with_prompt_ensemble(model, tokenizer, obj, device):
    """Encode text with prompt ensemble"""
    print(f"\n=== Encoding text prompts for: {obj} ===")
    
    prompt_normal = ['{}', 'flawless {}', 'perfect {}', 'unblemished {}',
                     '{} without flaw', '{} without defect', '{} without damage']
    prompt_abnormal = ['damaged {}', 'broken {}', '{} with flaw',
                       '{} with defect', '{} with damage']
    prompt_state = [prompt_normal, prompt_abnormal]
    
    prompt_templates = [
        'a bad photo of a {}.', 'a low resolution photo of the {}.',
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
        'this is one {} in the scene.'
    ]
    
    text_features_list = []
    
    for i in range(len(prompt_state)):
        print(f"Processing prompt set {i + 1}/{len(prompt_state)}")
        
        prompted_state = [state.format(obj) for state in prompt_state[i]]
        
        prompted_sentences = []
        for s in prompted_state:
            for template in prompt_templates:
                prompted_sentences.append(template.format(s))
        
        print(f"  Generated {len(prompted_sentences)} prompts")
        
        # Tokenize and encode
        with torch.no_grad(), torch.cuda.amp.autocast():
            text_tokens = tokenizer(prompted_sentences).to(device)
            text_features = model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            text_features = text_features.mean(dim=0)
            text_features = text_features / text_features.norm()
        
        text_features_list.append(text_features)
    
    # Stack features [normal, abnormal]
    text_features = torch.stack(text_features_list, dim=0)
    print(f"Text features shape: {text_features.shape}")
    
    return text_features


def compute_pro_metric(masks, scores, num_th=200):
    """Compute Per-Region Overlap (PRO) metric"""
    if masks.ndim == 2:
        masks = masks[np.newaxis, ...]
        scores = scores[np.newaxis, ...]
    
    binary_structure = generate_binary_structure(2, 2)
    pros_mean = []
    threds = np.linspace(0, 1, num_th)
    
    for thred in threds:
        binary_score_maps = (scores >= thred).astype(np.int32)
        
        pro_list = []
        for binary_mask, binary_score_map in zip(masks, binary_score_maps):
            if binary_mask.sum() == 0:
                continue
            
            labeled_mask, num_components = label(binary_mask, structure=binary_structure)
            
            for component_id in range(1, num_components + 1):
                component_mask = (labeled_mask == component_id)
                intersection = (component_mask & binary_score_map).sum()
                component_area = component_mask.sum()
                
                if component_area > 0:
                    pro_list.append(intersection / component_area)
        
        if len(pro_list) > 0:
            pros_mean.append(np.mean(pro_list))
        else:
            pros_mean.append(0)
    
    pros_mean = np.array(pros_mean)
    fpr_range = np.linspace(0, 0.3, len(threds))
    pro_auc = auc(fpr_range, pros_mean)
    
    return pro_auc


# ==================== REPLACE THE WHOLE test() FUNCTION ====================
def test(args, model, tokenizer, test_loader, text_features):
    """Test the model – works with BiomedCLIP (TimmModel)"""
    print("\n=== Starting Testing ===")

    gt_list = []
    gt_mask_list = []
    image_scores = []
    segment_scores = []

    model.eval()

    # -----------------------------------------------------------------
    # Hook to capture patch tokens from the selected transformer layers
    # -----------------------------------------------------------------
    patch_token_dict = {}          # layer_idx -> tensor (B, L, C)
    def get_hook(layer_idx):
        def hook(module, inp, out):
            # out shape: (B, L+1, C)  (L+1 because of CLS token)
            patch_token_dict[layer_idx] = out[:, 1:, :]   # remove CLS token
        return hook

    # Register hooks on the requested layers
    handles = []
    for i, block in enumerate(model.visual.transformer.resblocks):
        if (i + 1) in args.features_list:
            h = block.register_forward_hook(get_hook(i + 1))
            handles.append(h)

    # -----------------------------------------------------------------
    # Main testing loop
    # -----------------------------------------------------------------
    with torch.no_grad():
        for image, y, mask in tqdm(test_loader, desc="Testing"):
            image = image.to(device)                     # (B,3,H,W)
            mask_np = mask.cpu().numpy()
            mask_np = (mask_np > 0.5).astype(np.float32)

            # ----- image-level pooled embedding (CLS token) -----
            pooled_feat = model.encode_image(image)      # (_about, C)
            pooled_feat = pooled_feat / pooled_feat.norm(dim=-1, keepdim=True)

            # ----- image-level anomaly score -----
            sim = 100.0 * pooled_feat @ text_features.T          # (B, 2)
            anomaly_score = torch.softmax(sim, dim=-1)[:, 1].mean()
            image_scores.append(anomaly_score.cpu().item())
            gt_list.append(y.item())

            # ----- pixel-level anomaly map (only if we captured patch tokens) -----
            if args.features_list and patch_token_dict:
                anomaly_maps = []
                for layer_idx in args.features_list:
                    tokens = patch_token_dict[layer_idx]          # (B, L, C)
                    B, L, C = tokens.shape
                    H = int(np.sqrt(L))

                    tokens_norm = tokens / tokens.norm(dim=-1, keepdim=True)
                    similarity = 100.0 * tokens_norm @ text_features.T   # (B, L, 2)

                    # reshape to (B, 2, H, H) and upsample
                    sim_map = similarity.permute(0, 2, 1).view(B, 2, H, H)
                    sim_map = F.interpolate(
                        sim_map,
                        size=args.img_size,
                        mode='bilinear',
                        align_corners=True
                    )
                    anomaly_map = torch.softmax(sim_map, dim=1)[:, 1, :, :]   # (B, H, W)
                    anomaly_maps.append(anomaly_map.cpu().numpy())

                final_map = np.mean(anomaly_maps, axis=0)          # (B, H, W)
                segment_scores.append(final_map.squeeze())
                gt_mask_list.append(mask_np.squeeze())
            else:
                # fallback – no patch tokens (e.g. features_list empty)
                segment_scores.append(np.zeros((args.img_size, args.img_size)))
                gt_mask_list.append(mask_np.squeeze())

            # clear dict for next batch
            patch_token_dict.clear()

    # -----------------------------------------------------------------
    # Remove hooks
    # -----------------------------------------------------------------
    for h in handles:
        h.remove()

    # -----------------------------------------------------------------
    # Convert to numpy & normalize
    # -----------------------------------------------------------------
    gt_list = np.array(gt_list)
    image_scores = np.array(image_scores)

    if image_scores.max() > image_scores.min():
        image_scores = (image_scores - image_scores.min()) / (image_scores.max() - image_scores.min())

    # -----------------------------------------------------------------
    # Image-level AUC
    # -----------------------------------------------------------------
    img_roc_auc = roc_auc_score(gt_list, image_scores)
    print(f'{args.obj} Image AUC: {round(img_roc_auc, 4)}')

    # -----------------------------------------------------------------
    # Pixel-level metrics (if masks exist)
    # -----------------------------------------------------------------
    pixel_roc_auc = pro_score = 0.0
    if CLASS_INDEX[args.obj] > 0 and gt_list.sum() > 0:
        anomaly_idx = gt_list == 1
        gt_mask_anomaly = np.array(gt_mask_list)[anomaly_idx]
        seg_anomaly = np.array(segment_scores)[anomaly_idx]

        if gt_mask_anomaly.size > 0 and gt_mask_anomaly.sum() > 0:
            pixel_roc_auc = roc_auc_score(
                gt_mask_anomaly.flatten(),
                seg_anomaly.flatten()
            )
            print(f'{args.obj} Pixel AUC: {round(pixel_roc_auc, 4)}')

            try:
                pro_score = compute_pro_metric(gt_mask_anomaly, seg_anomaly)
                print(f'{args.obj} PRO: {round(pro_score, 4)}')
            except Exception as e:
                print(f"PRO computation failed: {e}")

    results = {
        'object': args.obj,
        'image_auc': float(img_roc_auc),
        'pixel_auc': float(pixel_roc_auc),
        'pro_score': float(pro_score)
    }

    return results

def main():
    """Main function"""
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    os.environ["TRANSFORMERS_NO_ADDITIONAL_CHAT_TEMPLATES"] = "1"
    
    args = parse_args()
    
    print(f"Using device: {device}")
    print("\nParsed arguments:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
    
    # Set seed
    setup_seed(args.seed)
    
    # Load model
    model, tokenizer, preprocess = load_biomedclip_model()
    
    # Load dataset
    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
    test_dataset = MedTestDataset(args.data_path, args.obj, args.img_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, **kwargs)
    print(f"\nTest dataset size: {len(test_dataset)}")
    
    # Encode text
    obj_real_name = REAL_NAME.get(args.obj, args.obj.lower())
    text_features = encode_text_with_prompt_ensemble(model, tokenizer, obj_real_name, device)
    
    # Test
    results = test(args, model, tokenizer, test_loader, text_features)
    
    # Save results
    os.makedirs(args.save_results, exist_ok=True)
    results_file = os.path.join(args.save_results, f'{args.obj}_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\n=== Complete ===")
    print(f"Results saved to: {results_file}")


if __name__ == '__main__':
    main()