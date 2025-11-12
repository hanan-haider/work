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

    # load dataset and loader
    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
    test_dataset = MedTestDataset(args.data_path, args.obj, args.img_size)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, **kwargs)



        # losses
    loss_focal = FocalLoss()
    loss_dice = BinaryDiceLoss()
    loss_bce = torch.nn.BCEWithLogitsLoss()