import argparse
import os
from os.path import join
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch import FloatTensor
# import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

from net.unet import U_Net
from net.attention_unet import AttU_Net
from dataset import FireDataset

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

def get_dataset(img_dir, mask_dir):

    #Need to add more transforms like random flipping, cropping, image saturation change
    transform_img = transforms.Compose([
        transforms.Resize([256,256]), #Resize the input image to this size
        transforms.ToTensor()
        #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    transform_mask = transforms.Compose([
        transforms.Resize([256,256]),
        transforms.ToTensor()
        ])

    return FireDataset(img_dir, mask_dir, transform_img, transform_mask)

# +
def evaluate(args, model, test_dataloader, cutoff):
    model.eval()
    criterion = nn.BCELoss()
    eval_losses = []
    sensitivities = []
    specificities = []
    ppvs = []
    npvs = []

    for step, (images, masks) in enumerate(test_dataloader):

        if args.cuda:
            images = images.cuda(args.device_id)
            masks = masks.cuda(args.device_id)

        with torch.no_grad():
            output_masks = model(images)
            output_masks = torch.sigmoid(output_masks)
            loss = criterion(output_masks, masks)
            eval_losses.append(loss.item())
            
            output_masks = output_masks.detach().cpu().numpy()
            output_masks = np.where(output_masks > cutoff, 1, 0)
            masks = masks.detach().cpu().numpy()
            
            #Scoring

            tp = ((output_masks == 1) & (masks == 1)).sum()
            tn = ((output_masks == 0) & (masks == 0)).sum()
            fp = ((output_masks == 1) & (masks == 0)).sum()
            fn = ((output_masks == 0) & (masks == 1)).sum()
            
#             print("tp, tn, fp ,fn", tp, tn, fp ,fn)
            
            sensitivity = tp / (tp + fn)
            specificity = tn / (fp + tn)
            ppv = tp / (fp + tp)
            npv = tn / (fn + tn)
            
            sensitivities.append(sensitivity)
            specificities.append(specificity)
            ppvs.append(ppv)
            npvs.append(npv)

#     return np.mean(eval_losses)
    return np.mean(eval_losses), np.mean(sensitivities), np.mean(specificities), np.mean(ppvs), np.mean(npvs)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Eval Model')

    parser.add_argument("--test_img_dir", default="data/test_images",
        type=str, help="Directory containing test images")
    parser.add_argument("--test_mask_dir", default="data/test_masks",
        type=str, help="Directory containing test masks")

    parser.add_argument('--trained_model', type=str, default='trained_model_attn_unet', help='Trained model')
    parser.add_argument('--net', type=str, default='attn_unet', help='Model to train: unet|attn_unet')

    parser.add_argument('--device_id', type=int, default=0, help='GPU Device ID number if gpu is avaliable')
    parser.add_argument("--seed", type=int, default=10, help="random seed for initialization")

    args = parser.parse_args()

    # Set CUDA
    args.cuda = True if torch.cuda.is_available() else False
    print("Cuda",args.cuda)

    # Set seed
    set_seed(args)

    # Get datasets
    test_dataset = get_dataset(args.test_img_dir, args.test_mask_dir)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    #Model
    if args.net == 'unet':
        model = U_Net(img_ch=1, output_ch=1)
    if args.net == 'attn_unet':
        model = AttU_Net(img_ch=1, output_ch=1)

    model.load_state_dict(torch.load(args.trained_model))

    if args.cuda:
        model.cuda(args.device_id)

    # cutoffs = [i for i in range(0, 1, 0.05)]
    cutoffs = np.arange(0.3, 0.9, 0.05)
    print("Cutoff being tested:", cutoffs)

    for cutoff in cutoffs:
        eval_loss, eval_sens, eval_spec, eval_ppv, eval_npv = evaluate(args, model, test_dataloader, cutoff)
        print("Cutoff %.2f, Eval: loss %.3f, Sens %.3f, Spec %.3f, ppv %.3f, npv %.3f" \
                  %(cutoff, eval_loss, eval_sens, eval_spec, eval_ppv, eval_npv))



