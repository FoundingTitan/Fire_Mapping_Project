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

def get_dataset(img_dir, mask_dir, transform_mode, transform_types, mode):
    dataset_class = FireDataset(img_dir, mask_dir, transform_mode, transform_types, mode, return_name=True)
    return dataset_class

# +
def evaluate(args, model, test_dataloader, input_cutoff):
    model.eval()
    criterion = nn.BCELoss()
    eval_losses = []
    sensitivities = []
    specificities = []
    ppvs = []
    npvs = []
    f1s = []
    cutoffs = np.linspace(0,1,101)
    output_masks_list = []
    mask_list = []
    f2s = []

    #create output masks
    for step, (images, masks, names) in enumerate(test_dataloader):
        if args.cuda:
            images = images.cuda(args.device_id)
            masks = masks.cuda(args.device_id)

        with torch.no_grad():
            output_masks = model(images)
            output_masks = torch.sigmoid(output_masks)
            loss = criterion(output_masks, masks)
            eval_losses.append(loss.item())
            
            output_masks_list.append(output_masks)
            mask_list.append(masks)


    #cutoff loop
    for cutoff in cutoffs:
        #statistics for all images
        tps = []
        tns = []
        fps = []
        fns = []

        for output_masks, masks in zip(output_masks_list, mask_list):
            output_masks = output_masks.detach().cpu().numpy()
            output_masks = np.where(output_masks > cutoff, 1, 0)
            masks = masks.detach().cpu().numpy()

                #Save the eval output into images
                #if not os.path.exists(args.save_dir):
                #    os.makedirs(args.save_dir)
                ## img_name = step
                #img_name = names[0]
                #print("Saving prediction for",img_name)
                #img_savename = join(args.save_dir, str(img_name) + '.png')
                #output_mask_single = np.squeeze(output_masks)
                # print(output_mask_single.shape)

                #img = Image.fromarray(np.uint8(output_mask_single * 255) , 'L')
                #img.save(img_savename)
                ## Image.fromarray(output_mask_single, mode='L').save(img_savename)

                #Save the masks as well, as they are already resized, so its good for visualization
                #resized_mask = np.squeeze(masks)
                #resized_mask_img = Image.fromarray(np.uint8(resized_mask * 255) , 'L')
                #mask_savedir = "test_resized_masks"
                #if not os.path.exists(mask_savedir):
                #    os.makedirs(mask_savedir)
                #mask_img_savename = join(mask_savedir, str(img_name) + '.png')
                #resized_mask_img.save(mask_img_savename)
            
            #Scoring
            #individual statistics per image
            tp = ((output_masks == 1) & (masks == 1)).sum()
            tn = ((output_masks == 0) & (masks == 0)).sum()
            fp = ((output_masks == 1) & (masks == 0)).sum()
            fn = ((output_masks == 0) & (masks == 1)).sum()

            tps.append(tp)
            tns.append(tn)
            fps.append(fp)
            fns.append(fn)
        
        #sum all statistics of all images
        tpsum = np.sum(tps)
        tnsum = np.sum(tns)
        fpsum = np.sum(fps)
        fnsum = np.sum(fns)

        #performance metrics of all images
        f1 = 2*tpsum/(2*tpsum + fpsum + fnsum)
            
#             print("tp, tn, fp ,fn", tp, tn, fp ,fn)
        sensitivity = tpsum / (tpsum + fnsum)
        specificity = tnsum / (fpsum + tnsum)
        if (tpsum + fpsum) == 0:
          ppv = 1.0
        else:
          ppv = tpsum / (fpsum + tpsum)
        npv = tnsum / (fnsum + tnsum)

        #append performance metrics of all images for each cutoff    
        sensitivities.append(sensitivity)
        specificities.append(specificity)
        ppvs.append(ppv)
        npvs.append(npv)
        f1s.append(f1)

#     return np.mean(eval_losses)
    return cutoffs, eval_losses, sensitivities, specificities, f1s, ppvs, npvs #each cell is per cutoff


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Eval Model')

    parser.add_argument("--test_img_dir", default="data/test_images",
        type=str, help="Directory containing test images")
    parser.add_argument("--test_mask_dir", default="data/test_masks",
        type=str, help="Directory containing test masks")

    parser.add_argument('--trained_model', type=str, default='trained_model_attn_unet', help='Trained model')
    parser.add_argument('--net', type=str, default='attn_unet', help='Model to train: unet|attn_unet')
    parser.add_argument('--cutoff', type=float, default=0.3, help='Model cutoff')
    parser.add_argument('--save_dir', type=str, default='eval_predicted_masks', help='Directory to save prediction')
    parser.add_argument('--transform_mode', type=str, default='basic', help='basic | transform')
    parser.add_argument('--transform_types', type=str, nargs='*', default=['crop'],
                        help='crop, hflip, vflip')

    parser.add_argument('--device_id', type=int, default=0, help='GPU Device ID number if gpu is avaliable')
    parser.add_argument("--seed", type=int, default=10, help="random seed for initialization")

    args = parser.parse_args()

    # Set CUDA
    args.cuda = True if torch.cuda.is_available() else False
    print("Cuda",args.cuda)

    if args.cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # Set seed
    set_seed(args)

    # Get datasets
    test_dataset = get_dataset(args.test_img_dir, args.test_mask_dir, "basic",
                    transform_types=[], mode="test")
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    #Model
    if args.net == 'unet':
        model = U_Net(img_ch=1, output_ch=1)
    if args.net == 'attn_unet':
        model = AttU_Net(img_ch=1, output_ch=1)

    model.load_state_dict(torch.load(args.trained_model, map_location=device))

    if args.cuda:
        model.cuda(args.device_id)

    all_cutoffs, eval_loss, eval_sens, eval_spec, eval_f1, eval_ppv, eval_npv = evaluate(args, model, test_dataloader, args.cutoff)
    #print("Cutoff %.2f, Eval: Loss %.3f, Sens %.3f, Spec %.3f, F1 %.3f, PPV %.3f, NPV %.3f" \
              #%(args.cutoff, eval_loss, eval_sens, eval_spec, eval_f1, eval_ppv, eval_npv))
    print('cutoffs used: ', all_cutoffs)
    print('Eval loss: ', eval_loss)
    print('Sensitivity: ', eval_sens)
    print('Specificity:', eval_spec)
    print('F1: ', eval_f1)
    print('PPV: ', eval_ppv)
    print('NPV: ', eval_npv)


