import argparse
import os
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

from net.unet import U_Net
from net.attention_unet import AttU_Net
from dataset import FireDataset

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

def get_dataset(img_dir, mask_dir, transform_mode='basic'):
    return FireDataset(img_dir, mask_dir, transform_mode)

# +
def evaluate(args, model, test_dataloader):
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
            output_masks = np.where(output_masks > args.cutoff, 1, 0)
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


# -

def train(args, train_dataloader, test_dataloader):

    epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = args.lr
    log_interval = args.log_interval

    # Model
    if args.net == 'unet':
        model = U_Net(img_ch=1, output_ch=1)
    if args.net == 'attn_unet':
        model = AttU_Net(img_ch=1, output_ch=1)

    if args.cuda:
        model.cuda(args.device_id)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Loss criterion - BCE: binary cross entropy loss, as 2 classes in mask
    criterion = nn.BCELoss()


    # Begin training
    print("Model :", args.net)
    print("Cutoff :", args.cutoff)
    print("Train Dataloader size :",len(train_dataloader))
    print("Transform mode :",args.transform_mode)
    print("Batch size :", batch_size)
    print("Epochs :",epochs)
    print("Begin Training")
    model.train()
    model.zero_grad()

    best_eval_sens = 9999

    for epoch in range(epochs):
        epoch_losses = []
        for step, (images, masks) in enumerate(train_dataloader):
            model.train()

            if args.cuda:
                images = images.cuda(args.device_id)
                masks = masks.cuda(args.device_id)

            output_masks = model(images)
            output_masks = torch.sigmoid(output_masks)
            loss = criterion(output_masks, masks)
            # print(loss)

            loss.backward()
            optimizer.step()
            model.zero_grad()

            epoch_losses.append(loss.item())

        # Printing
        if epoch % log_interval == 0:
            current_epoch_loss = np.mean(epoch_losses)
            eval_loss, eval_sens, eval_spec, eval_ppv, eval_npv = evaluate(args, model, test_dataloader)
            print("Epoch %d, Loss %.3f, Eval: loss %.3f, Sens %.3f, Spec %.3f, ppv %.3f, npv %.3f" \
                  %(epoch, current_epoch_loss, eval_loss, eval_sens, eval_spec, eval_ppv, eval_npv))

            #Save the model with minimun evaluation sensitivity :
            if epoch > 10 and eval_sens != 1.0 and eval_sens > best_eval_sens:
                # Save the model
                print("Saving epoch %d model"%(epoch))
                torch.save(model.state_dict(), 'trained_model_'+str(epoch)+'_'+str(args.net))
                best_eval_sens = eval_sens


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Training Model')

    parser.add_argument("--train_img_dir", default="data/train_images",
        type=str, help="Directory containing train images")
    parser.add_argument("--train_mask_dir", default="data/train_masks",
        type=str, help="Directory containing train masks")
    parser.add_argument("--test_img_dir", default="data/test_images",
        type=str, help="Directory containing test images")
    parser.add_argument("--test_mask_dir", default="data/test_masks",
        type=str, help="Directory containing test masks")

    parser.add_argument('--net', type=str, default='unet', help='Model to train: unet|attn_unet')
    parser.add_argument('--epochs', type=int, default=100, help='Number training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--cutoff', type=float, default=0.30, help='Cutoff')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for optimizers')
    parser.add_argument('--log_interval', type=int, default=1, help='Print loss values every log_interval epochs.')
    parser.add_argument('--transform_mode', type=str, default='basic', help='basic | crop_hflip_vflip')

    parser.add_argument('--device_id', type=int, default=0, help='GPU Device ID number if gpu is avaliable')
    parser.add_argument("--seed", type=int, default=10, help="random seed for initialization")

    args = parser.parse_args()

    # Set CUDA
    args.cuda = True if torch.cuda.is_available() else False
    print("Cuda",args.cuda)

    # Set seed
    set_seed(args)

    # Get datasets
    train_dataset = get_dataset(args.train_img_dir, args.train_mask_dir, args.transform_mode)
    test_dataset = get_dataset(args.test_img_dir, args.test_mask_dir, args.transform_mode)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    train(args, train_dataloader, test_dataloader)