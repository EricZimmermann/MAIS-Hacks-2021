import numpy as np
import torch
import torch.nn
import torch.nn.functional as F
from torch.optim import AdamW
from models.stn import STN
from kornia.augmentation import RandomElasticTransform, RandomAffine
from torch.utils.data import DataLoader 
import os
import argparse
from loader import generate_datasets

def threshold(tensor, T):
    tensor[tensor >= T] = 1.0
    tensor[tensor < T] = 0.0

def add_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d','--data', type=str, required=True, help='path to data')
    parser.add_argument('-o','--output', type=str, required=True, help='output directory')
    parser.add_argument('-g','--gpu', required=False, default=None, help='gpus')
    parser.add_argument('-l','--lr', type=float, required=False, default=3e-4, help='learning rate')
    parser.add_argument('-e','--epoch', type=int, required=False, default=100, help='num epochs') 
    parser.add_argument('-b','--batch', type=int, required=False, default=64, help='num epochs') 
    return parser

def main():
    parser = add_argparser()
    args = parser.parse_args()
    device = torch.device('cpu') if args.gpu is None else torch.device(f'cuda:{args.gpu}')
    classes, train_set, val_set = generate_datasets(args.data)
    train_loader = DataLoader(train_set, batch_size=args.batch, num_workers=4, pin_memory=True, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch, num_workers=4)
    stn = STN().to(device)
    optimizer = AdamW(stn.parameters(), lr=args.lr)
    elastic_aug = RandomElasticTransform(kernel_size=(33,33),sigma=(10,10), alpha=(5,5),p=0.5, mode='bilinear').to(device)
    affine_aug = RandomAffine(degrees=(-40,40), scale=(0.7, 1.15), shear=(-20,20), p=0.5, resample='bilinear').to(device)
    best_loss = np.inf
    
    # load dataset object
    for e in range(1, args.epoch+1):
        batch_loss = 0.0
        stn.train()
        for (batch, _) in train_loader:
            optimizer.zero_grad()
            with torch.no_grad():
                batch = batch.to(device)
                source = batch.detach().clone()
                augmented = elastic_aug(batch)
                augmented = affine_aug(augmented)
                
            registered = stn(augmented)
            loss = F.l1_loss(registered, source)
            loss.backward()
            optimizer.step()      
            batch_loss += loss.item()
        
        print(f'Train Loss {e}: {batch_loss / len(train_loader)}')
        
        batch_loss = 0.0
        stn.eval()
        for (batch, _) in val_loader:
            with torch.no_grad():
                batch = batch.to(device)
                source = batch.detach().clone()
                augmented = elastic_aug(batch)
                augmented = affine_aug(augmented)
                registered = stn(source)
                loss = F.mse_loss(registered, source)
                batch_loss += loss.item()
                
        print(f'Validation Loss {e}: {batch_loss / len(val_loader)}')
        
        if best_loss > batch_loss and e > 5:
            best_loss = batch_loss
            best_model = stn.state_dict()
            
    if not os.path.exists(args.output):
        os.path.mkdir(args.output)
    
    torch.save(best_model, os.path.join(args.output, 'stn.pth.tar'))
                             
if __name__ == "__main__":
    main()