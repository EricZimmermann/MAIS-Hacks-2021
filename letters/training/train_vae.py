import numpy as np
import torch
import torch.nn
import torch.nn.functional as F
from torch.optim import AdamW
from models.vae import VAEClassifier, variational_classifier_loss
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
    args = add_argparser()
    device = torch.device('cpu') if args.gpu is None else torch.device(f'cuda:{args.gpu}')
    classes, train_set, val_set = generate_datasets(args.data)
    train_loader = DataLoader(train_set, batch_size=args.batch, num_workers=4, pin_memory=True, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch, num_workers=4)
    vae_cls = VAEClassifier(1, 128, [32, 64, 128], len(classes)).to(device)
    optimizer = Adam(vae_cls.parameters(), lr=args.lr)
    elastic_aug = RandomElasticTransform(kernel_size=(33,33),sigma=(10,10), alpha=(5,5),p=0.5, mode='bilinear').to(device)
    affine_aug = RandomAffine(degrees=(-30,30), scale=(0.7, 1.15), shear=(15,15), p=0.5, resample='bilinear').to(device)
    
    best_loss = np.inf
    tr_kld_wt = arg.batch / len(train_loader)
    vl_kld_wt = arg.batch / len(val_loader)
    cls_wt = 1.0
    
    # load dataset object
    for e in range(1, args.epochs+1):
        batch_loss = 0.0
        batch_recon_loss = 0.0
        batch_kld_loss = 0.0
        batch_cls_loss = 0.0
        vae_cls.train()
        for (batch, cls) in train_loader:
            optimizer.zero_grad()
            with torch.no_grad():
                cls = cls.to(device)
                batch = batch.to(device)
                batch = elastic_aug(batch)
                augmented = affine_aug(batch)
                
            output, mu, log_var, cls = vae_clas(augmented)
            losses = variational_classifier_loss(augmented, output, cls, mu, log_var, tr_kld_wt, cls_wt)
            total_loss, recon_loss, kld_loss, cls_loss = losses
            total_loss.backwrd()
            optimizer.step()      
            batch_loss += total_loss.item()
            batch_recon_loss += recon_loss.item()
            batch_kld_loss += kld_loss.item()
            batch_cls_loss += cls_loss.item()
        
        print(f'Total Train Loss {e}: {batch_loss / len(train_loader)}')
        print(f'Recon Train Loss {e}: {batch_recon_loss / len(train_loader)}')
        print(f'KLD Train Loss {e}: {batch_kld_loss / len(train_loader)}')
        print(f'CLS Train Loss {e}: {batch_cls_loss / len(train_loader)}')
        
        batch_loss = 0.0
        batch_recon_loss = 0.0
        batch_kld_loss = 0.0
        batch_cls_loss = 0.0
        vae_cls.eval()
        for (batch, cls) in val_loader:
            with torch.no_grad():
                cls = cls.to(device)
                batch = batch.to(device)
                batch = elastic_aug(batch)
                augmented = affine_aug(batch)
                output, mu, log_var, cls = vae_clas(augmented)
                losses = variational_classifier_loss(augmented, output, cls, mu, log_var, vl_kld_wt, cls_wt)
                total_loss, recon_loss, kld_loss, cls_loss = losses
                batch_loss += total_loss.item()
                batch_recon_loss += recon_loss.item()
                batch_kld_loss += kld_loss.item()
                batch_cls_loss += cls_loss.item()
        
        print(f'Total Val Loss {e}: {batch_loss / len(val_loader)}')
        print(f'Recon Val Loss {e}: {batch_recon_loss / len(val_loader)}')
        print(f'KLD Val Loss {e}: {batch_kld_loss / len(val_loader)}')
        print(f'CLS Val Loss {e}: {batch_cls_loss / len(val_loader)}')
        
        if best_loss > batch_loss:
            best_loss = batch_loss
            best_model = stn.state_dict()
            
    if not os.path.exists(args.output) and e > 5:
        os.path.mkdir(args.output)
    torch.save(best_model, os.path.join(args.output, 'vae.pth.tar'))
                             
if __name__ == "__main__":
    main()