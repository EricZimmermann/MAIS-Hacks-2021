import numpy as np
import torch
import torch.nn
from torch.nn.optim import AdamW
from models.stn import STN
from kornia.augmentation import RandomElasticTransform, RandomAffine
import os

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
    stn = STN().to(device)
    optimizer = Adam(stn.parameters(), lr=args.lr)
    elastic_aug = RandomElasticTransform().to(device)
    affine_aug = RandomAffine(degrees=(-25,25), 
                              translate=None, 
                              scale=(0.8, 1.05),
                              shear=(8,8), 
                              p=1).to(device)
    best_loss = np.inf
    
    # load dataset object
    for e in range(1, args.epochs+1):
        batch_loss = 0.0
        stn.train()
        for batch in train_loader:
            optimizer.zero_grad()
            with torch.no_grad():
                batch = batch.to(device)
                batch = elastic_aug(batch)
                source = batch.detach().clone()
                target = affine_aug(source)
                
            registered = stn(source)
            loss = F.mse_loss(registered, target)
            loss.backwrd()
            optimizer.step()      
            batch_loss += loss.item()
        
        print(f'Train Loss {e}: {batch_loss / len(train_loader)}')
        
        batch_loss = 0.0
        stn.eval()
        for batch in val_loader:
            with torch.no_grad():
                batch = batch.to(device)
                batch = elastic_aug(batch)
                source = batch.detach().clone()
                target = affine_aug(source)
                registered = stn(source)
                loss = F.mse_loss(registered, target)
                batch_loss += loss.item()
                
        print(f'Validation Loss {e}: {batch_loss / len(train_loader)}')
        
        if best_loss > batch_loss:
            best_loss = batch_loss
            best_model = stn.state_dict()
            
    if not os.path.exists(args.output):
        os.path.mkdir(args.output)
        torch.save(best_model, os.path.join(args.output, 'stn.pth.tar'))
                             
if __name__ == "__main__":
    main()