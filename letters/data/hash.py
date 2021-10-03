import torch
import torch.nn.functional as F
import os

class NeuralHash:
    '''
        Query the closest image in the dataset using VAE latent space using class keys and max cosine agreement
    '''
    def __init__(self, encoder, dataset, classes, device):
        self.encoder = encoder
        self.classes = classes
        self.device= device
        self.hash = {}
        
        # generate quick hash DB for searching
        for cls in classes:
            self.hash[cls] = {'image': [], 'value' : []}
            
        for idx in range(len(dataset)):
            with torch.no_grad():
                image, cls = dataset[idx]
                image = torch.tensor(image).float().unsqueeze(0)
                embedding = F.normalize(self.encoder(image)[0].view(-1).detach(), dim=0)
                self.hash[classes[cls]]['image'].append(image)
                self.hash[classes[cls]]['value'].append(embedding)
       
        for cls in classes:
            self.hash[cls]['value'] = torch.stack(self.hash[cls]['value'], dim=0)
    
    def __call__(self, latent, cls):
        with torch.no_grad():
            embedding = F.normalize(latent.view(-1), dim=0)
            cosine_distance = torch.matmul(embedding, self.hash[cls]['value'].T)
            neighbour = self.hash[cls]['image'][torch.argmax(cosine_distance)]
            return neighbour
        
    def closest_nbs(self, latent):
        matches = {}
        for cls in self.hash.keys():
            matches[cls] = self.__call__(latent, cls)
        return matches
