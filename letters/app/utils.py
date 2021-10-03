import torch
from torch import nn
import torch.nn.functional as F
from captum.attr import LayerGradCam
import numpy as np
import os
import pickle
import sys
sys.path.insert(1, '..')
from data.hash import NeuralHash
from training.loader import generate_all_datasets
from training.models.stn import STN
from training.models.vae import VAE
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML

def create_html_animation(img):
    print(img.shape)
    num_frames = len(img)
    fig, ax = plt.subplots()
    ax.axis('off')
    im = ax.imshow(img[0], animated=True)

    def update(i):
        im.set_data(img[i])
        return im,

    anim = animation.FuncAnimation(fig, update, interval=num_frames, blit=True, save_count=num_frames)
    #anim.save("test.gif", writer="imagemagick")
    html = anim.to_jshtml()
    return HTML(html)

def get_slider_images(decoder, start, end, num_slices):
    dl = (end - start) / (num_slices-1)
    stack = torch.cat([start+i*dl for i in range(num_slices)], dim=0)
    decoded_stack = decoder(stack)
    return decoded_stack

def load_vae(path, ncls):
    vae = VAE(128, 256, ncls)
    vae_state_dict = torch.load(os.path.join(path, 'vae.pth.tar'))
    vae.load_state_dict(vae_state_dict)
    vae.eval()
    return vae

def load_stn(path):
    stn = STN()
    stn_state_dict = torch.load(os.path.join(path, 'stn.pth.tar'))
    stn.load_state_dict(stn_state_dict)
    stn.eval()
    return stn

class CAMWrapper(nn.Module):
    def __init__(self, vae):
        super(CAMWrapper, self).__init__()
        self.vae = vae
    
    def forward(self, img):
        latent = self.vae.encoder(img)[0]
        cls = self.vae.classifier(latent)
        return cls

class StateMachine():
    def __init__(self, model_folder, data_folder, audio_folder, device):
        
        with open(os.path.join(audio_folder, 'text-audio.pickle'), 'rb') as handle:
            self.audio = pickle.load(handle)
        
        self.device = device
        self.english_data_path = os.path.join(data_folder, 'english')
        self.english_stn_path = os.path.join(model_folder, 'english')
        self.english_vae_path = os.path.join(model_folder, 'english')
        self.english_classes, self.english_dataset = generate_all_datasets(self.english_data_path)
        self.english_stn = load_stn(self.english_stn_path)
        self.english_vae = load_vae(self.english_vae_path, len(self.english_classes))
        self.english_hash = NeuralHash(self.english_vae.encoder, 
                                       self.english_dataset, 
                                       self.english_classes, 
                                       device)

        # greek
        self.greek_data_path = os.path.join(data_folder, 'greek')
        self.greek_stn_path = os.path.join(model_folder, 'greek')
        self.greek_vae_path = os.path.join(model_folder, 'greek')
        self.greek_classes, self.greek_dataset = generate_all_datasets(self.greek_data_path)
        self.greek_stn = load_stn(self.greek_stn_path)
        self.greek_vae = load_vae(self.greek_vae_path, len(self.greek_classes))
        self.greek_hash = NeuralHash(self.greek_vae.encoder, 
                                     self.greek_dataset, 
                                     self.greek_classes, 
                                     device)

        # hebrew
        self.hebrew_data_path = os.path.join(data_folder, 'hebrew')
        self.hebrew_stn_path = os.path.join(model_folder, 'hebrew')
        self.hebrew_vae_path = os.path.join(model_folder, 'hebrew')
        self.hebrew_classes, self.hebrew_dataset = generate_all_datasets(self.hebrew_data_path)
        self.hebrew_stn = load_stn(self.hebrew_stn_path)
        self.hebrew_vae = load_vae(self.hebrew_vae_path, len(self.hebrew_classes))
        self.hebrew_hash = NeuralHash(self.hebrew_vae.encoder, 
                                      self.hebrew_dataset, 
                                      self.hebrew_classes,
                                      device)

        # hiragana
        self.hiragana_data_path = os.path.join(data_folder, 'japanese_hiragana')
        self.hiragana_stn_path = os.path.join(model_folder, 'japanese_hiragana')
        self.hiragana_vae_path = os.path.join(model_folder, 'japanese_hiragana')
        self.hiragana_classes, self.hiragana_dataset = generate_all_datasets(self.hiragana_data_path)
        self.hiragana_stn = load_stn(self.hiragana_stn_path)
        self.hiragana_vae = load_vae(self.hiragana_vae_path, len(self.hiragana_classes))
        self.hiragana_hash = NeuralHash(self.hiragana_vae.encoder, 
                                        self.hiragana_dataset, 
                                        self.hiragana_classes, 
                                        device)

        # katakana
        self.katakana_data_path = os.path.join(data_folder, 'japanese_katakana')
        self.katakana_stn_path = os.path.join(model_folder, 'japanese_katakana')
        self.katakana_vae_path = os.path.join(model_folder, 'japanese_katakana')
        self.katakana_classes, self.katakana_dataset = generate_all_datasets(self.katakana_data_path)
        self.katakana_stn = load_stn(self.katakana_stn_path)
        self.katakana_vae = load_vae(self.katakana_vae_path, len(self.katakana_classes))
        self.katakana_hash = NeuralHash(self.katakana_vae.encoder, 
                                        self.katakana_dataset, 
                                        self.katakana_classes, 
                                        device)
        
        # state variables
        self.languages = ['ENGLISH', 'GREEK', 'HEBREW', 'JAPANESE_KATAKANA', 'JAPANESE_HIRAGANA']
        self.language_index = 0
        
        # prompt type
        self.modality = ['audio', 'visual', 'text']
        self.modality_index = 0
        
        # sample image
        self.transition_states = ['sample', 'prompt', 'diagnose', 'video', 'clear']
        self.current_state = 0
        
        # init
        self.has_started = False
        
    def grad_cam(self, vae, sample, cls):
        cam_model = CAMWrapper(vae)
        cam_model.eval()
        layer_gc = LayerGradCam(cam_model, cam_model.vae.encoder.conv1)
        attr = layer_gc.attribute(sample, cls, relu_attributions=True)
        cam_mask = layer_gc.interpolate(attr, (32, 32))
        return cam_mask
    
    def update(self, language, prompt, sketch):
    
        instructions = None
        visual_prompt = None
        audio_prompt = None
        text_prompt = None
        label_output = None
        interpret_image = None
        latent_video = None
                    
        self.language_index = language
        self.modality_index = prompt

        if self.language_index == 0:
            dataset = self.english_dataset
            classes = self.english_classes
            stn = self.english_stn
            vae = self.english_vae
            neural_hash = self.english_hash

        elif self.language_index == 1:
            dataset = self.greek_dataset
            classes = self.greek_classes
            stn = self.greek_stn
            vae = self.greek_vae
            neural_hash = self.greek_hash

        elif self.language_index == 2:
            dataset = self.hebrew_dataset
            classes = self.hebrew_classes
            stn = self.hebrew_stn
            vae = self.hebrew_vae
            neural_hash = self.hebrew_hash

        elif self.language_index == 3:
            dataset = self.katakana_dataset
            classes = self.katakana_classes
            stn = self.katakana_stn
            vae = self.katakana_vae
            neural_hash = self.katakana_hash

        elif self.language_index == 4:
            dataset = self.hiragana_dataset
            classes = self.hiragana_classes
            stn = self.hiragana_stn
            vae = self.hiragana_vae
            neural_hash = self.hiragana_hash

        print(self.current_state)

        if self.current_state == 0:
            self.current_state = 1
            instructions = "Press Submit to get your prompt!"
            return instructions, visual_prompt, audio_prompt, text_prompt, label_output, interpret_image, latent_video

        if self.current_state == 1:
            instructions = "Press Submit to get your results!"
            self.sample, self.label = dataset[np.random.randint(len(dataset))]
            if self.modality_index == 0:
                self.current_state = 2
                selected_lang = self.languages[self.language_index]
                audio_prompt = self.audio[selected_lang][classes[self.label]]['audio']
                return instructions, visual_prompt, audio_prompt, text_prompt, label_output, interpret_image, latent_video

            if self.modality_index == 1:
                self.current_state = 2
                visual_prompt = np.transpose(self.sample, (1,2,0))
                visual_prompt = 1 - np.repeat(visual_prompt, 3, axis=-1)
                return instructions, visual_prompt, audio_prompt, text_prompt, label_output, interpret_image, latent_video

            if self.modality_index == 2:
                self.current_state = 2
                selected_lang = self.languages[self.language_index]
                text_prompt = self.audio[selected_lang][classes[self.label]]['text']
                return instructions, visual_prompt, audio_prompt, text_prompt, label_output, interpret_image, latent_video


        if self.current_state == 2:
            instructions = "Press Submit to get your fixes!"
            self.current_state = 3
            user_sample = torch.tensor(sketch, dtype=torch.float).unsqueeze(0)
            user_sample = user_sample / 255.0
            input_tensor = user_sample.clone().unsqueeze(0)
            latent = vae.encoder(input_tensor)[0]
            classification = vae.classifier(latent)
            classification = F.softmax(classification, 1).squeeze()
            cam_mask = self.grad_cam(vae, input_tensor, torch.argmax(classification))
            classification = classification.detach().numpy().tolist()
            label_output = {cls:score for cls, score in zip(classes, classification)}

            # Generate cam image
            interpret_image = np.transpose(user_sample, (1,2,0))
            interpret_image = 1 - np.repeat(interpret_image, 3, axis=-1)
            
            plt.figure(figsize=(20,20))
            plt.xticks([])
            plt.yticks([])
            plt.imshow(interpret_image)
            plt.imshow(cam_mask.detach().squeeze().numpy(), cmap='jet', alpha=0.5)

            interpret_image = plt

            return instructions, visual_prompt, audio_prompt, text_prompt, label_output, interpret_image, latent_video


        if self.current_state == 3:
            self.current_state = 4
            with torch.no_grad():
                user_sample = torch.tensor(sketch, dtype=torch.float).unsqueeze(0).unsqueeze(0)
                user_sample = user_sample / 255.0
                reg_user_sample = stn(user_sample)
                reg_latent = vae.encoder(reg_user_sample)[0]
                original_latent = vae.encoder(user_sample)[0]
                neighbour = neural_hash(reg_latent, classes[self.label])
                nb_latent = vae.encoder(neighbour)[0]
                slices = get_slider_images(vae.decoder, original_latent, nb_latent, 150)
                slices = slices.permute(0,2,3,1).detach().numpy()

                # Create HTML animation
                latent_video = create_html_animation(slices)
                return instructions, visual_prompt, audio_prompt, text_prompt, label_output, interpret_image, latent_video               


        if self.current_state == 4:
            instructions = 'Please press submit to reset'
            self.language_index = 0
            self.modality_index = 0
            self.current_state = 0
            return instructions, visual_prompt, audio_prompt, text_prompt, label_output, interpret_image, latent_video               
