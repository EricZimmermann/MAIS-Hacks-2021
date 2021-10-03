import torch
import sys
sys.path.insert(1, '..')
from training.models.stn import STN
from training.models.vae import VAE

def get_slider_images(decoder, start, end, num_slices):
    dl = (end - start) / (num_slices-1)
    stack = torch.tensor([start for i in range(num_slices)])
    stack += torch.arange(0, num_sclices) * dl
    decoded_stack = decoder(stack)
    return decoded_stack

def load_vae(path, ncls):
    vae = VAE(128, 256, ncls)
    vae_state_dict = torch.load(os.path.join(language_path, 'vae.pth.tar'))
    vae.load_state_dict(vae_state_dict)
    vae.eval()
    return vae

def load_stn(path):
    stn = STN()
    stn_state_dict = torch.load(os.path.join(language_path, 'stn.pth.tar'))
    stn.load_state_dict(stn_state_dict)
    stn.eval()
    return stn

class CAMWrapper(nn.Module):
    def __init__(self, vae):
        super(CAMWrapper, self).__init__()
        self.vae = vae
    
    def forward(self, img):
        latent = self.encoder(img)
        cls = self.classifier(latent)
        return cls

class StateMachine():
    def __init__(self, model_folder, data_folder, audio_folder, device):
        
        with open(os.path.join(audio_folder, 'text-audio.pickle'), 'rb') as handle:
            self.audio = pickle.load(handle)
        
        self.device = device
        self.english_data_path = os.path.join(data_folder, 'english')
        self.english_stn_path = os.path.join(model_folder, 'english/stn.pth.tar')
        self.english_vae_path = os.path.join(model_folder, 'english/vae.pth.tar')
        self.english_classes, self.english_dataset = generate_all_datasets(self.english_data_path)
        self.english_stn = load_stn(self.english_stn_path)
        self.english_vae = load_vae(self.english_vae_path, len(self.english_classes))
        self.english_hash = NerualHash(english_stn.encoder, 
                                       self.engish_dataset, 
                                       self.english_classes, 
                                       device)

        # greek
        self.greek_data_path = os.path.join(data_folder, 'greek')
        self.greek_stn_path = os.path.join(model_folder, 'greek/stn.pth.tar')
        self.greek_vae_path = os.path.join(model_folder, 'greek/vae.pth.tar')
        self.greek_classes, self.greek_dataset = generate_all_datasets(self.greek_data_path)
        self.greek_stn = self.load_stn(self.greek_stn_path)
        self.greek_vae = load_vae(greek_vae_path, len(greek_classes))
        self.greek_hash = NerualHash(self.greek_stn.encoder, 
                                     self.greek_dataset, 
                                     self.greek_classes, 
                                     device)

        # hebrew
        self.hebrew_data_path = os.path.join(data_folder, 'hebrew')
        self.hebrew_stn_path = os.path.join(model_folder, 'hebrew/stn.pth.tar')
        self.hebrew_vae_path = os.path.join(model_folder, 'hebrew/vae.pth.tar')
        self.hebrew_classes, hself.ebrew_dataset = generate_all_datasets(self.hebrew_data_path)
        self.hebrew_stn = load_stn(self.hebrew_stn_path)
        self.hebrew_vae = load_vae(self.hebrew_vae_path, len(self.hebrew_classes))
        self.hebrew_hash = NerualHash(self.hebrew_stn.encoder, 
                                      self.hebrew_dataset, 
                                      self.hebrew_classes,
                                      device)

        # hiragana
        self.hiragana_data_path = os.path.join(data_folder, 'hiragana')
        self.hiragana_stn_path = os.path.join(model_folder, 'hiragana/stn.pth.tar')
        self.hiragana_vae_path = os.path.join(model_folder, 'hiragana/vae.pth.tar')
        self.hiragana_classes, self.hiragana_dataset = generate_all_datasets(self.hiragana_data_path)
        self.hiragana_stn = load_stn(self.hiragana_stn_path)
        self.hiragana_vae = load_vae(self.hiragana_vae_path, len(self.hiragana_classes))
        self.hiragana_hash = NerualHash(self.hiragana_stn.encoder, 
                                        self.hiragana_dataset, 
                                        self.hiragana_classes, 
                                        device)

        # katakana
        self.katakana_data_path = os.path.join(data_folder, 'katakana')
        self.katakana_stn_path = os.path.join(model_folder, 'katakana/stn.pth.tar')
        self.katakana_vae_path = os.path.join(model_folder, 'katakana/vae.pth.tar')
        self.katakana_classes, self.katakana_dataset = generate_all_datasets(self.katakana_data_path)
        self.katakana_stn = load_stn(self.katakana_stn_path)
        self.katakana_vae = load_vae(self.katakana_vae_path, len(self.katakana_classes))
        self.katakana_hash = NerualHash(self.katakana_stn.encoder, 
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
        grad_cam = LayerGradCam(cam_model, cam_model.encoder.conv1)
        attr = layer_gc.attribute(sample, cls)
        cam_mask = LayerAttribution.interpolate(attr, (32, 32))
        return cam_mask
    
    def update(self, language, prompt, sketch)
    
        instructions = None
        visual_prompt = None
        audio_prompt = None
        text_prompt = None
        label_output = None
        interpret_image = None
        latent_video = None
                    
        self.language = language
        self.modality_index = prompt

        if self.language_index == 0:
            dataset = english_dataset
            classes = english_classes
            stn = english_stn
            vae = english_vae
            neural_hash = english_hash

        elif self.language_index == 1:
            dataset = greek_dataset
            classes = greek_classes
            stn = greek_stn
            vae = greek_vae
            neural_hash = greek_hash

        elif self.language_index == 2:
            dataset = hebrew_dataset
            classes = hebrew_classes
            stn = hebrew_stn
            vae = hebrew_vae
            neural_hash = hebrew_hash

        elif self.language_index == 3:
            dataset = hiragana_dataset
            classes = hiragana_classes
            stn = hiragana_stn
            vae = hiragana_vae
            neural_hash = hiragana_hash

        elif self.language_index == 4:
            dataset = katakana_dataset
            classes = katakana_classes
            stn = katakana_stn
            vae = katakana_vae
            neural_hash = katakana_hash

        if self.current_state == 0:
            self.sample, self.label = dataset[np.random.randint(len(dataset))]
            self.current_state = 1
            instructions = "Press Submit to get your prompt!"
            return instructions, visual_prompt, audio_prompt, text_prompt, label_output, interpret_image         

        if self.current_state == 1:
            instructions = "Press Submit to get your results!"
            if self.modality_index == 0:
                self.curerent_state = 2
                selected_lang = self.languages[self.language_index]
                audio_prompt = self.audio[selected_lang][self.classes[self.label]]['audio']
                return instructions, visual_prompt, audio_prompt, text_prompt, label_output, interpret_image
            if self.modality_index == 1:
                self.curerent_state = 2
                visual_prompt = np.transpose(self.sample, (1,2,0))
                return instructions, visual_prompt, audio_prompt, text_prompt, label_output, interpret_image
            if self.modality_index == 2:
                self.curerent_state = 2
                selected_lang = self.languages[self.language_index]
                text_prompt = self.audio[selected_lang][self.classes[self.label]]['text']
                return instructions, visual_prompt, audio_prompt, text_prompt, label_output, interpret_image

        if self.current_state == 2:
            instructions = "Press Submit to get your fixes!"
            self.curerent_state = 3
            input_tensor = torch.tensor(self.sample).unsqueeze(0)
            latent = self.vae.encoder(input_tensor)
            classification = self.vae.classifier(latent)
            classification = F.softmax(classification, 1).squeeze()
            label_output = {cls:score for cls, score in zip(self.classes, classification)}
            cam_mask = grad_cam(vae, input_tensor, torch.argmax(classification))
            return instructions, visual_prompt, audio_prompt, text_prompt, label_output, interpret_image

        if self.current_state == 3:
            self.current_state = 4
            with torch.no_grad():
                user_sample = torch.tensor(sketch).permute(1,2,0).unsqueeze(0)
                reg_user_sample = self.stn(user_sample)
                reg_latent = self.vae.encode(reg_user_sample)
                original_latent = self.vae.encode(user_sample)
                neighbour = self.hash(reg_latent, self.label)
                nb_latent = self.vae.encode(neighbour)
                slices = get_slider_images(self.vae.decode, original_latent, nb_latent, 300)
                slices = slices.permute(0,2,3,1).detach().numpy()

        if self.current_state == 4:
            instructions = 'Please press submit to reset"
            self.language_index = 0
            self.modality_index = 0
            self.current_state = 0
            return instructions, visual_prompt, audio_prompt, text_prompt, label_output, interpret_image                