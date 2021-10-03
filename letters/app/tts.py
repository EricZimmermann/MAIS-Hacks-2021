from gtts import gTTS
import os
import sys
sys.path.insert(0, "../data")
from char_lut import CHARACTERS as chars
import pickle

# CHANGE ACCORDING TO SYSTEM
OUT_DIR = '/usr/local/data/kvasilev/mais/MAIS-Hacks-2021/letters/data/audio'
WAV = '.wav'
LANGUAGE="en"

# Create output directory for audio
if not os.path.exists(OUT_DIR):
    os.mkdir(OUT_DIR)

# go through LUT, generate audio, and save text/audio in new dict
text_audio_dict = {}
for lang in chars.keys():
    
    lang_dict = {}
    for letter in chars[lang].keys():
        
        wav_name = f"{lang}_{letter}{WAV}"
        wav_dir = f"{OUT_DIR}/{wav_name}"
        
        # generate audio
        prompt_text = f"Please draw letter {chars[lang][letter]}"
#         audio = gTTS(text=prompt_text, lang=LANGUAGE, slow=False)  # UNCOMMENT if want to regenerate audio files
#         audio.save(wav_dir)
        
        # add to dict
        output = {}
        output['text'] = chars[lang][letter]
        output['audio'] = wav_dir
        
        # add dict to lang dict
        lang_dict[letter] = output
        
    # add dictionary to lang
    text_audio_dict[lang] = lang_dict

# === Save text+audio dict ===
DICT_DIR = '/usr/local/data/kvasilev/mais/MAIS-Hacks-2021/letters/data/' # CHANGE PATH AS NEEDED
with open(f'{DICT_DIR}text-audio.pickle', 'wb') as handle:
    pickle.dump(text_audio_dict, handle)