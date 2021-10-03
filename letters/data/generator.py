import pygame
import os
from glob import glob

ENGLISH = 'english'
GREEK = 'greek'
HEBREW = 'hebrew'
JAPANESE_KATAKANA = 'japanese_katakana'
JAPANESE_HIRAGANA = 'japanese_hiragana'
LANGUAGES = [ ENGLISH, GREEK, HEBREW, JAPANESE_KATAKANA, JAPANESE_HIRAGANA ]

CHARACTERS = {
    ENGLISH: 'AaBbCcDdEeFfGgHhIiJjKkLlMmNnOoPpQqRrSsTtUuVvWwXxYyZz',
    GREEK: 'ΑαΒβΓγΔδΕεΖζΗηΘθΙιΚκΛλΜμΝνΞξΟοΠπΡρΣσςΤτΥυΦφΧχΨψΩω',
    HEBREW: 'אבגדהוזחטיכךלמםנןסעפףצץקרשת',
    JAPANESE_KATAKANA: 'アイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワヲン',
    JAPANESE_HIRAGANA: 'あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをん',
}

OUTPUT_FOLDER = 'characters/'

def main():

    language_fonts = {}
    for language in LANGUAGES:
        fonts = glob(f'fonts/{language}/*/*.ttf') + glob(f'fonts/{language}/*/*/*.ttf')
        fonts = list(filter(lambda f: 'VariableFont' not in f, fonts))
        language_fonts[language] = fonts
        print(f'{language} - Number of fonts: {len(fonts)}')

    pygame.init()

    screen = pygame.display.set_mode((32,32))

    for language in LANGUAGES:
        for font_path in language_fonts[language]:
            font_name = font_path.split('/')[-1].split('.')[0]
            font = pygame.font.Font(font_path, 32)

            for char in CHARACTERS[language]:
                text = font.render(char, False, (0,0,0))

                screen.fill((255,255,255))
                screen.blit(text, ((screen.get_width() - text.get_width())/2,
                                   (screen.get_height() - text.get_height())/2))
                pygame.display.update()
                output_dir = f'{OUTPUT_FOLDER}/{language}/{char}'
                if not os.path.isdir(output_dir):
                    os.makedirs(output_dir)
                pygame.image.save(screen, f'{output_dir}/{font_name}.png')

    pygame.quit()

if __name__ == '__main__':
    main()
