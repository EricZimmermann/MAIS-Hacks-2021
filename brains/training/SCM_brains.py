import numpy as np
import lz4
import nibabel
def load_one_brain(path):
    img=nibabel.load(path)
    print(img)
    print(img.shape)




