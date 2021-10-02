#no data allowed to be uploaded :,(

from torch.utils.data import Dataset
import os
from glob import glob
import nibabel as nib
import numpy as np
import pandas as pd

MODALS = {
    'FLR':'_flair.nii.gz',
    'SEG':'_seg.nii.gz',
    'T1C':'_t1ce.nii.gz',
    'T1':'_t1.nii.gz',
    'T2':'_t2.nii.gz'
}

# for testing purposes
ROOT_DIR = '/usr/local/faststorage/BraTS19_Data/'

class VolumeLoader(Dataset):
    def __init__(self, root_dir, dataset_type, mode=MODALS['T1']):
        
        if (dataset_type == 'train'):
            self.root_dir = f"{root_dir}Training"    
            self.train_val = 'Training'
            self.clinical_dir = f"{self.root_dir}/survival_data.csv"
        elif (dataset_type == 'val'):
            self.root_dir = f"{root_dir}Validation"
            self.train_val = 'Validation'
            self.clinical_dir = f"{self.root_dir}/survival_evaluation.csv"     
        
        self.data_dir = f"{self.root_dir}/Data"
        self.mode = mode
        
        
    def __len__(self):
        '''
            get num of imaging files/dirs
        '''
        return len(glob(f"{self.data_dir}/*"))
    
    def __getitem__(self, idx):
        
        usr_dirs = glob(f"{self.data_dir}/*")
        output = {}
        
        # read clinical data
        clinical_df = pd.read_csv(self.clinical_dir)
        
        for usr in usr_dirs:
            usr_dict = {}
            
            # get user key, e.g. BraTS19_CBICA_ALA_1
            usr_key = os.path.basename(os.path.normpath(usr)) 
            
            # get mri dir, import as numpy arr, add to dict
            usr_mri_dir = f"{usr}/{usr_key}{self.mode}"
            usr_dict['MRI'] = nib.load(usr_mri_dir).get_fdata()
            
            # get clinical for user and add to dict
            usr_clinical = clinical_df.loc[clinical_df['BraTS19ID'] == usr_key]
            usr_dict['Age'] = usr_clinical['Age']
            usr_dict['ResectionStatus'] = usr_clinical['ResectionStatus']
            if (self.train_val == 'Training'):      # only present in training data
                usr_dict['Survival'] = usr_clinical['Survival']
                
            # add to dataset dict
            output[usr_key] = usr_dict