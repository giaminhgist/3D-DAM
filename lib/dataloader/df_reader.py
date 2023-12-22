import pandas as pd
import os
import numpy as np


# Read data from Excel file and save as image_[ath and label dict
def df_reader(df_path, process_path='/media/tedi/Elements/ADNI_Database/Images/PROCESS/subjects/'):
    image_path = []
    label_dict = {}
    feature_dict = {}
    df = pd.read_excel(df_path)
    for index, row in df.iterrows():
        participant_id = row['participant_id']
        session_id = row['session_id']
        diagnosis = row['diagnosis']

        idx = ''.join([participant_id, '_', session_id, '_',
                       'T1w_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_T1w.nii.gz'])
        img_path = f'{process_path}/{participant_id}/{session_id}/t1_linear/{idx}'

        if os.path.isfile(img_path):
            image_path.append(img_path)
            label_dict[idx] = diagnosis
            try:
                feature_dict[idx] = np.array([row['MMSE'], row['CDR'], row['APOE1'], row['APOE2'], row['AGE']])
            except:
                feature_dict[idx] = None

    return image_path, label_dict, feature_dict
