'''
Created on 2020. 1. 24.

@author: Inwoo Chung (gutomitai@gmail.com)
'''

import pandas as pd
from skimage.io import imread


def extract_valid_train_list():
    train_df = pd.read_csv('train_annotations_object_segmentation.csv')
    result_df = pd.DataFrame(columns=train_df.columns)
    
    for i in range(train_df.shape[0]):
        s = train_df.iloc[i]
        s_file_name = s.iloc[0].split('_')[0] + '.jpg'
        try:
            imread('train\\' + s_file_name)
        except:
            continue
            result_df = result_df.append(s)
    
    result_df.to_csv('train-valid-annotation-object-segmentation.csv')