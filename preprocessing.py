import landmarks as ln
# import face_api as face
import options as opt
import face_utils as fu

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

import skimage
import skimage.transform
import cv2
from PIL import Image

from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

import warnings


warnings.filterwarnings('ignore')
data = pd.read_csv(opt.data_path)



"""
# Returns the corner points and center of each of the 4 edges of the image to concatenate while warping
"""
def frame_borders(landmarks, h = 1080, w = 1440):
    border = np.array([(0,0), (w/2,0), (w-1,0), (w-1,h/2), ( w-1, h-1 ), ( w/2, h-1 ), (0, h-1), (0,h/2) ])
    if landmarks.ndim == 2:
        return np.concatenate((landmarks, border), axis = 0)
    elif landmarks.ndim == 3:
        border = np.asarray([border])
        return np.concatenate((landmarks, np.repeat(border, landmarks.shape[0], axis=0)), axis=1)


"""
Given the landmarks, smoothes the landmarks temporally using nth degree polynomial regression. 
n here is the degree defined in the options.py
landmarks are video landmarks with shape m*68*2. m is the number of frames
If landmarks_to_return == None, the same number of landmarks are retuned which is m
"""
def smooth_landmarks(landmarks, landmarks_to_return = None, degree = None):
    if degree == None:
        degree = opt.degree
    landmarks = np.asarray(landmarks)
    x = np.linspace(0, 100, landmarks.shape[0])
    X = x[:, np.newaxis]
    
    if(landmarks_to_return == None):
        landmarks_to_return = landmarks.shape[0]
    
    x_test = np.linspace(0, 100, landmarks_to_return)
    X_test = x_test[:, np.newaxis]
    
    final_landmarks = np.zeros((landmarks_to_return, landmarks.shape[1], landmarks.shape[2]), dtype = float)
#     final_landmarks = np.asarray(landmarks, dtype = float)

    nose_tips = landmarks[:,33]
    nose_tips_smooth = np.zeros((landmarks_to_return, 2))
    y0 = nose_tips[:, 0]
    y1 = nose_tips[:, 1]

    model0 = make_pipeline(PolynomialFeatures(degree), Ridge())
    model0.fit(X, y0)
    nose_tips[:, 0] = model0.predict(X)
    nose_tips_smooth[:, 0] = model0.predict(X_test)

    model1 = make_pipeline(PolynomialFeatures(degree), Ridge())
    model1.fit(X, y1)
    nose_tips[:, 1] = model1.predict(X)
    nose_tips_smooth[:, 1] = model1.predict(X_test)
    
#     landmarks = landmarks-nose_tips
    landmarks_copy = np.asarray([landmarks[i]-nose_tips[i] for i in range(landmarks.shape[0])])
    landmarks = landmarks_copy
    
    for i in range(landmarks.shape[1]):
        y0 = landmarks[:, i, 0]
        y1 = landmarks[:, i, 1]
        
        model0 = make_pipeline(PolynomialFeatures(degree), Ridge())
        model0.fit(X, y0)
        final_landmarks[:, i, 0] = model0.predict(X_test)
        
        model1 = make_pipeline(PolynomialFeatures(degree), Ridge())
        model1.fit(X, y1)
        final_landmarks[:, i, 1] = model1.predict(X_test)
        
    return_landmarks = np.asarray([final_landmarks[i] + nose_tips_smooth[i] for i in range(final_landmarks.shape[0])])
        
    return return_landmarks


"""
Given the conversational video, finds the source landmarks and destination landmarks
"""
def get_destination_landmarks(conv_video_path, impact = 1,
                              degree = None, visualize = False, landmarks_to_return = None):
    if degree == None:
        degree = opt.degree
    this_row = data[data['video'] == conv_video_path].iloc[0]
    subject = this_row['subject']
    token = this_row['token']
    
    src_ln = np.loadtxt(this_row['landmarks'])[this_row['mid_vowel_frame']]
    src_ln = src_ln.reshape(68, 2)
    
    if target == "self":
        conv_data = data[data['video'] == conv_video_path]
    elif target == "avg":
        conv_data = data[(data['subject'] != subject) 
                               & (data['token'] == token) 
                               & (data['speech_type'] == 'conv')
                               & (data['corresponding'] != -1)]
    elif target =="nn":
        print("getting the nearest neighbors")
        nearest_neighbors = get_nearest_neighbors(conv_video_path, opt.nn)
        conv_data = data[data['video'].isin(nearest_neighbors)]
        
    print("processing clear landmarks")
    corresponding = list(conv_data['corresponding'])
    clear_data = data.iloc[corresponding]
    
    clear_word_length = np.asarray(clear_data['word_end_time'] - clear_data['word_start_time'])
    conv_word_length = np.asarray(conv_data['word_end_time'] - conv_data['word_start_time'])
    length_diff = clear_word_length - conv_word_length
    length_diff = length_diff*impact
    clear_word_length = conv_word_length + length_diff  
    
    if landmarks_to_return == None:
        landmarks_to_return = np.mean(landmarks_to_return, dtype=np.int)
        
    src_landmarks = frame_borders(src_ln)
    
    warp_trans = skimage.transform.PiecewiseAffineTransform()

    clear_landmarks_array = []
    conv_landmarks_array = []
    
    for idx, row in tqdm(conv_data.iterrows()):
#         if str(row['subject'])[:2] == str(subject)[:2]:
        if 1 == 1:
            conv_real_start_frame = row['word_start_frame']
            conv_end_frame = row['word_end_frame']
            conv_start_frame = conv_real_start_frame - (conv_end_frame - conv_real_start_frame)
            
            conv_landmarks = np.loadtxt(row['landmarks'])[conv_start_frame:conv_end_frame+1]
            conv_landmarks = conv_landmarks.reshape(conv_landmarks.shape[0], 68, 2)
#             conv_landmarks = np.asarray([ln.normalize_landmarks(landmark) for landmark in conv_landmarks])
            conv_landmarks = smooth_landmarks(conv_landmarks, landmarks_to_return)
            conv_landmarks = frame_borders(conv_landmarks)
    
#             conv_mid_frame_landmarks = np.loadtxt(row['landmarks'])[row['mid_vowel_frame']]
#             conv_mid_frame_landmarks = conv_mid_frame_landmarks.reshape(68, 2)
            conv_mid_frame_landmarks = conv_landmarks[row['mid_vowel_frame'] - conv_start_frame]
    
            clear_row = data.iloc[row['corresponding']]
            clear_real_start_frame = clear_row['word_start_frame']
            clear_end_frame = clear_row['word_end_frame']
            clear_start_frame = clear_real_start_frame - (clear_end_frame - clear_real_start_frame)
            
            clear_landmarks = np.loadtxt(clear_row['landmarks'])[clear_start_frame:clear_end_frame+1]
            clear_landmarks = clear_landmarks.reshape(clear_landmarks.shape[0], 68, 2)
#             clear_landmarks = np.asarray([ln.normalize_landmarks(landmark) for landmark in clear_landmarks])
            clear_landmarks = smooth_landmarks(clear_landmarks, landmarks_to_return)
            clear_landmarks = frame_borders(clear_landmarks)
        
            warp_trans.estimate(conv_mid_frame_landmarks, src_landmarks)
            
            for i in range(landmarks_to_return):
                tform =  skimage.transform.estimate_transform('similarity', clear_landmarks[i, opt.stablePntsIDs, :], conv_landmarks[i, opt.stablePntsIDs, :])
                clear_landmarks[i] = warp_trans(tform(clear_landmarks[i]))
                conv_landmarks[i] = warp_trans(tform(conv_landmarks[i]))
                
#             conv_landmarks = np.asarray([warp_trans(conv_landmarks[i]) for i in range(landmarks_to_return)])
#             clear_landmarks = np.asarray([warp_trans(clear_landmarks[i]) for i in range(landmarks_to_return)])
            
            conv_landmarks_array.append(conv_landmarks)
            clear_landmarks_array.append(clear_landmarks)
#         break
    clear_landmarks_array = np.asarray(clear_landmarks_array)
    return_landmarks = np.mean(clear_landmarks_array, axis=0)
    return_landamrks = smooth_landmarks(return_landmarks)
    
    return src_landmarks[:, :68, :], return_landmarks[:, :68, :]
