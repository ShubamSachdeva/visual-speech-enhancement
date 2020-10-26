import landmarks as ln
# import face_api as face
import options as opt
import face_utils as fu
import warp
import preprocessing as prep

import skimage
import skimage.transform
import cv2
from PIL import Image
import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# from sklearn.linear_model import Ridge
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.pipeline import make_pipeline

import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm


import soundfile as sf
import pyrubberband as pyrb
from pydub import AudioSegment
import librosa


data = pd.read_csv(opt.data_path)


def audio_stretch(audio_path, start_point, end_point, stretch, out_path):
    full_audio = AudioSegment.from_wav(audio_path)
    to_stetch = full_audio[start_point*1000:end_point*1000]

    y, sr = sf.read(audio_path)
    to_stretch_array = np.asarray(to_stetch.get_array_of_samples())

    stretched_audio = pyrb.time_stretch(to_stretch_array, sr, 1/stretch)

    librosa.output.write_wav('temp.wav', stretched_audio, sr)

    part1, _ = librosa.core.load(audio_path, duration = start_point, sr = sr)
    part2, _ = librosa.core.load('temp.wav', sr = sr)
    part3, _ = librosa.core.load(audio_path, offset = end_point, sr = sr)

    final_audio = np.concatenate((part1, part2, part3))
    
    librosa.output.write_wav(out_path, final_audio, sr)
    
    
def frame_borders(landmarks, h = 1080, w = 1440):
    border = np.array([(0,0), (w/2,0), (w-1,0), (w-1,h/2), ( w-1, h-1 ), ( w/2, h-1 ), (0, h-1), (0,h/2) ])
    if landmarks.ndim == 2:
        return np.concatenate((landmarks, border), axis = 0)
    elif landmarks.ndim == 3:
        border = np.asarray([border])
        return np.concatenate((landmarks, np.repeat(border, landmarks.shape[0], axis=0)), axis=1)
    
    
def interpolate_landmarks(src, dst, dst_ratio):
    return ((1-dst_ratio)*src) + (dst_ratio*dst)


def frame_to_video(conv_video_path, target = 'self', impact = 1):
    conv_row = data[data['video'] == conv_video_path].iloc[0]
    token = conv_row['token']
    
    real_start_time = conv_row['word_start_time']
    end_time = conv_row['word_end_time']
    start_time = real_start_time - (end_time-real_start_time)
    
    real_start_frame = conv_row['word_start_frame']
    end_frame = conv_row['word_end_frame']
    start_frame = real_start_frame - (end_frame-real_start_frame)
        
    audio = conv_row['textGrid'].split('/')
    if audio[-1][-11:] == 'NR.TextGrid':
        audio[-1] = conv_video_path.split('/')[-1][:-4]+'_NR.wav'
    else:
        audio[-1] = conv_video_path.split('/')[-1][:-4]+'.wav'
    audio_path = '/'.join(audio) 
    
    conv_row = data[data['video'] == conv_video_path].iloc[0]
    
    src_landmarks, dst_landmarks = prep.get_destination_landmarks(conv_row['video'], target = target, impact = impact)

    conv_data = data[(data['subject'] != conv_row['subject']) 
                     & (data['token'] == conv_row['token']) 
                     & (data['speech_type'] == 'conv')
                     & (data['corresponding'] != -1)]

    conv_data = conv_data[~conv_data['subject'].isin(opt.test_subjects)]

    conv_length_list = []
    clear_length_list = []

    for idx, row in conv_data.iterrows():    
        conv_length_list.append(row['word_end_time'] - row['word_start_time'])

        corresponding_row = data.iloc[row['corresponding']]
        clear_length_list.append(corresponding_row['word_end_time'] - corresponding_row['word_start_time'])          

    conv_vowel_length = np.mean(np.asarray(conv_length_list))
    clear_vowel_length = np.mean(np.asarray(clear_length_list))
    
#     print("clear:", clear_vowel_length)
    length_diff = clear_vowel_length - conv_vowel_length
    length_diff = length_diff*impact
    clear_vowel_length = conv_vowel_length + length_diff
    
    length_ratio = clear_vowel_length/conv_vowel_length
    conv_video = cv2.VideoCapture(conv_row['video'])
    
    fps = 29 #conv_video.get(cv2.CAP_PROP_FPS)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     1110-cud-self-2-interpolation
#     name = "results/"+str(conv_row['subject'])+"_"+conv_row['token']+"_"+target+"_"+str(impact)+"_interpolation.mp4"
#     name = "frame_to_video/"+str(conv_row['subject'])+"_"+conv_row['token']+"_"+target+"_"+str(impact)+"_frame.mp4"
    name = "example.mp4"
    print("Generating: ", name)
    out = cv2.VideoWriter('temp.mp4', fourcc, fps, (1440,1080))
    
    return_landmarks = np.loadtxt(conv_row["landmarks"])[:start_frame]
    return_landmarks = return_landmarks.reshape(return_landmarks.shape[0], 68, 2)
    
    frame_count = 0
    res = True
    while res == True:
        res, image = conv_video.read()
        frame_count += 1
        if (frame_count < start_frame):
            out.write(np.uint8(image))
        if frame_count == conv_row['mid_vowel_frame']:
            src_face = image
        if frame_count == end_frame:
            break
    
    conv_mid_landmarks = np.loadtxt(conv_row["landmarks"])[conv_row['mid_vowel_frame']]
    conv_mid_landmarks = conv_mid_landmarks.reshape(68, 2)
#     print(src_landmarks.shape, dst_landmarks.shape)
    for i in range(src_landmarks.shape[0]):     
        warped_landmarks, image = warp.face_warp(src_face, conv_mid_landmarks, src_landmarks[i], dst_landmarks[i], 
                               impact = impact, method = 'copy_landmarks', warp_method = 'triangulation')
        out.write(np.uint8(image))
        warped_landmarks = warped_landmarks.reshape(1, 68, 2)
        return_landmarks = np.concatenate((return_landmarks, warped_landmarks), axis = 0)
#         plt.imshow(image)
#         plt.show()
    
    while res == True:
        res, image = conv_video.read()
        if res:
            out.write(np.uint8(image))
        else:
            break
    
    out.release()
    audio_stretch(audio_path, start_time, end_time, length_ratio+0.000001, 'temp.wav')
    
    return_landmarks_last_part = np.loadtxt(conv_row["landmarks"])[end_frame+1:]
    return_landmarks_last_part = return_landmarks_last_part.reshape(return_landmarks_last_part.shape[0], 68, 2)
    return_landmarks = np.concatenate((return_landmarks, return_landmarks_last_part), axis = 0)
    return_landmarks = return_landmarks.reshape(return_landmarks.shape[0], 68 * 2)
#     print("return landmarks shape: ",return_landmarks.shape)
    np.savetxt(name[:-3]+"txt", return_landmarks)
    
    if os.path.isfile(name):
        os.system("rm "+name)
    os_command = "ffmpeg -i temp.mp4 -i temp.wav -c:v copy -c:a aac "+ name
    os.system(os_command)
    os.system("rm temp.wav")
    os.system("rm temp.mp4")
    
