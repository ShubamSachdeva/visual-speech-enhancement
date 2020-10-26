import os
import cv2
import sys
import pandas as pd
import numpy as np
import dlib
from tqdm import tqdm
import landmarks as ln
import face_utils as fu
import options as opt


"""
load the data. The data path is in options.py
"""
def load_data(path):
    if path == None:
        path == opt.data_path
    if not os.path.isfile(path):
        data = get_info("all")
        data.to_csv(path)
    return pd.load_csv(path)




"""
Goes into every folder and finds the corresponding audio and textGrid files.
if subjects == all, it fins info for all subjects.
Other options are males, females, or individual speakers.
"""
def get_videos_info(subjects="all"):
    males = [1101, 1102, 1103, 1104, 1106, 1107, 1109, 1110]
    females = [1201, 1202, 1203, 1204, 1205, 1206, 1207, 1208, 1209, 1210]
    fall = [1101, 1104, 1107, 1109, 1110, 1201, 1202, 1203, 1206, 1210]
    if subjects == "males":
        subjects = males
    elif subjects == "females":
        subjects = females
    elif subjects == "all":
        subjects = males + females
    print("Trying to obtain info for subject(s) : ",subjects)
    video_paths = []
    audio_paths = []
    tokens = []
    textGrid_paths = []
    subject_numbers = []
    segmentation_numbers = []
    elicitations = []
    words_of_interest = ['cod','cooed','could','cud','keyed','kid']
    videos_folder = '../videos'
    audios_folder = '../audios'
    print("Getting path informations")
    for speaker in tqdm(os.listdir(videos_folder)):
        if speaker in map(str, subjects):
            speaker_folder = videos_folder + "/" + speaker
            for e in os.listdir(speaker_folder):
                if e in ['e1', 'e2', 'e3']:
                    time_folder = audios_folder+"/"+speaker+"_NR"
                    time_folder = time_folder+"/"+speaker+"_"+e+"_NR"
                    time_folder = time_folder+"/"+speaker+"_"+e+"_NR Vowel"
                    
                    if not os.path.isdir(time_folder):
                        time_folder = time_folder+"s"
                        
                    for file in os.listdir(speaker_folder+'/'+e): #1101_92_kid_00.11.36_HQ.avi
                        if file[-6:] in ['HQ.wmv', 'HQ.avi', 'HQ.mp4']:
                            token = file.split('_')[2]
                            if(token in words_of_interest):
                                time_file = file.split('.')[0]+"_"+file.split('.')[1]+"_"+file.split('.')[2]+".TextGrid"
                                if speaker == "1203":
                                    if e == "e1":
                                        time_file = time_file.split(".")[0]+"_HQ.TextGrid"
                                    else:
                                        time_file = time_file.split(".")[0]+"_NR.TextGrid"
                                if os.path.isfile(time_folder+"/"+time_file) :
                                    video_path = speaker_folder+"/"+e+"/"+file
                                    video_paths.append(video_path)
                                    tokens.append(token)
                                    textGrid_path = time_folder+"/"+time_file
                                    textGrid_paths.append(textGrid_path)
                                    subject_numbers.append(int(speaker))
                                    elicitations.append(e)
                                    segmentation_numbers.append(int(file.split('_')[1]))
                                    
                                    audio = textGrid_paths[-1].split('/')
                                    if audio[-1][-11:] == 'NR.TextGrid':
                                        audio[-1] = video_paths[-1].split('/')[-1][:-4]+'_NR.wav'
                                    else:
                                        audio[-1] = video_paths[-1].split('/')[-1][:-4]+'.wav'
                                    audio_path = '/'.join(audio)
                                    audio_paths.append(audio_path)
                                    
    df = pd.DataFrame(list(zip(video_paths, tokens, textGrid_paths, audio_paths, subject_numbers,segmentation_numbers,elicitations)), 
                   columns =["video", "token", "textGrid", "audio", "subject","segmentation","elicitation"]) 

    df = df.sort_values(by =['subject', 'elicitation', 'segmentation'])
    df = df.reset_index(drop=True)

    print("Found :", len(df), " data points")
    return df



"""
Write to the data the position (frame and time) of where the word starts, vowel startes, etc.
"""
def get_landmarks_info(data):
    df = data.copy(deep = True)
    keep_flag = []
    swc = []
    svc = []
    evc = []
    ewc = []
    mvc = []
    swt = []
    svt = []
    evt = []
    ewt = []
    ln_txt = []
    print("writing landmarks")
    for i in tqdm(range(len(df))):
        video_path = df["video"][i]
        time_path = df["textGrid"][i]
        text_path = video_path[:-3]+"txt"
        swf, svf, evf, ewf = write_landmarks(video_path, time_path, text_path)
        mvf = (evf + svf) // 2
        if(ewf == 0):
            keep_flag.append(False)
        else:
            keep_flag.append(True)
        swc.append(swf)
        svc.append(svf)
        evc.append(evf)
        ewc.append(ewf)
        mvc.append(mvf)
        word_start, vowel_start, vowel_end, word_end = fu.read_textGrid(time_path)
        swt.append(word_start)
        svt.append(vowel_start)
        evt.append(vowel_end)
        ewt.append(word_end)
        ln_txt.append(text_path)
    df['word_start_frame'] = swc
    df['vowel_start_frame'] = svc
    df['vowel_end_frame'] = evc
    df['word_end_frame'] = ewc
    df['word_start_time'] = swt
    df['vowel_start_time'] = svt
    df['vowel_end_time'] = evt
    df['word_end_time'] = ewt
    df['vowel_duration_time'] = df['vowel_end_time'] - df['vowel_start_time']
    df['word_duration_time'] = df['word_end_time'] - df['word_start_time']
    df['mid_vowel_frame'] = mvc
    df['vowel_duration_frames'] = df['vowel_end_frame'] - df['vowel_start_frame'] + 1
    df['word_duration_frames'] = df['word_end_frame'] - df['word_start_frame'] + 1
#     print(len(swt), len(keep_flag))
    df['keep_flag'] = keep_flag
    df['landmarks'] = ln_txt
    
    df = df[df['keep_flag'] == True]
    df = df.drop(columns=['keep_flag'])
    df = df.reset_index(drop=True)
    print("Data size rduced to :", len(df))
    
    print("Finding the pairs")
    speech_type = ["conv"]
    for i in range(1, len(df)):
        if (df.iloc[i]['token'] == df.iloc[i-1]['token']) \
        and (df.iloc[i]['subject'] == df.iloc[i-1]['subject']) \
        and (df.iloc[i]['elicitation'] == df.iloc[i-1]['elicitation']) \
        and (df.iloc[i]['segmentation'] == df.iloc[i-1]['segmentation']+1):
            speech_type.append('clear')
        else :
            speech_type.append('conv')
#     len(speech_type)
    df['speech_type'] = speech_type
    
    pair_idx = []
    for i in tqdm(range(len(df)-1)):
        if((df.iloc[i]['speech_type'] == 'conv') and (df.iloc[i+1]['speech_type'] == 'clear') and (df['token'][i] == df['token'][i+1])):
            pair_idx.append(i)
            
    corresponding = []
    for i in range(len(df)):
        if i in pair_idx :
            corresponding.append(i+1)
        elif i-1 in pair_idx :
            corresponding.append(i-1)
        else :
            corresponding.append(-1)
    df['corresponding'] = corresponding
    
    print("Got info for the subject(s) : ", np.unique(df["subject"])[:])
    
    return df




"""
Write the landmarks and saves it to the defined text_path.
"""
def write_landmarks(video_path, time_path, text_path, n = 68):
  
    landmarks = []
    
    word_start, vowel_start, vowel_end, word_end = fu.read_textGrid(time_path)
    if (word_end == 0.0):
        return 0, 0, 0, 0
    
    video = cv2.VideoCapture(video_path)
    fps = 29 #video.get(cv2.CAP_PROP_FPS)
    
    swf =round(fps*word_start)
    svf =round(fps*vowel_start)
    evf =round(fps*vowel_end)
    ewf =round(fps*word_end)
    
    if video_path[:4] not in ["1101", "1110", "1202", "1203"]:
    #     if os.path.isfile(text_path):
    #         return swf, svf, evf, ewf

        mvf = (svf+evf)//2
        left = 10000
        top = 10000
        right = 0
        bottom = 0

        count = 0
        start_point = swf-(ewf-swf)
        end_point = ewf

        while(video.isOpened()):
            ret, frame = video.read()
            if ret and (start_point-1)<count<(end_point+1):
                face = ln._detect_face_dlib(frame)[0]
                left = min(left, face.left())
                top = min(top, face.top())
                right = max(right, face.right())
                bottom = max(bottom, face.bottom())
    #             break
            if count>end_point+1:
                break
            count += 1
        video.release()
        face = dlib.rectangle(left-10, top-10, right+10, bottom+30)                     

        count = 0
        didnt_find = []
        video = cv2.VideoCapture(video_path)
        while(video.isOpened()):
            ret, frame = video.read()
            if ret:
                try :
                preds, _ = ln.det.detect(frame, [face.left(), face.top(), face.right(), face.bottom()])
    #                 preds = ln.predictor(frame, face)
    #                 preds = ln._shape_to_array(preds)
    #                 print("prediction shape:", preds.shape)
                    landmarks.append(preds)
                except :
                    landmarks.append(landmarks[-1])
                    didnt_find.append(count)
                    print("couldn't find landmarks for frame:", count)
                    continue
                count += 1
            else:
                break
        video.release()

        if len(didnt_find)>0:
            for idx in didnt_find:
                landmarks[idx] = (landmarks[idx-1] + landmarks[idx + 1]) // 2

        landmarks = np.asarray(landmarks)
    #     print("landmarks shape:", landmarks.shape)
    #     print("init shape", landmarks.shape)
    #     landmarks = ((landmarks[2:]+landmarks[:-2])/4) + (landmarks[1:-1]/2) # 50% itself, 25% before and 25% after
    #     print("last shape", landmarks.shape)

        np.savetxt(text_path, landmarks.reshape(landmarks.shape[0], n*2))
    
    return swf, svf, evf, ewf



"""
Write the position of where maximum stretches occured. Like 10% into the word, 100% into the word etc.
"""
def write_max_features(data):
    df = data.copy(deep=True)
    vertical_stretch = []
    horizontal_stretch = []
    jaw = []
    eccentricity = []
    area = []
    
    vertical_stretch_position = []
    horizontal_stretch_position = []
    jaw_position = []
    eccentricity_position = []
    area_position = []
    for i in tqdm(range(len(df))) :
        video = df.iloc[i]['video']
        arg_vert_ln = 0
        arg_horz_ln = 0
        arg_jaw_ln = 0
        arg_e = 0
        arg_area_ln = 0
        svf = df.iloc[i]['vowel_start_frame']
        evf = df.iloc[i]['vowel_end_frame']+1
        if evf > svf and i not in [679,680,681]:
            dur = evf-svf
            ln_txt = df.iloc[i]['landmarks']
            landmarks = np.asarray(np.loadtxt(ln_txt)[svf:evf])
            landmarks = np.reshape(landmarks, (landmarks.shape[0], int(landmarks.shape[1]/2), 2))
            if(landmarks.shape[0] == 0):
                print(ln_txt,evf, svf)
            vert_ln = np.sqrt(np.square(landmarks[:,66,1] - landmarks[:,62,1]) 
                              + np.square(landmarks[:,66,0] - landmarks[:,62,0]))
            
            horz_ln = np.sqrt(np.square(landmarks[:,64,1] - landmarks[:,60,1]) 
                              + np.square(landmarks[:,64,0] - landmarks[:,60,0]))
            
            jaw_ln = np.sqrt(np.square(landmarks[:,8,1] - landmarks[:,33,1]) 
                             + np.square(landmarks[:,8,0] - landmarks[:,33,0]))
            
            major_axis = np.max([vert_ln, horz_ln], axis = 0)
            minor_axis = np.min([vert_ln, horz_ln], axis = 0)
            e = np.sqrt(1-(np.square(minor_axis)/np.square(major_axis)))
            
            area_ln = np.pi*horz_ln*vert_ln*0.25
#             print(vert_ln, horz_ln, jaw_ln, e, area_ln)
            arg_vert_ln = np.argmax(vert_ln)
            arg_horz_ln = np.argmax(horz_ln)
            arg_jaw_ln = np.argmax(jaw_ln)
            arg_e = np.argmax(e)
            arg_area_ln = np.argmax(area_ln)
        else:
            vert_ln = np.asarray([0, -1])
            horz_ln = np.asarray([0, -1])
            jaw_ln = np.asarray([0, -1])
            c = np.asarray([0, -1])
            e = np.asarray([0, -1])
            area_ln = np.asarray([0, -1])

        vertical_stretch.append(np.max(vert_ln, initial = 0))
        horizontal_stretch.append(np.max(horz_ln, initial = 0))
        jaw.append(np.max(jaw_ln, initial = 0))
        eccentricity.append(np.max(e, initial = 0))
        area.append(np.max(area_ln, initial = 0))

        vertical_stretch_position.append((arg_vert_ln+1)/dur)
        horizontal_stretch_position.append((arg_horz_ln+1)/dur)
        jaw_position.append((arg_jaw_ln+1)/dur)
        eccentricity_position.append((arg_e+1)/dur)
        area_position.append((arg_area_ln+1)/dur)
        
#     print(len(vertical_stretch), len(vertical_stretch_position))
            
#     print("shape of vertical_stretch : ", len(vertical_stretch))
    df['vertical_stretch'] = vertical_stretch
    df['horizontal_stretch'] = horizontal_stretch
    df['jaw'] = jaw
    df['eccentricity'] = eccentricity
    df['area']= area
    
    df['vertical_stretch_position'] = vertical_stretch_position
    df['horizontal_stretch_position'] = horizontal_stretch_position
    df['jaw_position'] = jaw_position
    df['eccentricity_position'] = eccentricity_position
    df['area_position']= area_position
    
    return df


if __name__ == "__main__":
    data = get_videos_info()
    data = get_landmarks_info(data)
    data = write_max_features(data)
    data.to_csv(opt.data_path)