import os
import sys
import numpy as np
import matplotlib.pyplot as plt


def read_textGrid(textGrid_path):
    content = open(textGrid_path, "r").readlines()
    num_lines = sum(1 for line in content)
    
    if num_lines == 46:
        start_word = content[16].split()[-1]
        end_word = content[40].split()[-1]
        start_vowel = content[20].split()[-1]
        end_vowel = content[36].split()[-1]
    elif num_lines == 34:
        start_word = content[16].split()[-1]
        end_word = content[28].split()[-1]
        start_vowel = content[20].split()[-1]
        end_vowel = content[27].split()[-1]
    else:
        #         raise NameError('Unknown format for file:', textGrid_path)
        print('Unknown format for file:', textGrid_path, 'returning all zeros')
        start_word = 0
        end_word = 0
        start_vowel = 0
        end_vowel = 0

    return float(start_word), float(start_vowel), float(end_vowel), float(end_word)


def visualize_landmarks(landmarks_list, plot_color = "blue"):
    if isinstance(landmarks_list, list):
        n = len(landmarks_list)
    elif isinstance(landmarks_list, np.ndarray):
        if landmarks_list.ndim == 2:
            n = 1
            landmarks_list = [landmarks_list]
        elif landmarks_list.ndim == 3:
            n = landmarks_list.shape[0]
        else:
            return
    else:
        return
    
    if n == 1:
        r = landmarks_list[0]
        fig, axs = plt.subplots()
        axs.scatter(r[:68,0], r[:68,1], s=10, color = plot_color)
        axs.plot(r[:17,0], r[:17,1], color = plot_color)
        axs.plot(r[17:22,0], r[17:22,1], color = plot_color)
        axs.plot(r[22:27,0], r[22:27,1], color = plot_color)
        axs.plot(r[27:36,0], r[27:36,1], color = plot_color)
        axs.plot(r[48:60,0], r[48:60,1], color = plot_color)
        axs.plot(r[60:,0], r[60:,1], color = plot_color)
        axs.invert_yaxis()
        plt.show()
        return
    
    fig, axs = plt.subplots(1, n, figsize = [n*5,5])
    for i in range(n):
        r = landmarks_list[i]
        axs[i].scatter(r[:68,0], r[:68,1], s=10, color = plot_color)
        axs[i].plot(r[:17,0], r[:17,1], color = plot_color)
        axs[i].plot(r[17:22,0], r[17:22,1], color = plot_color)
        axs[i].plot(r[22:27,0], r[22:27,1], color = plot_color)
        axs[i].plot(r[27:36,0], r[27:36,1], color = plot_color)
        axs[i].plot(r[48:60,0], r[48:60,1], color = plot_color)
        axs[i].plot(r[60:,0], r[60:,1], color = plot_color)
        axs[i].invert_yaxis()
    plt.show()


def visualize_stretches(landmarks_lists, labels = None):
    
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize = [25,5])
    n = len(landmarks_lists)
    
    if labels == None:
        labels = [str(i) for i in range(n)]
    
    for i in range(n):
        landmarks = np.asarray(landmarks_lists[i])
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
        
        ax1.plot(#np.linspace(0, clear_landmarks.shape[0] - 1, conv_landmarks.shape[0]),
            vert_ln, label=labels[i])
        
        ax2.plot(#np.linspace(0, clear_landmarks.shape[0] - 1, conv_landmarks.shape[0]),
            horz_ln, np.arange(landmarks.shape[0]), label=labels[i])
        
        ax3.plot(#np.linspace(0, clear_landmarks.shape[0] - 1, conv_landmarks.shape[0]),
            jaw_ln, label = labels[i])
        
        ax4.plot(#np.linspace(0, clear_landmarks.shape[0] - 1, conv_landmarks.shape[0]),
            e, label=labels[i])
        
        ax5.plot(#np.linspace(0, clear_landmarks.shape[0] - 1, conv_landmarks.shape[0]),
            area_ln, label=labels[i])
        
    bottom, top = ax1.get_ylim()
    if top > bottom:
        ax1.invert_yaxis()
    ax1.set_title("Vertical lip stretch \n lower position means more stretching")
    ax1.set_xlabel('frames')
    ax1.set_ylabel('Vertical stretch')
    
    bottom, top = ax2.get_ylim()
    if bottom <= 0:
        ax2.invert_yaxis()
    left, right = ax2.get_xlim()
    if left > right:
        ax2.invert_xaxis()
    ax2.set_title("Horizontal lip stretch \n Right position means more stretching")
    ax2.set_xlabel('Horiontal stretch')
    ax2.set_ylabel('frames')
    
    bottom, top = ax3.get_ylim()
    if top > bottom:
        ax3.invert_yaxis()
    ax3.set_title("Jaw displacement \n lower position means lower jaw position")
    ax3.set_xlabel('frames')
    ax3.set_ylabel('Jaw displacement')
    
    bottom, top = ax4.get_ylim()
    if bottom > top:
        ax4.invert_yaxis()
    ax4.set_title("Lip rounding \n higher position indicates more lip rounding")
    ax4.set_xlabel('frames')
    ax4.set_ylabel('Lip rounding')
    
    bottom, top = ax5.get_ylim()
    if bottom > top:
        ax5.invert_yaxis()
    ax5.set_title("Area \n higher position indicates more lip rounding")
    ax5.set_xlabel('frames')
    ax5.set_ylabel('Lip area')
    
    plt.legend()
    plt.show()
    