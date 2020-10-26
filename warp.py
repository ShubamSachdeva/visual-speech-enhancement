import numpy as np
import os
from os import listdir
from os.path import isfile, join
import sys
import dlib
import skimage
import skimage.transform
import cv2
import math
from PIL import Image

#landmarks
import landmarks as ln
import options as opt
import preprocessing as prep

#warping
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline


"""
# crop the image to be a square. Square is the best to apply any warping.
# This function is no lnger used anywhere.
"""
def square_image(image):
    h, w, c = image.shape
    pad = int((w-h)/2)
#     print(h, w, pad, pad, w-pad)
    return image[:, pad:w-pad]


# def frame_borders(image):
#     h, w = image.shape[:2]
#     border_array = np.asarray([[0, 0]])
#     y_array = np.linspace(0, h-1, 4, dtype=np.int)
#     x_array = np.linspace(0, w-1, 4, dtype=np.int)
#     for y in y_array:
#         border_array = np.concatenate((border_array, [[y, 0]], [[y, w-1]]), axis=0)
#     for x in x_array:
#         border_array = np.concatenate((border_array, [[0, x]], [[h-1, x]]), axis= 0)
#     return border_array[1:]




"""
# Returns the corner points and center of each of the 4 edges of the image to concatenate while warping
"""
def frame_borders(image):
    h, w = image.shape[:2]    
    return np.array([(0,0), (w/2,0), (w-1,0), (w-1,h/2), ( w-1, h-1 ), ( w/2, h-1 ), (0, h-1), (0,h/2) ])


"""
# this function applies the warping on image according to the change.
# the change is defined as the ratio that when multiplied with the conversational landmarks gives the clear landmarks.
"""
def face_warp_from_change(src_face, src_ln, change_ratio, dst_ratio = 1, impact = 1, only_desired_features = True):
    
    src_face = square_image(src_face)
    output_shape = src_face.shape[:2]
    
    src_face_coord = np.asarray(src_ln, dtype = float)
    
    #tip of the nose
    src_face_coord_33 = src_face_coord[33]
    
    #normalize
    src_face_coord = src_face_coord - src_face_coord_33
    
    #get destination landmarks
    dst_face_coord = src_face_coord*change_ratio
    
    #reverse normalize
    src_face_coord = src_face_coord + src_face_coord_33
    dst_face_coord = dst_face_coord + src_face_coord_33
    dst_ln = dst_face_coord.copy()
    
    
    if only_desired_features:
        dst_face_coord[0:6] = src_face_coord[0:6]
        dst_face_coord[11:48] = src_face_coord[11:48]
    
    diff_face_coord = (dst_face_coord - src_face_coord)*impact
    dst_face_coord = src_face_coord + diff_face_coord*dst_ratio
    
    src_face_coord = np.append(src_face_coord, frame_borders(src_face), axis = 0)
    dst_face_coord = np.append(dst_face_coord, frame_borders(src_face), axis = 0)
    
    warp_trans = skimage.transform.PiecewiseAffineTransform()
    warp_trans.estimate(dst_face_coord, src_face_coord)
    warped_face = skimage.transform.warp(src_face, warp_trans, output_shape=output_shape, mode='reflect')
    
    return dst_face_coord[:68], warped_face 



"""
# Given landmarks, it stretches the lips horizontally only.
# First it calculates the actual horizontal stretch and then given the scale; like 1.25; increases the stretch to 1.25 times
"""
def apply_horizontal_stretch(landmarks, scale_factor):
    src_mouth = landmarks[48:]
    src_mouth_center = np.mean(src_mouth, axis=0, dtype=np.int)
    src_mouth = src_mouth - src_mouth_center

    src_left_lip_tip = np.mean([src_mouth[0], src_mouth[12]], axis=0, dtype=np.int)
    src_right_lip_tip = np.mean([src_mouth[6], src_mouth[16]], axis=0, dtype=np.int)
    
    dY = src_right_lip_tip[1] - src_left_lip_tip[1]
    dX = src_right_lip_tip[0] - src_left_lip_tip[0]
    angle = np.degrees(np.arctan2(dY, dX)) - 180
    theta = np.radians(angle)

    aligned_src_mouth = ln.rotate_landmarks(src_mouth, theta)
    aligned_src_mouth[:,0] = aligned_src_mouth[:,0]*scale_factor

    src_mouth = ln.rotate_landmarks(aligned_src_mouth, -theta)
    src_mouth = src_mouth+src_mouth_center
    
    return_landmarks = np.concatenate((landmarks[:48], src_mouth), axis=0)
    
    return return_landmarks
    
    

"""
# Given landmarks, it stretches the lips vertically only.
# First it calculates the actual vertically stretch and then given the scale; like 1.25; increases the stretch to 1.25 times
"""
def apply_vertical_stretch(landmarks, scale_factor):
    src_mouth = landmarks[48:]
    
    left_top_lip_size = np.linalg.norm(landmarks[61]-landmarks[50])
    middle_top_lip_size = np.linalg.norm(landmarks[62]-landmarks[51])
    right_top_lip_size = np.linalg.norm(landmarks[63]-landmarks[52])
    left_bottom_lip_size = np.linalg.norm(landmarks[58]-landmarks[67])
    middle_bottom_lip_size = np.linalg.norm(landmarks[57]-landmarks[66])
    right_bottom_lip_size = np.linalg.norm(landmarks[56]-landmarks[65])
#     jaw_lowerings = []
#     for i in range(5,12):
#         jaw_lowerings.append(landmarks[58,1]-landmarks[i,1])
    
    origin = landmarks[51]
#     origin = np.mean((landmarks[51], landmarks[62]), axis=0, dtype=np.int)
    src_mouth = src_mouth - origin
    
    src_left_lip_tip = np.mean([src_mouth[0], src_mouth[12]], axis=0, dtype=np.int)
    src_right_lip_tip = np.mean([src_mouth[6], src_mouth[16]], axis=0, dtype=np.int)
    
    dY = src_right_lip_tip[1] - src_left_lip_tip[1]
    dX = src_right_lip_tip[0] - src_left_lip_tip[0]
    angle = np.degrees(np.arctan2(dY, dX)) - 180
    theta = np.radians(angle)

    aligned_src_mouth = ln.rotate_landmarks(src_mouth, theta)
    aligned_src_mouth[:,1] = aligned_src_mouth[:,1]*scale_factor

    src_mouth = ln.rotate_landmarks(aligned_src_mouth, -theta)
    src_mouth = src_mouth+origin
    
    return_landmarks = np.concatenate((landmarks[:48], src_mouth), axis=0)
    return_landmarks[61,1] = return_landmarks[50,1] + left_top_lip_size
    return_landmarks[62,1] = return_landmarks[51,1] + middle_top_lip_size
    return_landmarks[63,1] = return_landmarks[52,1] + right_top_lip_size
    
    return_landmarks[67,1] = return_landmarks[58,1] - left_bottom_lip_size
    return_landmarks[66,1] = return_landmarks[57,1] - middle_bottom_lip_size
    return_landmarks[65,1] = return_landmarks[56,1] - right_bottom_lip_size
    
#     for i,j in enumerate(range(5,12)):
#         return_landmarks[j,1] = jaw_lowerings[i] + return_landmarks[58,1]
        
    return return_landmarks
    
    
"""    
# Given landmarks, it lowers the jaw only.
# First it calculates the actual jaw positoin respective to the nose tip
# and then given the scale; like 1.25; lowers the jaw by 1.25 times
"""
def apply_jaw_drop(landmarks, scale_factor):
    src_mouth = landmarks[48:]
    jaw = landmarks[5:12]
    origin = landmarks[51]
#     origin = np.mean((landmarks[51], landmarks[62]), axis=0, dtype=np.int)
    jaw = jaw - origin
    
    src_left_lip_tip = np.mean([src_mouth[0], src_mouth[12]], axis=0, dtype=np.int)
    src_right_lip_tip = np.mean([src_mouth[6], src_mouth[16]], axis=0, dtype=np.int)
    
    dY = src_right_lip_tip[1] - src_left_lip_tip[1]
    dX = src_right_lip_tip[0] - src_left_lip_tip[0]
    angle = np.degrees(np.arctan2(dY, dX)) - 180
    theta = np.radians(angle)

    aligned_jaw = ln.rotate_landmarks(jaw, theta)
    aligned_jaw[:,1] = aligned_jaw[:,1]*scale_factor

    jaw = ln.rotate_landmarks(aligned_jaw, -theta)
    jaw = jaw+origin
    
    return_landmarks = np.concatenate((landmarks[:5], jaw, landmarks[12:68]), axis=0)
    
    return return_landmarks



"""
# Given source and destination landmarks, it transforms the destination landmarks according to the source landmarks.
# It brings the nose tip to same coordinates, make the head size equal, etc (normalization).
"""
def scale_landmarks(dst, src):    
    #src landmarks
    src_landmarks = src.copy()
    src_landmarks = ln.align_landmarks(src_landmarks)
    right_eye_center = np.mean(src_landmarks[36:42], axis=0, dtype=np.int)
    left_eye_center = np.mean(src_landmarks[42:48], axis=0, dtype=np.int)
    eyes_center = (left_eye_center + right_eye_center)/2
    nose = src_landmarks[33]
    src_dY = np.abs(nose[1] - eyes_center[1])
    src_dX = np.abs(right_eye_center[0] - left_eye_center[0])
#     landmarks = landmarks - landmarks[33] # bringing nose to the origin

    #dst landmarks
    dst_org_nose = dst[33]
    dst_landmarks = dst.copy()
    dst_angle = ln.get_alignment_angle(dst_landmarks)
    dst_landmarks = ln.rotate_landmarks(dst_landmarks, dst_angle)
    right_eye_center = np.mean(dst_landmarks[36:42], axis=0, dtype=np.int)
    left_eye_center = np.mean(dst_landmarks[42:48], axis=0, dtype=np.int)
    dst_nose = dst_landmarks[33]
    eyes_center = (left_eye_center + right_eye_center)/2
    dst_dY = np.abs(dst_nose[1] - eyes_center[1])
    dst_dX = np.abs(right_eye_center[0] - left_eye_center[0])
    
    #making changes
    dst_landmarks = dst_landmarks-dst_nose
    dst_landmarks = (dst_landmarks/[dst_dX, dst_dY])*[src_dX, src_dY]
    dst_landmarks = dst_landmarks + dst_nose
    dst_landmarks = ln.rotate_landmarks(dst_landmarks, -dst_angle)
    
    #just a safety check (dst_org_nose should be equal to dst_landmarks[33])
    dst_landmarks = dst_landmarks + (dst_org_nose - dst_landmarks[33])
    
    return dst_landmarks


"""
# The way that the warping is applied is that it transforms the destination landmarks to source landmarks then copies the
# clear landmarks. Applies to the conversational landmarks.
# If word == None, every strecth is applied. If the word is specified, the respective strecth is applied.
"""
def face_warp_copy_landmarks(src_face, face_ln, src_ln, dst_ln, dst_ratio = 1, impact = 1, 
                             warp_method = 'triangulation', word = None):
#     dst_face = square_image(dst_face)
#     src_face = square_image(src_face)
    
    output_shape = src_face.shape[:2]
    
    src_face_coord = prep.frame_borders(np.asarray(src_ln, dtype = float))
    dst_face_coord = prep.frame_borders(np.asarray(dst_ln, dtype = float))
    face_coord = prep.frame_borders(np.asarray(face_ln, dtype = float))
    
    tform =  skimage.transform.estimate_transform('similarity', dst_face_coord[opt.stablePntsIDs, :], face_coord[opt.stablePntsIDs, :])
    dst_face_coord = tform(dst_face_coord)[:68]
    tform =  skimage.transform.estimate_transform('similarity', src_face_coord[opt.stablePntsIDs, :], face_coord[opt.stablePntsIDs, :])
    src_face_coord = tform(src_face_coord)[:68]
    face_coord = face_coord[:68]
    
    diff_face_coord = (dst_face_coord - src_face_coord)*impact
    dst_face_coord = src_face_coord + (diff_face_coord*dst_ratio)
    
    # impact factor of essentially zero for everywhere except lips, and jaw
    dst_face_coord[0:6] = face_coord[0:6]
#         dst_face_coord[6:11,0] = src_face_coord[6:11,0]
    dst_face_coord[11:48] = face_coord[11:48]
    
    if (word in ['keyed', 'kid']):
        dst_face_coord[48:,1] = face_coord[48:,1]
        dst_face_coord[6:11] = face_coord[6:11]
    elif (word in ['could', 'cooed']):
        dst_face_coord[48:,0] = face_coord[48:,0]
        dst_face_coord[6:11] = face_coord[6:11]
    elif (word in ['cod', 'cud']):
        dst_face_coord[48:,0] = face_coord[48:,0]
        dst_face_coord[6:11:,0] = face_coord[6:11:,0]
        
#     selection = [49, 53, 55, 59]
#     dst_face_coord = dst_face_coord[np.setdiff1d(np.arange(dst_face_coord.shape[0]), selection)]
#     src_face_coord = src_face_coord[np.setdiff1d(np.arange(src_face_coord.shape[0]), selection)]
    
    face_coord = np.append(face_coord, frame_borders(src_face), axis = 0)
    dst_face_coord = np.append(dst_face_coord, frame_borders(src_face), axis = 0)
    
    warped_image = image_warp_affine(src_face, face_coord, dst_face_coord)
        
#     print("dst shape is ", dst_face_coord[:68].shape)
        
    return dst_face_coord[:68], warped_image



"""
# The way that the warping is applied is that first we calculate the stretch like vertical stretch for the conversational
# landmarks and then calculate it for the clear landmarks.
# The clear landmarks is then applied to the conversational landmarks using the impact factor.
# If word == None, every strecth is applied. If the word is specified, the respective strecth is applied.
"""
def face_warp_apply_stretches(src_face, face_ln, src_ln, dst_ln, dst_ratio = 1, impact = 1, 
                              warp_method = 'triangulation', word = None):
#     dst_face = square_image(dst_face)
#     src_face = square_image(src_face)
    
    output_shape = src_face.shape[:2]
    
    face_coord = np.asarray(face_ln, dtype = float)
    src_face_coord = np.asarray(src_ln, dtype = float)
    dst_face_coord = np.asarray(dst_ln, dtype = float)
    
    src_face_coord = scale_landmarks(src_face_coord, face_coord)
    dst_face_coord = scale_landmarks(dst_face_coord, face_coord)
    
    org = src_face_coord.copy()
    intermediate_dst = src_face_coord.copy()
    
    if (word in ['keyed', 'kid']) or (word == None):
        #horizontal stretch
        src_left_lip_tip = np.mean([src_face_coord[48], src_face_coord[60]], axis=0)
        src_right_lip_tip = np.mean([src_face_coord[54], src_face_coord[64]], axis=0)
        src_horz_stretch = np.linalg.norm(src_left_lip_tip - src_right_lip_tip)
        
        dst_left_lip_tip = np.mean([dst_face_coord[48], dst_face_coord[60]], axis=0)
        dst_right_lip_tip = np.mean([dst_face_coord[54], dst_face_coord[64]], axis=0)
        dst_horz_stretch = np.linalg.norm(dst_left_lip_tip - dst_right_lip_tip)
        
        stretch_diff = dst_horz_stretch - src_horz_stretch
        stretch_to_apply = src_horz_stretch + ((stretch_diff*impact)*dst_ratio)
        stretch_scale = stretch_to_apply/src_horz_stretch
        
        intermediate_dst = apply_horizontal_stretch(intermediate_dst, stretch_scale)
    
    if (word in ['could', 'cooed', 'cud', 'cod']) or (word == None):
        #vertical stretch
        src_lip_top = np.mean([src_face_coord[51], src_face_coord[62]], axis=0)
        src_lip_bottom = np.mean([src_face_coord[57], src_face_coord[66]], axis=0)
        src_vert_stretch = np.linalg.norm(src_lip_bottom - src_lip_top)
        
        dst_lip_top = np.mean([dst_face_coord[51], dst_face_coord[62]], axis=0)
        dst_lip_bottom = np.mean([dst_face_coord[57], dst_face_coord[66]], axis=0)
        dst_vert_stretch = np.linalg.norm(dst_lip_bottom - dst_lip_top)
        
        stretch_diff = dst_vert_stretch - src_vert_stretch
        stretch_to_apply = src_vert_stretch + ((stretch_diff*impact)*dst_ratio)
        stretch_scale = stretch_to_apply/src_vert_stretch
        
        intermediate_dst = apply_vertical_stretch(intermediate_dst, stretch_scale)
    
    if (word in ['cud', 'cod']) or (word == None):
        #jaw lowering
        src_lip_top = np.mean([src_face_coord[51], src_face_coord[62]], axis=0)
        src_jaw = np.mean(src_face_coord[7:10], axis=0)
        src_jaw_stretch = np.linalg.norm(src_jaw - src_lip_top)
        
        dst_lip_top = np.mean([dst_face_coord[51], dst_face_coord[62]], axis=0)
        dst_jaw = np.mean(dst_face_coord[7:10], axis=0)
        dst_jaw_stretch = np.linalg.norm(dst_jaw - dst_lip_top)
        
        stretch_diff = dst_jaw_stretch - src_jaw_stretch
        stretch_to_apply = src_jaw_stretch + ((stretch_diff*impact)*dst_ratio)
        stretch_scale = stretch_to_apply/src_jaw_stretch
        
        intermediate_dst = apply_jaw_drop(intermediate_dst, stretch_scale)

    dst_face_coord = intermediate_dst
    
    face_coord = np.append(face_coord, frame_borders(src_face), axis = 0)
    dst_face_coord = np.append(dst_face_coord, frame_borders(src_face), axis = 0)
    
    warped_image = image_warp_affine(src_face, face_coord, dst_face_coord)
        
    return dst_face_coord[:68], warped_image



"""
# the main function of the file. This is the called everywhere with different parameters.
Parameters:
src_face : The image to be warped
face_ln : Landmarks of the src_face
src_ln = Conversational landmarks.
dst_ln : Clear landmarks
dst_ratio : Goes from 0 to 1. 0 means just copy the src_ln. 1 means just copy the dst_ln. Anything between is interpolation between src_ln to dst_ln
impact : The impact factor to be applied
method: apply_streches vs copy_landmarks. The two functions are defined above
warp_method : triangulation vs piecewise-affine. The methods are defined below.
"""
def face_warp(src_face, face_ln, src_ln, dst_ln, dst_ratio = 1, impact = 1, 
              method = opt.apply_change_method, warp_method = opt.warp_method, word = None):
    if method == 'apply_stretches':
        return_landmarks, warp_image = face_warp_apply_stretches(src_face, face_ln, src_ln, dst_ln, dst_ratio = 1, 
                                                                 impact = 1, word = None)
    elif method == 'copy_landmarks':
        return_landmarks, warp_image = face_warp_copy_landmarks(src_face, face_ln, src_ln, dst_ln, dst_ratio = 1, 
                                                                impact = 1, word = None)
        
    return return_landmarks, warp_image
    
   
    
"""
It warps the image usingpiecewise affine transform.
image is the image to be warped
src_points are the starting points
dst_points are the destinatin points.
The affine transform is calculated from src to dst and then applied to the image.
"""
def image_warp_affine(image, src_points, dst_points):
    output_shape = image.shape[:2]
    warp_trans = skimage.transform.PiecewiseAffineTransform()
    warp_trans.estimate(dst_points, src_points)
    warped_face = skimage.transform.warp(image, warp_trans, output_shape=output_shape, mode='reflect')
#     return warped_face*255
#     print('max: ', warped_face.max())
    warped_face *= 255 #/warped_face.max() 
    
    return np.uint8(warped_face)




