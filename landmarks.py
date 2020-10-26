import numpy as np
import torch
import os
from os import listdir
from os.path import isfile, join
import sys
import dlib
import cv2
from SAN.san_api import SanLandmarkDetector
import warnings


warnings.filterwarnings('ignore')

model_path = "SAN/checkpoint_49.pth.tar"

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

det = SanLandmarkDetector(model_path, device)

detector = dlib.get_frontal_face_detector()

# Dlib landmarks predictor
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


def _detect_face_dlib(image):
    """
    Function to detect faces in the input image with dlib

    :param image: grayscale image with face(s)
    :return: dlib regtangles object with detected face regions
    """
    return detector(image, 1)


def _detect_face_opencv(image, cascade):
    """
    Function to detect faces in the input image with OpenCV
    :param image: grayscale image with face(s)
    :param cascade: OpenCV CascadeClassifier object
    :return: array of detected face regions
    """
    if image.ndim > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(5, 5))
    if len(rects) == 0:
        return []
    rects[:, 2:] += rects[:, :2]
    return sorted(rects, key=lambda rect: rect[2] - rect[0], reverse=True)

def _array_to_dlib_rectangles(rects_array):
    """
    Function to convert array of rectangles (in format [[x1,y1,w1,h1],[x2,y2,w2,h2],...]) to dlib regtangles objects.
    Usually input array is a results of OpenCV face detection and output dlib regtangles are used for landmark detection.
    :param rects_array: array with results of OpenCV face detection
    :return: dlib rectangles object
    """
    rects_dlib = dlib.rectangles()
    for (left, top, right, bottom) in rects_array:
        rects_dlib.append(dlib.rectangle(
            int(left),
            int(top),
            int(right),
            int(bottom)))
    return rects_dlib

def _shape_to_array(shape):
    """
    Function to convert dlib shape object to array
    :param shape:
    :return:
    """
    return np.array([[p.x, p.y] for p in shape.parts()], dtype=float)

def find_landmarks(image, visualise=False, opencv_facedetector=False):
    """
    Function to find face landmarks (coordinates of nose, eyes, mouth etc) with dlib face landmarks predictor.
    :param image: greyscale image which contains face
    :param predictor: dlib object, shape predictor
    :param visualise: flag to draw detected landmarks on top of the face
    :param opencv_facedetector: flag to switch face detection to OpenCV inplace of dlib HOG detector
    :return: dlib shape object with coordinates for 68 facial landmarks
    """
    if opencv_facedetector:
        # Use OpenCV face detection for a really fast but less accurate results
        faces = _detect_face_opencv(image, face_cascade)
        dets = _array_to_dlib_rectangles(faces)
    else:
        # Use dlib face detection for a more precise face detection,
        # but with lower fps rate
        dets = _detect_face_dlib(image)

    try:
        shape = predictor(image, dets[0])
        i = 0
        if visualise:
            while i < shape.num_parts:
                p = shape.part(i)
                cv2.circle(image, (p.x, p.y), 2, (0, 255, 0), 2)
                i += 1
    except:
        shape = None
    return _shape_to_array(shape)


def get_landmarks(image, n=68, backend = 'dlib'):
    if backend == 'dlib':
        return find_landmarks(image)
    face = _detect_face_dlib(image)[0]
    left = face.left()
    top = face.top()
    right = face.right()
    bottom = face.bottom()
    landmarks, scores = det.detect(image, [left, top, right, bottom])
    if n == 66:  # GANnotation uses 66 landmarks instead of 68
        landmarks = np.delete(landmarks, [60, 64], axis=0)
    return landmarks


def get_landmarks_from_video(video_path, n=68):
    landmarks = np.zeros((n, 2, 1))
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    while success:
        preds = get_landmarks(image, n)
        landmarks = np.concatenate((landmarks, preds.reshape(n, 2, 1)), axis=2)
        success, image = vidcap.read()
    return landmarks[:, :, 1:]


def write_landmarks(landmarks, text_path):
    n = landmarks.shape[0]
    np.savetxt(text_path, landmarks.reshape((n * 2, -1)).transpose())


def load_landmarks(text_path):
    landmarks = np.loadtxt(text_path)
    n = landmarks.shape[0] / 2
    landmarks = landmarks.reshape((n, 2, -1))
    return landmarks


def interpolate_landmarks(source, destination, dst_ratio, impact=1):
    diff = (destination - source) * impact
    return source + (dst_ratio * diff)


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


def get_mid_vowel_image(video_path, time_path):
    _, vowel_start, vowel_end, _ = read_textGrid(time_path)
    video = cv2.VideoCapture(video_path)
    fps = 29
    mid_frame = int(fps * (vowel_start + (vowel_end - vowel_start) / 2))

    video.set(cv2.CAP_PROP_POS_FRAMES, mid_frame - 1)
    res, image = video.read()
    return image


def get_mid_vowel_landmarks(video_path, time_path, n=68):
    image = get_mid_vowel_image(video_path, time_path)
    return get_landmarks(image, n)


def get_mid_vowel_face(video_path, time_path):
    image = get_mid_vowel_image(video_path, time_path)

    face = _detect_face_dlib(image)[0]
    left = face.left()
    top = face.top()
    right = face.right()
    bottom = face.bottom()
    return image[left:right, top:bottom]


def get_alignment_angle(landmarks):
    right_eye_center = np.mean(landmarks[36:42], axis=0, dtype=np.int)
    left_eye_center = np.mean(landmarks[42:48], axis=0, dtype=np.int)
    dY = right_eye_center[1] - left_eye_center[1]
    dX = right_eye_center[0] - left_eye_center[0]
    angle = np.degrees(np.arctan2(dY, dX)) - 180
    return np.radians(angle)

def rotate_landmarks(landmarks, theta):
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))
    return np.matmul(landmarks, R)


def align_landmarks(landmarks):
    angle = get_alignment_angle(landmarks)
    return rotate_landmarks(landmarks, angle)


def normalize_landmarks(landmarks):
#     landmarks = landmarks - landmarks[33] # bringing nose to the origin
    landmarks = align_landmarks(landmarks)
    right_eye_center = np.mean(landmarks[36:42], axis=0, dtype=np.int)
    left_eye_center = np.mean(landmarks[42:48], axis=0, dtype=np.int)
    eyes_center = (left_eye_center + right_eye_center)/2
    nose = landmarks[33]
    dY = np.abs(nose[1] - eyes_center[1])
    dX = np.abs(right_eye_center[0] - left_eye_center[0])
    landmarks = landmarks - landmarks[33] # bringing nose to the origin
    
    return landmarks/[dX, dY]

