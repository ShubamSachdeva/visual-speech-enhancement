# visual-speech-enhancement

This code is in support of our paper: "Plain-to-clear speech video conversion for enhanced intelligibility." The following describes the different python files included in this package:

    face_utils.py: This file contains various functions related to extracting vowel start and duration in both plain and clear speech, visualize facial landmarks and visualize the stretches going from plain speeck to clear speech.
    face_warp.py: This file contains the code for performing facial warping, which is a key step in the process of enhancing speech intelligibility. Optionally, it can also stretch the speech to match the word duration.
    landmarks.py: This file contains code for detecting facial landmarks, which are used for aligning the face in the video.
    options.py: This file contains various options (configurations) and settings that can be adjusted to customize the behavior of the code.
    prepare_data.py: This file contains code for preparing the data, including loading the videos, getting durations from audio data, extracting key type of stretches, and putting it all into a dataframe.
    preprocessing.py: This file contains code for preprocessing the video and audio data, including smoothing the landmarks temporally.
    warp.py: This file contains the main code for performing the plain-to-clear speech conversion. The file has functions for warping the face by either chosing to apply all stretches from plain to clear speech or just the most dominant one. 

To use this code, you will need to have Python and the following libraries installed: dlib, imutils, numpy, opencv-python, scipy, and librosa.
You can run the code by calling the 'warp.py' file and passing the necessary video file path and then the code will process the video and return the enhanced intelligibility video.

If you use this code, please cite our paper in any publications resulting from its use.
