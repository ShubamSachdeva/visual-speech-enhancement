# visual-speech-enhancement

This code is in support of our paper: "Plain-to-clear speech video conversion for enhanced intelligibility." 

Abstract: Clearly articulated speech, relative to plain-style speech, has been shown to improve intelligibility. We examine if visible speech cues in video can be systematically modified to enhance clear-speech features and improve intelligibility. We extract clear-speech visual features of English words varying in vowels produced by multiple male and female talkers. Via a frame-by-frame image-warping based video generation method with a controllable parameter (displacement factor), we apply the extracted clear-speech visual features to videos of plain speech to synthesize clear speech videos. We evaluate the generated videos using a robust, state of the art AI Lip Reader as well as human intelligibility testing. The contributions of this study are: (1) our pipeline successfully extracts relevant visual cues for video modifications across speech styles, and has achieved enhanced intelligibility for AI; (2) this work suggests that universal talker-independent clear-speech features may be utilized to modify any talker’s speech style; (3) we introduce “displacement factor” as a way of systematically scaling modifications between speech styles; and (4) the high definition generated videos make them perfect candidates for human-centric intelligibility and perceptual training studies. 



The following describes the different python files included in this package:

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
