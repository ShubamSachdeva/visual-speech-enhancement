# these are high level variables. Use these to obtain results similar to the ones reported.
# multiple combinations of these parameters have been tested on multiple videos.
# These are the ones which give the best results.

# degree of polynomial regression used for smoothing landmarks
degree = 3 


# points on the face landmarks those are fixed. These points are corner of eyes and nose tip.
stablePntsIDs = [33, 36, 39, 42, 45] 


# the path to the data
data_path = "data_info.csv" 


# the path to the data
generated_data_path = "modified_videos_info.csv" 


# two options are ratio and warp 
destination_landmarks_method = "warp"


# the other option is False. True indicates that for the particular word (say "kid") only the most relavant feature is applied (horizontal stretch for kid and keyed. Vertical stretch and jaw lowering for cud and cod. And vertical stretch for cooed and could.)
defining_features = True 


# the subjects to test
test_subjects = ["1101", "1110", "1202", "1203"]
