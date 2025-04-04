from utils.pose_extraction import pose_extraction # type: ignore
from utils.pose_evaluation import pose_evaluation # type: ignore
from utils.pose_counter import pose_counter # type: ignore
from utils.rotation import rotation_90_inplace # type: ignore
from utils.draw_keypoints import visualize_skeleton # type: ignore

user_name = input("English Name: ")
side = input("Side(left/right): ")
ViTPose_model = input("ViTPose model(small/base/large/huge): ")
# ViTPose_model = "small"
# print(f"Name: {user_name}, Side: {side}", f"ViTPose model: {ViTPose_model}")

##### Model configuration paths
CONFIG_FILE = f"./ViTPose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_{ViTPose_model}_coco_256x192.py"
CHECKPOINT_FILE = f"./checkpoints/vitpose_{ViTPose_model}.pth"

##### Video and JSON file paths
# Paths for label video and label keypoints
LABEL_INPUT_VIDEO = f"./label_data/label_video/label_{side}.mp4"
LABEL_OUTPUT_JSON = f"./label_data/label_keypoints/label_keypoints_{side}.json"
LABEL_OUTPUT_VIDEO = f"./label_data/label_keypoints/label_video_{side}.mp4"

# Paths for user video and keypoints
USER_INPUT_VIDEO = f"./user_data/user_video/{user_name}_{side}.mp4"
USER_OUTPUT_JSON = f"./user_data/user_keypoints/{user_name}_keypoints_{side}.json"
USER_OUTPUT_VIDEO = f"./user_data/user_keypoints/{user_name}_video_{side}.mp4"
USER_OUTPUT_VISUALIZED = f"./user_data/user_keypoints/{user_name}_visualization_{side}.mp4"  # Video with only keypoints visualized

# ##### Rotate 90 degrees clockwise if input is an iOS-captured video
# rotation_90_inplace(USER_INPUT_VIDEO)

# ##### Visualize keypoints and save as a video
# visualize_skeleton(USER_INPUT_VIDEO, USER_OUTPUT_VISUALIZED, USER_OUTPUT_JSON)
# print(f"Keypoint visualization saved: {USER_OUTPUT_VISUALIZED}")

# ##### Run pose estimation
# # Label pose estimation
# pose_extraction(CONFIG_FILE, CHECKPOINT_FILE, LABEL_INPUT_VIDEO, LABEL_OUTPUT_JSON, LABEL_OUTPUT_VIDEO, None)

# # User pose estimation
# pose_extraction(CONFIG_FILE, CHECKPOINT_FILE, USER_INPUT_VIDEO, USER_OUTPUT_JSON, USER_OUTPUT_VIDEO, LABEL_OUTPUT_JSON)

##### Run pose evaluation and feedback visualization
keypoint_RMSE, angle_RMSE, accuracy_final_score = pose_evaluation(USER_OUTPUT_JSON, LABEL_OUTPUT_JSON, USER_OUTPUT_VIDEO, LABEL_OUTPUT_VIDEO)

##### Run repetition count and speed evaluation
user_count, label_count, speed_final_score = pose_counter(USER_OUTPUT_JSON, LABEL_OUTPUT_JSON)

print("Accuracy score: ", accuracy_final_score)
print("User reps: ", user_count)
# print("User reps: ", label_count)
print("Speed score(Speed score: ", speed_final_score)
