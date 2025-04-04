import os
import cv2
import json
import torch
import numpy as np

from tqdm import tqdm
from mmpose.datasets.dataset_info import DatasetInfo
from mmpose.apis import init_pose_model, inference_top_down_pose_model, vis_pose_result





##### 각도 계산 함수 정의
def calculate_angle(p1, p2, p3):
    """
    세 개의 keypoint 좌표를 받아 각도를 계산
    - p1: 시작점
    - p2: 기준점
    - p3: 끝점
    """
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)
    cosine_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    theta = np.arccos(np.clip(cosine_theta, -1.0, 1.0))
    return np.degrees(theta)





##### 기울기 계산 함수 정의
def calculate_slope(p1, p2):
    """
    두 점을 입력받아 기울기(slope)를 계산
    - p1, p2: [x, y] 좌표 리스트
    """
    if p2[0] - p1[0] == 0:
        return float('inf')
    else:
        return float((p2[1] - p1[1]) / (p2[0] - p1[0]))





##### bbox 계산 함수 정의 
def get_global_bbox(keypoints):
    """
    키포인트의 최대, 최소점을 한계점으로 bbox 설정
    keypoints: [ (N×2), (N×2), ... ] 모든 프레임의 원본 좌표 리스트
    """
    all_x = np.concatenate([frame[:, 0] for frame in keypoints])
    all_y = np.concatenate([frame[:, 1] for frame in keypoints])
    x_min, x_max = np.min(all_x), np.max(all_x)
    y_min, y_max = np.min(all_y), np.max(all_y)
    return [x_min, y_min, x_max, y_max]





##### 이미지 픽셀 좌표계 -> bbox 좌표계 변환 함수 정의
def transform_keypoints(keypoints, x_min, y_min, x_max, y_max):
    """
    원본 키포인트 좌표를 BBox 기준 좌표계(0,0 ~ max,max)로 변환
    - keypoints: 원본 프레임 기준 키포인트
    - bbox: [x_min, y_min, x_max, y_max]
    """
    bbox_width = x_max - x_min
    bbox_height = y_max - y_min
    transformed_keypoints = np.copy(keypoints)

    transformed_keypoints[:, 0] = keypoints[:, 0] - x_min
    transformed_keypoints[:, 1] = keypoints[:, 1] - y_min

    return transformed_keypoints





##### user_bbox를 label_bbox에 맞게 scale 변환 함수 정의
def user_scaled_keypoints(keypoints, user_bbox, label_bbox):
    """
    user bbox 크기를 label bbox 크기에 맞추기 위해 keypoint 좌표를 스케일링
    - keypoints: user_data에서 변환된 bbox 좌표계 기준 keypoints
    - user_bbox: [x_min, y_min, x_max, y_max] (유저 bbox)
    - label_bbox: [x_min, y_min, x_max, y_max] (정답 bbox)
    """
    user_width = user_bbox[2] - user_bbox[0]
    user_height = user_bbox[3] - user_bbox[1]
    
    label_width = label_bbox[2] - label_bbox[0]
    label_height = label_bbox[3] - label_bbox[1]

    scale_x = label_width / user_width
    scale_y = label_height / user_height

    scaled_keypoints = np.copy(keypoints)
    scaled_keypoints[:, 0] *= scale_x
    scaled_keypoints[:, 1] *= scale_y

    return scaled_keypoints





##### Load label JSON 함수 정의
def load_label_bbox(label_json_path):
    """
    label_data에서 bbox 좌표를 로드하는 함수
    - label_json_path: label JSON 파일 경로
    """
    with open(label_json_path, "r") as f:
        label_data = json.load(f)
    return label_data["bbox"]





##### ViTPose Model 함수 정의
def pose_extraction(config_file, checkpoint_file, input_video, output_json, output_video, label_output_json):
    
    # 디바이스 설정
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"device: {device}")

    # 모델 선언 및 데이터셋 정보 로드
    model = init_pose_model(config_file, checkpoint_file, device=device)
    dataset_info = DatasetInfo(model.cfg.dataset_info)

    # 비디오 로드
    cap = cv2.VideoCapture(input_video)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 비디오 저장
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))

    # 1차 루프: 모든 프레임 원본 keypoints 추출, 비디오 시각화 저장
    # 모든 프레임의 원본 키포인트(픽셀 좌표) 누적
    bbox_keypoints = []
    
    # frame_idx 보관
    frame_idxs = []
    frame_idx = 0

    for _ in tqdm(range(total_frames), desc="Processing Frames", unit="frame"):
        ret, frame = cap.read()
        if not ret:
            break

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Bounding Box 
        person_results = [{'bbox': np.array([0, 0, frame_width, frame_height])}]

        # top down 방식으로 pose estimation
        pose_results, _ = inference_top_down_pose_model(
            model, img_rgb, person_results, bbox_thr=None, format='xywh',
            dataset_info=dataset_info, return_heatmap=False
        )

        # keypoint 시각화 (단순히 영상에 표시)
        vis_frame = vis_pose_result(model, frame, pose_results, kpt_score_thr=0.3, show=False)

        # person 결과 (여기서는 1명 가정)
        for person in pose_results:
            original_keypoints = np.array(person["keypoints"])[:, :2]
            # => bbox_keypoints에 누적 (여기서는 transform/scale 하지 않음)
            bbox_keypoints.append(original_keypoints)
            frame_idxs.append(frame_idx)

        # 비디오에 스켈레톤 시각화된 프레임 저장
        out.write(vis_frame)
        frame_idx += 1

    # 1차 루프 끝
    cap.release()

    # 전체 프레임 기반 bbox 구하기
    if len(bbox_keypoints) == 0:
        print("[ERROR] No keypoints found.")
        out.release()
        return

    # 모든 프레임의 원본 좌표에 대한 글로벌 bbox
    global_bbox = get_global_bbox(bbox_keypoints)
    x_min, y_min, x_max, y_max = map(int, global_bbox)

    # user vs label 구분
    is_user_data = ("user_data/user_video" in input_video)
    if is_user_data:
        # label_bbox 로드
        label_bbox = load_label_bbox(label_output_json)

    # 2차 루프: transform/scale + angles 계산 + JSON 저장
    keypoints_results = []

    # bbox_keypoints와 frame_idxs는 같은 길이 (각 프레임당 하나씩)
    for i, original_kpts in enumerate(bbox_keypoints):
        f_idx = frame_idxs[i]

        # transform (픽셀→bbox 좌표계)
        transformed_kpts = transform_keypoints(original_kpts, x_min, y_min, x_max, y_max)

        # user라면 label_bbox 스케일 적용
        if is_user_data:
            scaled_kpts = user_scaled_keypoints(transformed_kpts, [x_min, y_min, x_max, y_max], label_bbox)
        else:
            scaled_kpts = transformed_kpts

        # angles 계산 (original_kpts 기준으로 계산)
        left_shoulder = original_kpts[5]
        right_shoulder = original_kpts[6]
        shoulder_center = (left_shoulder + right_shoulder) / 2
        left_hip = original_kpts[11]
        right_hip = original_kpts[12]
        hip_center = (left_hip + right_hip) / 2

        angles = [
            # 왼쪽 어깨 - 왼쪽 팔꿈치 - 왼쪽 손목
            calculate_angle(original_kpts[5], original_kpts[7], original_kpts[9]),

            # 오른쪽 어깨 - 오른쪽 팔꿈치 - 오른쪽 손목
            calculate_angle(original_kpts[6], original_kpts[8], original_kpts[10]),

            # 골반 중앙 - 어깨 중심 - 코
            calculate_angle(hip_center, shoulder_center, original_kpts[0]),

            # 왼쪽 골반 - 왼쪽 무릎 - 왼쪽 발목
            calculate_angle(left_hip, original_kpts[13], original_kpts[15]),

            # 오른쪽 골반 - 오른쪽 무릎 - 오른쪽 발목
            calculate_angle(right_hip, original_kpts[14], original_kpts[16]),

            # 왼쪽 힙 - 왼쪽 어깨 - 왼쪽 팔꿈치
            calculate_angle(original_kpts[11], original_kpts[5], original_kpts[7]),

            # 오른쪽 힙 - 오른쪽 어깨 - 오른쪽 팔꿈치
            calculate_angle(original_kpts[12], original_kpts[6], original_kpts[8]),

            # 왼쪽 어깨 - 오른쪽 어깨
            calculate_slope(original_kpts[5], original_kpts[6]),

            # 왼쪽 골반 - 오른쪽 골반
            calculate_slope(original_kpts[11], original_kpts[12])
        ]

        # keypoints_results에 저장
        keypoints_results.append({
            "frame_idx": f_idx,
            "keypoints": scaled_kpts.tolist(),
            "angles": angles
        })

    # 최종 JSON 구성
    json_data = {
        "bbox": [x_min, y_min, x_max, y_max],
        "frames": keypoints_results
    }

    with open(output_json, "w") as f:
        json.dump(json_data, f, indent=4)

    out.release()
    print(f"Processing complete! Keypoints saved to {output_json}, Video saved to {output_video}")