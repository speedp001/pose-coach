import cv2
import json
import numpy as np

# ViTPose 17개 키포인트의 연결 정보 (COCO Keypoint Format)
LIMB_CONNECTIONS = [
        (0,5), (0,6),   # 코 어깨
        (5,6),          # 양 어깨
        (5,7),(7,9),    # 왼팔
        (6,8),(8,10),   # 오른팔
        (11,12),        # 양 골반
        (5,11),(6,12),  # 어깨→골반
        (11,13),(13,15),# 왼다리
        (12,14),(14,16), # 오른다리
        (0, 1), (0, 2),  # 코 → 눈
        (1, 3), (2, 4)  # 눈 → 귀
]

def draw_skeleton(keypoints, frame_size=(640, 480), radius=5, thickness=2):
    """
    ViTPose 17개 키포인트를 스켈레톤 형식으로 연결하여 검은 배경 위에 시각화하는 함수.
    - keypoints: (N, 2) 형태의 키포인트 리스트
    - frame_size: (width, height) 검은 배경 크기
    - radius: 키포인트 원 크기
    - thickness: 선의 두께
    """
    black_bg = np.zeros((frame_size[1], frame_size[0], 3), dtype=np.uint8)
    keypoint_color = (0, 255, 0)  # 초록색 (점)
    line_color = (255, 0, 0)  # 파란색 (선)

    # 키포인트 그리기
    for x, y in keypoints:
        cv2.circle(black_bg, (int(x), int(y)), radius, keypoint_color, -1)

    # 키포인트 연결선 그리기
    for (i, j) in LIMB_CONNECTIONS:
        pt1, pt2 = tuple(map(int, keypoints[i])), tuple(map(int, keypoints[j]))
        cv2.line(black_bg, pt1, pt2, line_color, thickness)

    return black_bg


def visualize_skeleton(input_video, output_video, user_output_json):
    """
    비디오 프레임에서 ViTPose 스켈레톤을 검은 배경에 시각화하여 새로운 비디오로 저장.
    - input_video: 원본 비디오 경로
    - output_video: 출력 비디오 경로
    - keypoints_list: 각 프레임의 keypoints 리스트
    """
    
    # 저장된 Keypoint 불러오기
    with open(user_output_json, "r") as f:
        user_data = json.load(f)

    # JSON에서 프레임별 키포인트 추출
    keypoints_list = [frame["keypoints"] for frame in user_data["frames"]]
    
    cap = cv2.VideoCapture(input_video)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))

    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame_idx >= len(keypoints_list):
            break

        keypoints = keypoints_list[frame_idx]
        vis_frame = draw_skeleton(keypoints, (frame_width, frame_height))

        out.write(vis_frame)
        frame_idx += 1

    cap.release()
    out.release()
    print(f"Skeleton visualization (black background) saved to {output_video}")
