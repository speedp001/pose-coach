import os
import cv2

##### 입력 비디오 형식 확인
# video_path = "/Users/sang-yun/Downloads/example.mp4"
# cap = cv2.VideoCapture(video_path)

# ret, frame = cap.read()
# if ret:
#     print("Frame shape:", frame.shape)  # (height, width, channels)

# cap.release()





##### 입력 비디오 회전
def rotation_90_inplace(input_path):
    """
    iOS 등에서 회전 메타데이터만 기록된 비디오를
    90도 시계 방향으로 회전하여 
    최종적으로 '같은 파일 이름'에 저장하는 함수.
    """

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {input_path}")
        return

    # 원본 속성 가져오기
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)

    # 회전된 결과를 임시 파일에 쓰기 (ex: "input.mp4_temp")
    base, ext = os.path.splitext(input_path)
    temp_path = base + "_temp" + ext

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # 90도 시계 방향 회전 → 해상도 (orig_h, orig_w)
    out = cv2.VideoWriter(temp_path, fourcc, fps, (orig_h, orig_w))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 90도 시계 방향 회전
        rotated_frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        out.write(rotated_frame)

    cap.release()
    out.release()

    # (선택) 원본 삭제 후, 임시파일 → 원본이름
    os.remove(input_path)             # 원본 삭제
    os.rename(temp_path, input_path)  # 임시파일을 기존 이름으로 변경

    print(f"[INFO] Rotation done. Overwrote {input_path} with rotated result.")