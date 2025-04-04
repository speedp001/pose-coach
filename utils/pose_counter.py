import os
import json
import numpy as np





##### JSON 로드 함수 정의
def load_json(json_path):
    with open(json_path, "r") as f:
        return json.load(f)





##### 운동 횟수 측정 함수 정의
def count_reps(frames, angle_index, threshold, min_gap):
    """
    - frames: JSON의 "frames" 리스트
    - angle_index: frames[i]["angles"][angle_index] 로 가져올 각도 인덱스
    - threshold: 각도 임계치 (기본값: 예 120도)
    - min_gap: 중복 카운팅 방지
    return:
      count(완료된 동작 횟수), [각 동작의 소요 프레임 수...]
    """

    # 히스테리시스 적용:
    # 올라갈 때(Down→Up)는 threshold_up 사용,
    # 내려갈 때(Up→Down)는 threshold_down 사용.
    # ex) threshold=120 이라면,
    #     threshold_up=120, threshold_down=110 등으로 설정.
    threshold_up = threshold
    threshold_down = threshold - 10  # 필요시 다른 값으로 조정 가능

    count = 0
    rep_times = []

    # "down" → "up" → "down" = 1회
    state = "down"
    start_frame_idx = None

    # 최근 카운트 완료 프레임 인덱스
    last_count_frame = -999

    for i, fr in enumerate(frames):
        angles = fr["angles"]
        if angle_index >= len(angles):
            continue
        cur_angle = angles[angle_index]

        # down 상태에서 threshold_up 이상 => up으로 전환
        if state == "down" and (cur_angle >= threshold_up):
            start_frame_idx = i
            state = "up"

        # up 상태에서 threshold_down 미만 => down으로 전환 => 1회 운동
        elif state == "up" and (cur_angle < threshold_down):
            end_frame_idx = i
            state = "down"

            # 최근 카운트와 min_gap 이상 차이나면 유효 카운팅
            if (end_frame_idx - last_count_frame) >= min_gap:
                count += 1
                last_count_frame = end_frame_idx

                # 운동 1회의 걸린 프레임 수
                if start_frame_idx is not None:
                    duration = end_frame_idx - start_frame_idx
                    rep_times.append(duration)

    return count, rep_times





##### 속도 점수 변환 함수 정의
def speed_score_piecewise(ratio):
    """
    0(매우 빠름) -> 50(빠름) -> 100(적정) -> 50(느림) -> 0(매우 느림)
    ratio=0.0   => score=0
    ratio=0.5   => score=50
    ratio=1.0   => score=100
    ratio=1.5   => score=50
    ratio=2.0   => score=0
    ratio>2.0   => score=0
    """
    
    if ratio < 0:
        ratio = 0

    # ratio <= 0.5 → 0~50
    if ratio <= 0.5:
        return 100 * ratio

    # 0.5 < ratio <= 1.0 → 50~100
    elif ratio <= 1.0:
        return 50 + 100 * (ratio - 0.5)

    # 1.0 < ratio <= 1.5 → 100~50
    elif ratio <= 1.5:
        return 100 - 100 * (ratio - 1.0)

    # 1.5 < ratio <= 2.0 → 50~0
    elif ratio <= 2.0:
        return 50 - 100 * (ratio - 1.5)

    # ratio > 2.0 → 0
    else:
        return 0.0





##### 운동 속도 측정 함수 정의
def pose_counter(user_json, label_json, angle_threshold=100.0, min_gap=20):
    """
    사용자 운동 속도와 라벨 운동 속도를 측정하여 비율을 계산
    0(매우 빠름) -> 50(빠름) -> 100(적정) -> 50(느림) -> 0(매우 느림)
    
    return:
      user_count, label_count, user_avg_time, label_avg_time, speed_ratio
    """
    
    # JSON 로드
    user_data  = load_json(user_json)
    label_data = load_json(label_json)

    # left/right에 따라 angle_index 결정
    filename = os.path.basename(label_json)
    
    if "left" in filename:
        angle_index = 5
    else:
        angle_index = 6

    # 사용자 운동 횟수/속도
    user_frames = user_data["frames"]
    user_count, user_rep_times = count_reps(
        user_frames,
        angle_index,
        angle_threshold,
        min_gap
    )
    if len(user_rep_times) > 0:
        user_avg_time = float(np.mean(user_rep_times))
    else:
        user_avg_time = 0.0

    # 라벨 운동 횟수/속도
    label_frames = label_data["frames"]
    label_count, label_rep_times = count_reps(
        frames=label_frames,
        angle_index=angle_index,
        threshold=angle_threshold,
        min_gap=min_gap
    )
    if len(label_rep_times) > 0:
        label_avg_time = float(np.mean(label_rep_times))
    else:
        label_avg_time = 1e-6

    # 속도 비교
    speed_ratio = user_avg_time / label_avg_time

    # 속도 점수(0~100)
    speed_score = speed_score_piecewise(speed_ratio)

    # speed_score도 반환(원한다면)
    return user_count, label_count, speed_score
