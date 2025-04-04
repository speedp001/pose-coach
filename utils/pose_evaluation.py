import os
import cv2
import json
import numpy as np

from collections import defaultdict





##### JSON 파일 로드 함수 정의
def load_json(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    return data





##### 이벤트 추출 함수 정의
def find_angle_events(frames, ang_index, ang_targets, tol, is_label):
    """
    frames   : label_data["frames"] or user_data["frames"]
    ang_index: angles 배열에서 사용할 인덱스 (예: 왼=5, 오른=6)
    ang_targets: [30, 60, 90, 120] 등 목표 각도 리스트
    tol      : ±오차 범위 (기본값 5도)
    is_label : True면 (각도, 방향)별 '최대 1회'만, False면 해당 각도 이벤트 전부

    return: [ { "frame_idx", "angle_val", "target", "direction" }, ...]
    """

    events = []
    prev_angle = None

    for i, fr in enumerate(frames):
        angles = fr["angles"]
        if ang_index >= len(angles):
            continue

        cur_angle = angles[ang_index]
        
        # 방향(Up/Down) 결정
        if prev_angle is None:
            direction = 0
        else:
            # 각도가 증가하고 있으면 +1, 감소하고 있으면 -1
            direction = +1 if (cur_angle > prev_angle) else -1

        # 목표 각도( ± tolerance) 범위에 해당하면 이벤트 추가
        for tgt in ang_targets:
            
            # 목표 각도와 현재 각도의 차이가 5도 미만 -> 이벤트 추가
            if abs(cur_angle - tgt) <= tol:
                events.append({
                    "frame_idx": i,
                    "angle_val": cur_angle,
                    "target": tgt,
                    "direction": direction
                })
                # 여기 break는 한 프레임에 대해서 30, 60, 90, 120 중 하나로 매칭되면 더 이상 볼 필요 없다는 것을 의미
                # 30도이면 60도 90도 120도일 수 없다는 것을 의미
                break

        prev_angle = cur_angle

    # 사용자용: 각 각도마다 여러개의 이벤트가 있을 수 있으므로 그대로 반환
    if not is_label:

        return events

    # 라벨용: (각도, 방향)별로 최대 1회만 이벤트를 채택
    found_map = {}  # (target, direction) → bool
    event = []
    
    for ev in events:
        tgt  = ev["target"]
        dire = ev["direction"]
        
        if dire == 0:
            continue

        # 아직 (tgt,dire)에 대해 등록된 적 없으면 채택
        if (tgt, dire) not in found_map:
            found_map[(tgt, dire)] = True
            event.append(ev)
        # 이미 있으면 무시

    return event





##### 매칭 함수 정의
def match_label_user_events(label_event, user_event):
    """
    (target, direction)이 같은 이벤트끼리 1:다 매칭
    return: [(label_event, user_event), ...]
    """
    
    matched_pairs = []
    for le in label_event:
        # direction=0은 연산 생략
        if le["direction"] == 0:
            continue
        for ue in user_event:
            # direction=0은 연산 생략
            if ue["direction"] == 0:
                continue
            if (ue["target"] == le["target"]) and (ue["direction"] == le["direction"]):
                matched_pairs.append((le, ue))
                
    return matched_pairs





##### 오차 계산 함수 정의
def compute_errors(label_json, user_json, matched_pairs):
    """
    matched_pairs로 frame 간 정확도 계산
         - kp_mean_diff : 키포인트 평균 오차
         - ang_mean_diff: 관절 각도 평균 오차
    """

    label_data = load_json(label_json)
    user_data  = load_json(user_json)
    label_frames = label_data["frames"]
    user_frames  = user_data["frames"]

    results = []
    for (le, ue) in matched_pairs:
        lf = le["frame_idx"]
        uf = ue["frame_idx"]
        
        # 유효 범위 체크
        if lf < 0 or lf >= len(label_frames):
            continue
        if uf < 0 or uf >= len(user_frames):
            continue

        # 키포인트/각도 로드
        label_kp = np.array(label_frames[lf]["keypoints"])
        user_kp  = np.array(user_frames[uf]["keypoints"])
        label_ang= np.array(label_frames[lf]["angles"])
        user_ang = np.array(user_frames[uf]["angles"])

        # 키포인트 평균 오차(각 프레임 기준)
        kp_diff = np.linalg.norm(label_kp - user_kp, axis=1)  # 각 keypoint별 거리
        kp_mean_diff = np.mean(kp_diff)

        # 관절 각도 평균 오차(각 프레임 기준))
        ang_diff = np.abs(label_ang - user_ang)
        ang_mean_diff = np.mean(ang_diff)

        results.append({
            "label_frame": lf,
            "label_kp": label_kp,
            "label_ang": label_ang,
            "user_frame": uf,
            "user_kp": user_kp,
            "user_ang": user_ang,
            "target": le["target"],
            "direction": le["direction"],
            "kp_mean_diff": kp_mean_diff,
            "ang_mean_diff": ang_mean_diff
        })
        
    return results




def rmse(result):
    """
    RMSE 계산
    return: keypoints RMSE, angles RMSE, final score
    """
    
    kp_mse_list  = []
    ang_mse_list = []

    for r in result:
        # keypoint MSE
        kp_mse = np.mean((r["label_kp"] - r["user_kp"])**2)
        # keypoint RMSE
        kp_rmse= np.sqrt(kp_mse)

        # angle MSE
        ang_mse = np.mean((r["label_ang"] - r["user_ang"])**2)
        # angle RMSE
        ang_rmse= np.sqrt(ang_mse)

        # 프레임 별 RMSE 리스트에 저장
        kp_mse_list.append(kp_rmse)
        ang_mse_list.append(ang_rmse)

        # 프레임 별 RMSE 출력
        # print(f"[Frame {r['label_frame']}] RMSE_kp={kp_rmse:.2f}, RMSE_ang={ang_rmse:.2f}")

    # 전체 평균 RMSE
    if len(result) > 0:
        avg_kp_rmse  = np.mean(kp_mse_list)
        avg_ang_rmse = np.mean(ang_mse_list)
        
    else:
        avg_kp_rmse  = 0.0
        avg_ang_rmse = 0.0

    # 최종 정확도 점수 계산
    acc_score = 100 - ( (avg_kp_rmse + avg_ang_rmse) / 2 )  
    # print(f"\n[RMSE] keypoints={avg_kp_rmse:.2f}, angles={avg_ang_rmse:.2f}")  
    # print(f"Final Score: {final_score:.2f}")

    return avg_kp_rmse, avg_ang_rmse, acc_score


##### worst frame 함수 정의
def group_and_find_worst(results, top_n=1):
    """
    (target, direction) 그룹으로 묶어 measure=(kp+ang) 내림차순 → 상위 top_n
    return: dict[ (target, direction) ] = [worst...]
    """
    
    groups = defaultdict(list)
    for r in results:
        measure = r["kp_mean_diff"] + r["ang_mean_diff"]
        r["measure"] = measure
        key = (r["target"], r["direction"])
        groups[key].append(r)

    worst_dict = {}
    for k, arr in groups.items():
        arr_sorted = sorted(arr, key=lambda x: x["measure"], reverse=True)
        # worst_dict은 (target, direction)의 키로 이루어져 있고 각 키에는 top_n개의 원소가 들어있음
        # 각 키의 요소에는 [{worst1}, {worst2}, ...] 형태
        """
        worst는 {"label_frame": lf,
            "label_kp": label_kp,
            "label_ang": label_ang,
            "user_frame": uf,
            "user_kp": user_kp,
            "user_ang": user_ang,
            "target": le["target"],
            "direction": le["direction"],
            "kp_mean_diff": kp_mean_diff,
            "ang_mean_diff": ang_mean_diff
        })
        """
        worst_dict[k] = arr_sorted[:top_n]
        
    return worst_dict





##### 시각화 함수 정의
def visualize_mismatch_threshold(l_data, u_data,
                                 user_video, label_video,
                                 label_idx,
                                 user_idx):
    """
    2*2 레이아웃:
      (왼위)  라벨 원본 프레임
      (오른위)사용자 원본 프레임
      (왼아래)검정 배경 + 라벨 스켈레톤(초록)
      (오른아래)검정 배경 + 사용자 스켈레톤(파랑) + worst 관절 전체 빨간선 + 중심점(주황)
    """

    lx_min, ly_min, lx_max, ly_max = l_data["bbox"]
    ux_min, uy_min, ux_max, uy_max = u_data ["bbox"]

    label_frames = l_data["frames"]
    user_frames  = u_data["frames"]

    # 해당 프레임 키포인트/각도
    label_kp = np.array(label_frames[label_idx]["keypoints"])
    user_kp  = np.array(user_frames[user_idx]["keypoints"])
    label_ang= np.array(label_frames[label_idx]["angles"])
    user_ang = np.array(user_frames[user_idx]["angles"])

    # 비디오 프레임 로드
    cap_label = cv2.VideoCapture(label_video)
    cap_user  = cv2.VideoCapture(user_video)
    
    cap_label.set(cv2.CAP_PROP_POS_FRAMES, label_idx)
    cap_user.set(cv2.CAP_PROP_POS_FRAMES,  user_idx)

    ret_l, frame_label = cap_label.read()
    ret_u, frame_user  = cap_user.read()
    if not (ret_l and ret_u):
        print("Fail to read frames")
        return

    # label 이미지 프레임에서 bbox 좌표만 crop
    lh, lw = frame_label.shape[:2]
    x1l = max(0, int(lx_min)); x2l = min(lw, int(lx_max))
    y1l = max(0, int(ly_min)); y2l = min(lh, int(ly_max))
    label_frame_original = frame_label[y1l:y2l, x1l:x2l].copy()
    # label crop 이미지 모양 확인
    # print(label_frame_original.shape)


    uh, uw = frame_user.shape[:2]
    x1u = max(0, int(ux_min)); x2u = min(uw, int(ux_max))
    y1u = max(0, int(uy_min)); y2u = min(uh, int(uy_max))
    user_frame_original = frame_user[y1u:y2u, x1u:x2u].copy()
    user_frame_original = cv2.resize(user_frame_original, (x2l-x1l, y2l-y1l))
    # user crop 이미지 모양 확인
    # print(user_frame_original.shape)

    # (왼아래) 라벨 스켈레톤(검정 배경)
    label_skel_only = np.zeros((y2l-y1l, x2l-x1l, 3), dtype=np.uint8)
    # (왼아래) 스켈레톤 crop 이미지 모양 확인
    # print((label_skel_only.shape))
    
    # (오른아래) 사용자 스켈레톤(검정 배경)
    user_skel_only  = np.zeros((y2l-y1l, x2l-x1l, 3), dtype=np.uint8)
    # (오른아래) 스켈레톤 crop 이미지 모양 확인
    # print((user_skel_only.shape))
    
    # 각도 차이 계산
    # 각도 배열에서 “worst angle” 찾기
    ang_diffs = np.abs(user_ang - label_ang)
    # # 관절 인덱스
    worst_ang_idx = np.argmax(ang_diffs)
    
    # ---------- Worst Angle ----------
    # # worst 관절 인덱스의 값
    # worst_ang_val = ang_diffs[worst_ang_idx]
    
    # ---------- Wrong Angle ----------
    angles_over_20 = np.where(ang_diffs > 20)[0]  # 20을 넘어서는 각도 인덱스 전부
    
    # 키포인트 차이 계산
    # 각 키포인트별 거리
    kp_diffs = np.linalg.norm(user_kp - label_kp, axis=1)  
    # 최악 키포인트 인덱스
    worst_kp_idx = int(np.argmax(kp_diffs))  
    # 최악 키포인트 거리             
    worst_kp_val = kp_diffs[worst_kp_idx]                 

    # angle 관절 인덱스 선언
    #  예: 0=왼팔(5,7,9), 1=오른팔(6,8,10), 3=왼다리(11,13,15), 4=오른다리(12,14,16)
    angle_joint_edges = {
        0: [(5,7), (7,9)],   # 왼팔
        1: [(6,8), (8,10)],  # 오른팔
        3: [(11,13),(13,15)],# 왼다리
        4: [(12,14),(14,16)] # 오른다리
    }
    
    # angle 관절 인덱스의 중심 인덱스 선언
    # 팔꿈치, 무릎
    angle_center_map = {
        0: 7,
        1: 8,
        3: 13,
        4: 14
    }

    # 전체 스켈레톤 연결
    skeleton_connections = [
        (0,5), (0,6),   # 코 어깨
        (5,6),          # 양 어깨
        (5,7),(7,9),    # 왼팔
        (6,8),(8,10),   # 오른팔
        (11,12),        # 양 골반
        (5,11),(6,12),  # 어깨→골반
        (11,13),(13,15),# 왼다리
        (12,14),(14,16) # 오른다리
    ]

    # 만약 worst_ang_idx가 범위 밖이면 왼팔(0)로 처리
    # dict.get(key, default) -> 예상 범위 밖 default로 설정
    highlight_edges = angle_joint_edges.get(worst_ang_idx, [(5,7),(7,9)])
    highlight_center = angle_center_map.get(worst_ang_idx, 7)

    # 라벨 스켈레톤 그리기 (초록)
    def draw_label_skeleton(img, kp):
        for (p1,p2) in skeleton_connections:
            pt1= tuple(kp[p1].astype(int))
            pt2= tuple(kp[p2].astype(int))
            cv2.line(img, pt1, pt2, (0,255,0),2)
            cv2.circle(img, pt1,4, (0,255,0),-1)
            cv2.circle(img, pt2,4, (0,255,0),-1)

    # 사용자 스켈레톤 그리기 (파랑) + worst 관절(빨강) + 중심점(오렌지)
    def draw_user_skeleton(img, kp):
        # 먼저 기본 골격(파랑)
        for (p1,p2) in skeleton_connections:
            pt1= tuple(kp[p1].astype(int))
            pt2= tuple(kp[p2].astype(int))
            cv2.line(img, pt1, pt2, (255,0,0),2)
            cv2.circle(img, pt1,4, (255,0,0),-1)
            cv2.circle(img, pt2,4, (255,0,0),-1)

        # ---------- Worst Angle만 출력하기 ----------
        # # worst 관절(빨강)으로 다시 그리기
        # for (p1,p2) in highlight_edges:
        #     pt1= tuple(kp[p1].astype(int))
        #     pt2= tuple(kp[p2].astype(int))
        #     cv2.line(img, pt1, pt2, (0,0,255),3)  # 좀 더 두껍게
        #     cv2.circle(img, pt1,5, (0,0,255),-1)
        #     cv2.circle(img, pt2,5, (0,0,255),-1)
            
        
        # # worst 관절 중심 키포인트(오렌지 원 + 텍스트)
        # pt_center = tuple(kp[highlight_center].astype(int))
        # cv2.circle(img, pt_center, 9, (0,165,255), -1)  # 오렌지
        # cv2.putText(img,
        #             "Angle Worst",
        #             (pt_center[0]+5, pt_center[1]-5),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.6,
        #             (0,165,255),2)

        # ---------- 오차 20도 이상 시각화하기 ----------
        # angles_over_20를 순회하며 빨강으로 표시
        for idx in angles_over_20:
            # 1) angle_joint_edges / angle_center_map 에서 해당 idx 관절의 연결부와 중심점 얻기
            highlight_edges = angle_joint_edges.get(idx, [])
            highlight_center= angle_center_map.get(idx, None)

            # 관절 연결부 (빨강)
            for (p1,p2) in highlight_edges:
                pt1= tuple(kp[p1].astype(int))
                pt2= tuple(kp[p2].astype(int))
                cv2.line(img, pt1, pt2, (0,0,255), 3)   # 더 두껍게 빨강
                cv2.circle(img, pt1,5, (0,0,255),-1)
                cv2.circle(img, pt2,5, (0,0,255),-1)

            # 중심점 (오렌지)
            if highlight_center is not None:
                pt_center = tuple(kp[highlight_center].astype(int))
                cv2.circle(img, pt_center, 9, (0,165,255), -1)  # 오렌지
                cv2.putText(img,
                            f"Wrong Angle",
                            (pt_center[0]+5, pt_center[1]-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0,165,255),2)
        
        
        # worst keypoint(분홍색 원 + 텍스트)
        pt_worst_kp = kp[worst_kp_idx].astype(int)
        cv2.circle(img, tuple(pt_worst_kp), 9, (255,0,255), -1)  # BGR=(255,0,255) → 분홍
        cv2.putText(img,
                    "Wrong Position",
                    (pt_worst_kp[0]+5, pt_worst_kp[1]-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255,0,255), 2)

    # (왼아래) 라벨(검정)
    draw_label_skeleton(label_skel_only, label_kp)
    # (오른아래) 사용자(검정) + worst 관절
    draw_user_skeleton(user_skel_only, user_kp)

    # 2*2 합치기
    top    = np.hstack((label_frame_original, user_frame_original))
    bottom = np.hstack((label_skel_only,      user_skel_only))
    final_2x2 = np.vstack((top, bottom))

    # 2*2 시각화
    cv2.imshow("Comparison (2x2) - Angle-based w/ Full Red Joint", final_2x2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cap_label.release()
    cap_user.release()





##### 사용자 포즈 평가 함수 정의
def pose_evaluation(user_json, label_json, user_video, label_video):
    """
    -user_json, label_json: 사용자 및 라벨 JSON 파일 경로
    -user_video, label_video: 사용자 및 라벨 비디오 파일 경로
    - side에 따라 angle_index 결정 (왼=6, 오른=5)  (원하는대로 수정 가능)
    - 라벨/사용자 JSON에서 각도 이벤트 추출 → 매칭 → 매칭 쌍별 오차 계산 → (각도,방향) 그룹별 worst top_n → 시각화
    """
    
    # 사용자 포즈 추정 동영상 재생
    cap_user = cv2.VideoCapture(user_video)
    if not cap_user.isOpened():
        print(f"[ERROR] Unable to open user video: {user_video}")
        return

    while True:
        ret, frame = cap_user.read()
        if not ret:
            print("[INFO] The user video has finished playing.")
            break
        cv2.imshow("UserVideo Playback", frame)

        # waitKey(1)로 영상 업데이트
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            print("[INFO] User pressed ESC. Stop playing.")
            break

    cap_user.release()
    cv2.destroyAllWindows()
    
    # left/right에 따라 JSON파일의 angle index 결정
    filename = os.path.basename(label_json)
    if "left" in filename:
        angle_index = 5
    else:
        angle_index = 6

    # 각도 목표값, 허용 오차
    angle_targets = [30, 60, 90, 120]
    tolerance = 5.0

    # 라벨 JSON 로드 & 이벤트 추출
    label_data= load_json(label_json)
    label_frames= label_data["frames"]
    label_events= find_angle_events(label_frames, angle_index, angle_targets, tolerance, is_label=True)

    # 사용자 JSON 로드 & 이벤트 추출
    user_data= load_json(user_json)
    user_frames= user_data["frames"]
    user_events= find_angle_events(user_frames, angle_index, angle_targets, tolerance, is_label=False)
    
    # 이벤트 매칭
    matched_pairs= match_label_user_events(label_events, user_events)
    # print(f"Number of matched pairs: {len(matched_pairs)}")

    # 매칭 오차 계산
    results= compute_errors(label_json, user_json, matched_pairs)

    # MSE/RMSE 계산
    kp_rmse, ang_rmse, accuracy_score= rmse(results)
    
    # 각도, 방향 별 worst top_n
    worst_dict= group_and_find_worst(results, top_n=1)

    # 사용자 피드백 시각화
    for (tgt, direc), arr in worst_dict.items():
        dir_str= "UP" if direc>0 else "DOWN"
        # print(f"\n--- Target={tgt}°, Direction={dir_str} ---")
        
        for rank, w in enumerate(arr, start=1):
            # print(f"[Rank {rank}] userF={w['user_frame']}, labelF={w['label_frame']}, "
            #       f"kp={w['kp_mean_diff']:.2f}, ang={w['ang_mean_diff']:.2f}, measure={w['measure']:.2f}")
            
            visualize_mismatch_threshold(
                label_data, user_data,
                user_video, label_video,
                w["label_frame"],
                w["user_frame"]
            )
            
    return kp_rmse, ang_rmse, accuracy_score