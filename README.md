# Pose-Coach

**Pose-Coach는 사용자의 운동 자세를 분석하고 피드백을 제공하는 AI 기반 자세 교정 시스템입니다.** 본 시스템은 딥러닝 기반 Human Pose Estimation 모델(ViTPose)을 활용하여, 사용자의 운동 정확성, 반복 횟수, 속도 등을 평가하고 직관적인 시각 피드백을 제공합니다.

특히, 기존의 포즈 분석 시스템들이 단순한 정확도 측정에 그친 반면, Pose-Coach는 다음과 같은 요소들을 통합적으로 제공합니다:

- **정확도 평가** (Keypoint, Joint Angle RMSE)
- **운동 반복 횟수 계산** (Repetition Count)
- **운동 속도 평가** (Speed Score)
- **틀린 자세 시각화 피드백** (Visual Feedback of Incorrect Pose)

운동 종류는 `Diagonal Arm Lift`를 기준으로 구현되었으며, 구조적으로는 타 운동으로의 확장이 가능합니다.

---

## Paper
AI-Based Pose Coach: Effectiveness Assessment and Feedback Through Diagonal Arm Lift Movements, submitted to [Conference Name]

---

## Index

- [Project Introduction](#project-introduction)
- [Development Environment](#development-environment)
- [Key Features](#key-features)
- [Requirements](#requirements)
- [Project Structure](#project-structure)
- [Contributor](#contributor)
- [Demo Video](#demo-video)

---

## Project Introduction

건강과 피트니스에 대한 관심이 높아지면서 전문적인 자세 교정과 코칭에 대한 수요도 함께 증가하고 있습니다. 그러나 많은 사람들은 개인 트레이너나 맞춤형 코칭 서비스에 시간적·공간적·경제적 제약으로 인해 쉽게 접근하기 어렵습니다. 이에 따라 Pose-Coach는 누구나 언제 어디서나 사용할 수 있는 AI 기반 자세 교정 시스템으로 설계되었습니다.

본 시스템은 사전학습된 Human Pose Estimation 모델인 ViTPose를 활용하여 사용자의 운동 자세를 정밀하게 분석하고, 이를 기반으로 종합적인 평가 및 피드백을 제공합니다. 초기에는 Diagonal Arm Lift 동작을 기준으로 구현되었지만, 소량의 라벨링된 영상만으로도 다른 운동으로 손쉽게 확장 가능하도록 구조화되어 있습니다.

또한, 다양한 체형이나 촬영 거리에서도 평가의 일관성을 유지하기 위해 활동 기반 바운딩 박스 정규화(Activity-based Bounding Box Normalization) 기법을 도입하였으며, 단순한 keypoint 추출을 넘어 자세 정확도, 반복 횟수, 운동 속도 등을 정량적으로 분석합니다. 아울러, 틀린 자세의 관절 위치와 각도를 시각적으로 피드백하여 사용자가 쉽게 개선점을 인식하고 교정할 수 있도록 도와줍니다.

20명을 대상으로 한 사용자 테스트 결과, 본 시스템은 실제로 자세 정확도 향상, 반복 수행 증가, 속도 조절 개선에 효과적인 것으로 나타났습니다. 이를 통해 Pose-Coach는 홈트레이닝, 재활 운동, 피트니스 교육 등 다양한 실생활 환경에서 유용한 스마트 코칭 솔루션으로 활용될 수 있습니다.

```text
[User Video]
   ↓
[Pose Estimation using ViTPose]
   ↓
[Activity-based BBox Normalization]
   ↓
[Keypoint & Angle Extraction]
   ↓
[Pose Matching with Labeled Video]
   ↓
[Evaluation: Accuracy, Speed, Repetition]
   ↓
[Feedback: Visualization + Scores]
```

## Key Features

| No. | Feature | Description |
|:---:|:--------|:------------|
| 1 | Human Pose Estimation | ViTPose 모델을 사용하여 2D keypoints를 정확하게 추출합니다. |
| 2 | Accuracy Evaluation | 사용자 자세를 분석하고 정확도를 점수화하여 사용자에게 제공합니다. |
| 3 | Repetition Counting | 운동 횟수를 자동으로 카운트합니다. |
| 4 | Speed Evaluation | 사용자의 운동 속도 조절을 도와줍니다. |
| 5 | Visualizaton | 사용자의 운동 자세와 모범 운동 자세를 비교하여 시각화합니다. 틀린 관절의 각도와 위치를 시각화하여 피드백합니다. |

## Key Features

| No. | Feature | Description |
|:---:|:--------|:------------|
| 1 | **High-Precision Pose Estimation** | 사전학습된 ViTPose 모델을 사용하여 사용자 영상에서 17개의 2D keypoints를 정밀하게 추출합니다. 이는 이후 모든 분석의 기반이 됩니다. |
| 2 | **Activity-based Bounding Box Normalization** | 사용자마다 체형, 카메라 거리, 프레임 내 위치가 다르므로 전체 keypoint 분포를 기반으로 [0,1] 범위로 정규화하여 일관된 자세 분석을 가능하게 합니다.<br>`x' = (x - xmin) / (xmax - xmin), y' = (y - ymin) / (ymax - ymin)` |
| 3 | **Joint Angle Computation (Cosine Rule)** | 자세 평가를 위해 관절 각도를 코사인 법칙을 통해 계산합니다.<br>`θ = cos⁻¹((A−B)·(C−B) / (‖A−B‖‖C−B‖))`<br>이는 어깨, 팔꿈치, 무릎 등 주요 부위의 움직임을 수치화하는 데 사용됩니다. |
| 4 | **Repetition Counting via Angle Transitions** | 팔 관절의 각도가 상향 → 하향 또는 그 반대로 전환되는 지점을 포착하여 반복 횟수를 계산합니다. <br>임계각도(`θ_up`, `θ_down`)와 최소 간격 조건(`Δt`)을 통해 중복 카운트를 방지합니다. |
| 5 | **Speed Analysis by Relative Timing** | 기준 영상의 반복 속도(`T_label`)와 사용자 반복 시간(`T_user`)을 비교하여 속도 점수를 부여합니다. <br>비율 `r = T_label / T_user`에 따라 속도가 너무 느리거나 빠르면 감점됩니다.<br>`S(r) = { 2r (r≤0.5), 1 (0.5<r≤1.0), 2(1.5−r) (1.0<r≤1.5), 0 (r>1.5) }` |
| 6 | **Accuracy Evaluation (RMSE of Keypoint & Angle)** | 라벨 영상과 사용자 프레임 간의 keypoint 및 angle의 Root Mean Square Error(RMSE)를 계산해 정확도를 정량화합니다.<br>`Accuracy = max(0, 1 - RMSE_avg)`,<br>여기서 `RMSE_avg = (RMSE_keypoint + RMSE_angle)/2` |
| 7 | **Visual Feedback with Skeleton Overlay** | 오차가 큰 프레임에서 사용자 skeleton과 라벨 skeleton을 동시에 시각화하고, 오차가 큰 관절을 강조하여 **직관적인 피드백**을 제공합니다. |

---

## Requirements

프로젝트를 실행하기 위해 다음 Python 라이브러리가 필요합니다:
```bash
pip install -r requirements.txt
```

ViTPose 라이브러리 설치
```bash
# mmcv 설치
git clone https://github.com/open-mmlab/mmcv.git
cd mmcv
git checkout v1.3.9
MMCV_WITH_OPS=1 pip install -e .
cd ..

# ViTPose 설치
git clone https://github.com/ViTAE-Transformer/ViTPose.git
cd ViTPose
pip install -v -e .
```

Pretrained Model 다운로드

https://github.com/ViTAE-Transformer/ViTPose

---

## Project Structure
