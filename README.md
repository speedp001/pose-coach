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

## Project Structure

<img width="3259" height="2200" alt="Fig 1" src="https://github.com/user-attachments/assets/8a8d35e9-2805-4a50-817e-ed0165e2d29f" />

---

## Key Features

Pose-Coach는 단순한 keypoint 추출에 그치지 않고, 정확한 자세 평가와 반복 횟수 측정, 운동 속도 분석, 시각적 피드백까지 통합한 AI 기반 자세 분석 시스템입니다. 특히 기존 시스템과 달리 동작의 ‘정확성’, ‘일관성’, ‘속도’, ‘반복성’까지 포괄적으로 평가할 수 있으며, 영상 기반 자동 비교 알고리즘을 통해 개별 사용자 맞춤 피드백을 제공합니다.

---

### 1. High-Precision Pose Estimation  
운동 자세 분석의 첫 단계는 **High-Precision Pose Estimation**입니다. 본 시스템은 사전학습된 ViTPose 모델을 사용하여 17개 관절의 2D keypoints를 프레임 단위로 정확히 추출합니다. ViTPose는 HRNet 대비 더 높은 해상도 처리 능력과 범용성으로, 다양한 촬영 조건에서도 일관된 keypoint 품질을 보장합니다.

---

### 2. Activity-based Bounding Box Normalization  
사람마다 체형이 다르고 촬영 거리나 위치도 다양하기 때문에, 동일한 동작이라도 raw keypoint 값에는 큰 차이가 발생할 수 있습니다. 이를 해결하기 위해 본 시스템은 **Activity-based Bounding Box Normalization** 기법을 적용합니다.

기존의 프레임별 bounding box와 달리, 한 세션 전체 프레임에서 관측된 keypoint들의 최소·최대 좌표 범위를 기준으로 정규화하여 운동 전후 위치 변화나 개개인 별 체형 차이에 강건합니다.

```text
x' = (x − xmin) / (xmax − xmin)  
y' = (y − ymin) / (ymax − ymin)
```

이러한 정규화는 keypoint 기반 비교의 기준을 통일하고, 모든 후속 분석에서 체형 편차의 영향을 최소화합니다.

---

### 3. Joint Angle Computation (Cosine Rule)  
정규화된 keypoint는 단순한 위치 정보에 그치지 않고, 관절 각도 측정을 통해 자세의 질을 정량적으로 평가하는 데 사용됩니다.  

어깨–팔꿈치–손목, 엉덩이–무릎–발목 등 세 점을 활용해 관절의 내각을 계산하며, 이를 위해 **코사인 법칙(Cosine Rule)**을 적용합니다.

```text
θ = cos⁻¹ ( ((A−B)·(C−B)) / (‖A−B‖‖C−B‖) )
```

기존 시스템은 keypoint의 좌표 차이만으로 평가했으나, 본 시스템은 관절 회전각 자체를 정량화함으로써 운동의 "형태 정확도"를 더 세밀하게 측정합니다.

---

### 4. Angle-based Keyframe Matching  
운동의 속도나 프레임 수가 사람마다 다르기 때문에 시간 기준 정렬은 오차를 유발합니다. 이를 해결하기 위해 본 시스템은 **Angle-based Keyframe Matching** 기법을 적용합니다.

어깨 관절의 각도가 30°, 60°, 90°, 120° 등 특정 기준 각도에 도달하는 시점을 기준으로 사용자의 프레임과 라벨 프레임을 동기화합니다.  

이 방식은 서로 다른 속도나 반복 수에서도 정확한 동작 구간 비교가 가능합니다.

---

### 5. Accuracy Evaluation (RMSE-based)  
자세의 정확도는 keypoint 위치 및 joint angle 오차를 기반으로 계산합니다. 동기화된 프레임쌍마다 RMSE를 계산하고 이를 평균내어 정확도를 정량화합니다.

```text
RMSE_avg = (RMSE_keypoint + RMSE_angle) / 2  
Accuracy = max(0, 1 - RMSE_avg)
```

정확도는 0~1 범위로 제공되며, 1에 가까울수록 정답 영상과의 유사도가 높음을 의미합니다.

---

### 6. Repetition Counting (Angle Transition Detection)  
반복 횟수는 **관절 각도의 상승→하강 전이**를 기반으로 계산됩니다. 기존의 좌표 y값 상승/하강 기준 방식은 팔 각도가 굽혀지거나 회전이 있으면 오탐 가능성이 컸습니다.

Pose-Coach는 팔꿈치 또는 어깨 각도의 상향(≥120°)과 하향(≤60°)을 기준으로 전이 구간을 탐지하며, 최소 시간 간격을 조건으로 중복 카운트를 방지합니다.

이 방식은 **움직임 의미 기반 반복 분석**을 가능하게 하여 더 정밀한 반복 측정이 가능합니다.

---

### 7. Speed Evaluation (Relative Tempo Score)  
운동 속도는 사용자의 반복 주기(`T_user`)와 기준 영상(`T_label`) 간의 시간 비율 `r`로 계산되며, 점수는 다음과 같은 piecewise 함수로 계산됩니다:

```text
S(r) = 
  2r              , r ≤ 0.5  
  1               , 0.5 < r ≤ 1.0  
  2(1.5 − r)      , 1.0 < r ≤ 1.5  
  0               , r > 1.5  
where r = T_label / T_user
```

이 방식은 단순한 빠름/느림 판단을 넘어서 **운동 속도**를 점수화하며, 트레이닝이나 재활 시 적정 속도 유지 유도에 유용합니다.

---

### 8. Visual Feedback (Skeleton Overlay with Error Highlighting)  
사용자가 정확하지 않은 자세를 수행했을 때 이를 쉽게 인지할 수 있도록, Pose-Coach는 **Skeleton Overlay 시각화**를 제공합니다.

사용자 skeleton과 라벨 skeleton을 동일 위치에 나란히 표시하고, 오차가 큰 관절은 색상이나 크기로 강조하여 **직관적인 오류 인식**이 가능하도록 설계되었습니다.

<img width="561" height="596" alt="Fig 2 (a)" src="https://github.com/user-attachments/assets/affe598a-e1e0-49c5-bfde-fb862907bfc2" /> | 
![Fig 2 (b)](https://github.com/user-attachments/assets/e384e10f-94c1-45b6-9ec3-2ee62a02709d) | 
<img width="561" height="596" alt="Fig 2 (c)" src="https://github.com/user-attachments/assets/6eef8c0e-139f-4620-b6ab-41ed87031d03" />


이 방식은 전문 지식이 없는 사용자도 오류를 빠르게 인지하고 개선할 수 있게 하며, 실제 사용자 테스트에서도 **정확도 향상과 반복 수행 증가에 긍정적인 영향을 실제 User Test**를 통해 입증하였습니다.

<img width="995" height="981" alt="Fig 4 (a)" src="https://github.com/user-attachments/assets/a4a28a2e-418d-49c3-85aa-bb2c02344014" /> | 
<img width="996" height="981" alt="Fig 4 (b)" src="https://github.com/user-attachments/assets/aef9518b-0859-4057-b885-00543a8a71cc" /> | 
<img width="995" height="981" alt="Fig 4 (c)" src="https://github.com/user-attachments/assets/47fcce7d-a9a1-4b32-83fd-5ca1227cfcfa" />

---

## Requirements

프로젝트를 실행하기 위한 필수 라이브러리
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

ViTPose GitHub 주소 및 Pretrained Model 다운 링크

https://github.com/ViTAE-Transformer/ViTPose

---

## Demo Video

## Demo Video

> YouTube Link  
https://youtu.be/Uw1KAmsxoRo
---
