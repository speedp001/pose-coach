# Pose-Coach

> **Pose-Coach는 사용자의 운동 자세를 분석하고 피드백을 제공하는 AI 기반 자세 교정 시스템입니다.**
>  
> 본 시스템은 딥러닝 기반 Human Pose Estimation 모델(ViTPose)을 활용하여, 사용자의 운동 정확성, 반복 횟수, 속도 등을 평가하고 직관적인 시각 피드백을 제공합니다.

---

## Paper
AI-Based Pose Coach: Effectiveness Assessment and Feedback Through Diagonal Arm Lift Movements, submitted to [Conference Name]

---

## 📂 Index

- [Project Introduction](#project-introduction)
- [Development Environment](#development-environment)
- [Key Features](#key-features)
- [Requirements](#requirements)
- [Installation Guide](#installation-guide)
  - [1. ViTPose 설치](#1-vitpose-설치)
  - [2. 사전학습 모델 다운로드](#2-사전학습-모델-다운로드)
  - [3. mmcv-full 설치 예시](#3-mmcv-full-설치-예시)
- [Project Structure](#project-structure)
- [Reference Models](#reference-models)
- [Contributor](#contributor)
- [Demo Video](#demo-video)

---

## Project Introduction

Pose-Coach는 사용자가 촬영한 운동 영상을 기반으로 **정확한 자세 평가와 피드백**을 제공하는 시스템입니다.  
특히, 기존의 포즈 분석 시스템들이 단순한 정확도 측정에 그친 반면, Pose-Coach는 다음과 같은 요소들을 통합적으로 제공합니다:

- **정확도 평가** (Keypoint, Joint Angle RMSE)
- **운동 반복 횟수 계산** (Repetition Count)
- **운동 속도 평가** (Speed Score)
- **틀린 자세 시각화 피드백** (Visual Feedback of Incorrect Pose)

운동 종류는 `Diagonal Arm Lift`를 기준으로 구현되었으며, 구조적으로는 타 운동으로의 확장이 가능합니다.

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

---

## Requirements

프로젝트를 실행하기 위해 다음 Python 라이브러리가 필요합니다:

```bash
pip install torch torchvision torchaudio
pip install mmcv-full
pip install opencv-python
pip install streamlit
