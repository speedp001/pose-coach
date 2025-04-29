# Pose-Coach

> **Pose-Coach는 사용자의 운동 자세를 분석하여 실시간 피드백을 제공하는 AI 기반 포즈 교정 코칭 프로그램입니다. 정확한 포즈 수행을 지원하고, 운동 효과를 극대화하는 것을 목표로 합니다.**

---

## Index
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
  - [ViTPose 모델](#vitpose-모델)
- [Contributor](#contributor)
- [Demo Video](#demo-video)

---

## Project Introduction

✏️ **(작성 공간)**  
Pose-Coach는 Human Pose Estimation 기술을 활용하여 사용자의 운동 자세를 분석하고 실시간으로 피드백을 제공하는 시스템입니다.  
운동 효과를 극대화하고, 잘못된 자세를 교정할 수 있도록 돕습니다.

---

## Development Environment

- Python 3.8+
- PyTorch 1.9+
- mmcv-full
- OpenCV
- Streamlit
- ViTPose Pretrained Models

---

## Key Features

| No. | Feature | Description |
|:---:|:--------|:------------|
| 1 | Human Pose Estimation | ViTPose 모델을 사용하여 2D keypoints를 정확하게 추출합니다. |
| 2 | Real-time Feedback | 실시간으로 사용자 자세를 분석하고 피드백을 제공합니다. |
| 3 | Repetition Counting | 반복 운동 횟수를 자동으로 카운트합니다. |
| 4 | Dynamic Time Warping | 다른 운동 속도를 고려한 유사성 보정을 적용합니다. |
| 5 | Personalized Analysis | 신체 조건에 맞춘 맞춤형 자세 분석 기능을 제공합니다. |

---

## Requirements

프로젝트를 실행하기 위해 다음 Python 라이브러리가 필요합니다:

```bash
pip install torch torchvision torchaudio
pip install mmcv-full
pip install opencv-python
pip install streamlit
