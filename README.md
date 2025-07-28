# Pose-Coach

**Pose-Coach is an AI-based posture correction system that analyzes users’ exercise postures and provides feedback.**  
This system leverages the deep learning–based Human Pose Estimation model (ViTPose) to evaluate users’ exercise accuracy, repetition counts, and speed, and it offers intuitive visual feedback.

Unlike many existing pose analysis systems that focus only on measuring accuracy, Pose-Coach integrates the following elements:

- **Accuracy Evaluation** (Keypoint, Joint Angle RMSE)  
- **Repetition Counting**  
- **Speed Evaluation** (Speed Score)  
- **Visual Feedback of Incorrect Pose**

The system is implemented based on the `Diagonal Arm Lift` exercise but is structurally designed to be easily extended to other types of exercises.  
<br></br>

## Paper
AI-Based Pose Coach: Effectiveness Assessment and Feedback Through Diagonal Arm Lift Movements, submitted to [Conference Name]  
<br></br>

## Index

- [Project Introduction](#project-introduction)  
- [Project Structure](#project-structure)  
- [Key Features](#key-features)
- [User Test](#user-test)
- [Requirements](#requirements)  
- [Demo Video](#demo-video)  
<br></br>

## Project Introduction

As public interest in health and fitness grows, the demand for professional posture correction and coaching is also increasing. However, many people face difficulties in accessing personal trainers or custom coaching services due to time, location, or financial constraints. Pose-Coach was designed as an AI-based posture correction system that anyone can use anytime and anywhere.

The system uses a pre-trained Human Pose Estimation model, ViTPose, to accurately analyze users’ exercise postures and provide comprehensive evaluation and feedback. Initially implemented with the Diagonal Arm Lift exercise, the framework is structured so that additional exercises can be integrated with only a small number of labeled videos.

To maintain evaluation consistency across different body shapes and camera distances, an **Activity-based Bounding Box Normalization** technique was introduced. Beyond simple keypoint extraction, Pose-Coach quantitatively analyzes posture accuracy, repetition counts, and exercise speed. Furthermore, it provides visual feedback on incorrect joint positions and angles, helping users easily identify and correct their movements.

In user tests with 20 participants, the system effectively improved posture accuracy, increased repetition counts, and enhanced speed control. Pose-Coach can thus serve as a practical smart coaching solution in home training, rehabilitation exercises, and fitness education.
```
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
<br></br>

## Project Structure
<img width="5324" height="1884" alt="Fig 1" src="https://github.com/user-attachments/assets/b14cef0b-9e90-4665-b095-7657d583810a" />
<br></br>


## Key Features

Pose-Coach is an AI-based posture analysis system that goes beyond simple keypoint extraction to integrate accurate posture evaluation, repetition count measurement, movement speed analysis, and visual feedback. Unlike existing systems, it comprehensively evaluates the "accuracy," "consistency," "speed," and "repetition" of movements, and provides customized feedback for each user through a video-based automatic comparison algorithm.

---

### 1. High-Precision Pose Estimation  
The first step in motion posture analysis is **High-Precision Pose Estimation**. This system uses a pre-trained ViTPose model to accurately extract 2D keypoints of 17 joints on a frame-by-frame basis. ViTPose offers higher resolution processing capabilities and versatility compared to HRNet, ensuring consistent keypoint quality even under various shooting conditions.

---

### 2. Activity-based Bounding Box Normalization  
Since each person has a different body type and shooting distance and position vary, there can be significant differences in raw keypoint values even for the same motion. To solve this problem, this system applies the **Activity-based Bounding Box Normalization** technique.

Unlike traditional frame-based bounding boxes, this technique normalizes the keypoints observed across all frames in a single session based on the minimum and maximum coordinate ranges, making it robust to changes in position before and after movement and individual differences in body type.

```text
x' = (x − xmin) / (xmax − xmin)  
y' = (y − ymin) / (ymax − ymin)
```

This normalization unifies the criteria for keypoint-based comparisons and minimizes the impact of body shape variations in all subsequent analyses.

---

### 3. Joint Angle Computation (Cosine Rule)  
Normalized keypoints are not limited to simple position information, but are also used to quantitatively evaluate posture quality by measuring joint angles.

The cosine rule is applied to calculate the internal angles of joints using three points, such as shoulder–elbow–wrist and hip–knee–ankle.

```text
θ = cos⁻¹ ( ((A−B)·(C−B)) / (‖A−B‖‖C−B‖) )
```

The existing system evaluated only the difference in keypoint coordinates, but this system quantifies the joint rotation angle itself, enabling more detailed measurement of the "form accuracy" of the movement.

---

### 4. Angle-based Keyframe Matching  
Since the speed and frame rate of movement vary from person to person, time-based alignment causes errors. To solve this problem, this system applies the **Angle-based Keyframe Matching** technique.

The user's frames and label frames are synchronized based on the point at which the shoulder joint reaches specific reference angles, such as 30°, 60°, 90°, and 120°.  

This method enables accurate comparison of motion intervals even at different speeds or repetition counts.
---

### 5. Accuracy Evaluation (RMSE-based)  
Posture accuracy is calculated based on keypoint position and joint angle errors. RMSE is calculated for each synchronized frame pair and averaged to quantify accuracy.
```text
RMSE_avg = (RMSE_keypoint + RMSE_angle) / 2  
Accuracy = max(0, 1 - RMSE_avg)
```

Accuracy is provided on a scale of 0 to 1, with values closer to 1 indicating a higher degree of similarity to the correct video.

---

### 6. Repetition Counting (Angle Transition Detection)  
The number of repetitions is calculated based on the transition from joint angle increase to decrease. The conventional method of calculating based on the increase/decrease in the y-coordinate value had a high probability of false detection when the arm angle was bent or rotated.

Pose-Coach detects transition intervals based on upward (≥120°) and downward (≤60°) movements of the elbow or shoulder angles, and prevents duplicate counts by setting a minimum time interval as a condition.

This method enables **motion-meaning-based repetition analysis**, allowing for more precise repetition measurements.

---

### 7. Speed Evaluation (Relative Tempo Score)  
The movement speed is calculated based on the time ratio `r` between the user's repetition cycle (`T_user`) and the reference video (`T_label`), and the score is calculated using the following piecewise function:
```text
S(r) = 
  2r              , r ≤ 0.5  
  1               , 0.5 < r ≤ 1.0  
  2(1.5 − r)      , 1.0 < r ≤ 1.5  
  0               , r > 1.5  
where r = T_label / T_user
```

This method goes beyond simply judging speed as fast or slow, and instead scores **movement speed**, which is useful for maintaining an appropriate speed during training or rehabilitation.

---

### 8. Visual Feedback (Skeleton Overlay with Error Highlighting)  
Pose-Coach provides **skeleton overlay visualization** so that users can easily recognize when they are performing an incorrect posture.

The user skeleton and label skeleton are displayed side by side in the same position, and joints with large errors are highlighted by color or size to enable **intuitive error recognition**.
<table>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/affe598a-e1e0-49c5-bfde-fb862907bfc2" width="300"/></td>
    <td><img src="https://github.com/user-attachments/assets/e384e10f-94c1-45b6-9ec3-2ee62a02709d" width="300"/></td>
    <td><img src="https://github.com/user-attachments/assets/6eef8c0e-139f-4620-b6ab-41ed87031d03" width="300"/></td>
  </tr>
</table>

This method allows even users without specialized knowledge to quickly recognize and correct errors, and its positive impact on accuracy improvement and repeat performance has been proven through actual user testing.
<table>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/a4a28a2e-418d-49c3-85aa-bb2c02344014" width="300"/></td>
    <td><img src="https://github.com/user-attachments/assets/aef9518b-0859-4057-b885-00543a8a71cc" width="300"/></td>
    <td><img src="https://github.com/user-attachments/assets/47fcce7d-a9a1-4b32-83fd-5ca1227cfcfa" width="300"/></td>
  </tr>
</table>
<br></br>

## User Test
<img width="1000" height="440" alt="User Test Flow Diagrams" src="https://github.com/user-attachments/assets/04d6931b-689f-4324-8028-a0bba9cc0249" />
The user study consisted of six sequential stages: initial instruction with a tutorial video, a pre-test without feedback, corrective feedback delivery, a post-test with feedback, additional refinement feedback, and a final questionnaire assessing feedback clarity, intuitiveness, and user satisfaction.

### Table 1. Effectiveness of AI-based feedback on exercise performance.
The results below show a statistically significant improvement in accuracy, speed, and repetition count after receiving feedback from the proposed system.  
Especially, the Δ Score and t-values confirm meaningful enhancements in all metrics.

<div align="center">

| Measure            | Δ Score ↑ | t-value ↑ | p-value ↓ |
|:------------------:|:---------:|:---------:|:----------:|
| Accuracy (Left, %) |  +8.62%   |   4.51    |  0.00024   |
| Accuracy (Right, %)|  +8.05%   |   3.15    |  0.00525   |
| Speed (Left, %)    | +18.63%   |   3.93    |  0.00089   |
| Speed (Right, %)   | +23.51%   |   4.38    |  0.00032   |
| Count (Left, 0–5)  |  +1.65    |   4.62    |  0.00019   |
| Count (Right, 0–5) |  +1.35    |   2.98    |  0.0073    |

</div>

### Table 2. Results of the user satisfaction survey.
Participants reported high satisfaction with the clarity and intuitiveness of the AI feedback.  
All users (100%) were able to identify posture mistakes through the system, and 95% found the feedback easy to understand.  
Additional comments suggested improvements such as real-time guidance, voice instructions, and more detailed correction prompts.

| Question                                                                                     | Yes        | No         |
|----------------------------------------------------------------------------------------------|------------|------------|
| Were you able to identify which part of your exercise posture was incorrect after receiving AI feedback? | 20 (100%)  | 0 (0%)     |
| Was the feedback provided by the AI system objective and easy to understand?                 | 19 (95%)   | 1 (5%)     |

## Requirements

Essential libraries for running the project
```bash
pip install -r requirements.txt
```

Installing the ViTPose library
```bash
# mmcv Install
git clone https://github.com/open-mmlab/mmcv.git
cd mmcv
git checkout v1.3.9
MMCV_WITH_OPS=1 pip install -e .
cd ..

# ViTPose Install
git clone https://github.com/ViTAE-Transformer/ViTPose.git
cd ViTPose
pip install -v -e .
```

>ViTPose GitHub address and pretrained model download link
>https://github.com/ViTAE-Transformer/ViTPose
<br></br>

## Demo Video

> YouTube Link  
>https://youtu.be/Uw1KAmsxoRo
