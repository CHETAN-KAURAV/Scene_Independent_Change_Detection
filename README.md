# Temporal-Change-Detection

## Scene-Independent Change Detection using Explicit Temporal Differencing and Deep Segmentation Models

---

# Overview

This project explores the problem of **change detection in video sequences**, where the objective is to identify meaningful motion or foreground changes between consecutive frames.

The project was developed as an experimental computer vision research pipeline focused on:

- temporal motion modeling
- foreground segmentation
- scene-independent change detection
- architecture exploration for motion understanding

The implementation uses explicit temporal differencing along with a deep segmentation architecture to detect moving objects in dynamic scenes.

---

# Motivation

Traditional foreground detection methods often struggle in scenarios involving:

- illumination changes
- camera noise
- dynamic backgrounds
- scene-independent environments
- weak motion boundaries

Instead of relying purely on implicit feature learning, this project investigates whether:

> **explicit temporal motion representation can improve change detection performance more effectively than increasing architectural complexity.**

---

# Problem Statement

Given a sequence of video frames:

```text
Frame(t-2), Frame(t-1), Frame(t)
```
the task is to predict a binary segmentation mask identifying:

foreground motion meaningful scene changes moving objects while suppressing background noise.

---

# Dataset

The project uses video sequences inspired by the CDnet2014-style change detection benchmark.

Each scene contains:

```
input/
groundtruth/
```
The pipeline automatically:

- handles frame ordering
- aligns ground-truth masks
- filters missing annotations
- constructs temporal samples

---

# Methodology

## Temporal Differencing

The core contribution of this project is the use of explicit temporal differencing.

Instead of feeding raw stacked frames directly into the network, temporal motion is explicitly modeled using:

```commandline
diff1 = abs(Frame(t) - Frame(t-1))
diff2 = abs(Frame(t) - Frame(t-2))
```

These difference maps are concatenated to form the final input tensor.

This significantly improved motion localization compared to implicit temporal learning.

## Deep Segmentation Architecture

The segmentation model consists of:

- Encoder
- ResNet18 backbone
- modified for 6-channel temporal input
- Decoder
- lightweight U-Net style decoder
- progressive upsampling using transposed convolutions

The architecture performs dense foreground prediction at pixel level.


---

# Architecture

```commandline
Temporal Frames
       │
       ▼
Explicit Frame Differencing
       │
       ▼
6-Channel Motion Representation
       │
       ▼
ResNet18 Encoder
       │
       ▼
U-Net Style Decoder
       │
       ▼
Foreground Change Mask
```

---

# Experiments Conducted

This project involved multiple experimental explorations.

## Baseline Model
Initial experiments used simple temporal frame stacking without explicit differencing.

### Observation:
- weak motion understanding
- blurry segmentation
- poor localization

## U-Net Decoder Integration
A decoder with progressive upsampling was introduced.

### Improvement:
- significantly better spatial reconstruction
- improved segmentation quality

## Explicit Temporal Differencing
Explicit motion representation using frame differences was introduced.
### Key Observation:
Explicit temporal differencing contributed more to performance improvement than increasing architectural complexity.

This became the most effective component in the pipeline.

## Temporal Attention Experiments
Lightweight temporal attention mechanisms were explored.

### Observation
- marginal improvements
- explicit motion representation already captured most temporal information

## Self-Supervised Pretraining
Temporal order prediction was explored as a self-supervised pretraining task.

### Observation
- limited gains
- explicit motion cues already provided strong supervision

## Edge-Aware Supervision
Boundary-aware and edge-guided supervision strategies were tested.

### Observation
- unstable optimization
- aggressive boundary weighting sometimes degraded segmentation quality.



## Final Observations
The final experiments revealed several important insights:

Insight 1:

```Explicit temporal representation is more effective than relying solely on implicit temporal feature learning.```

Insight 2:

```Increasing architectural complexity does not necessarily improve segmentation quality.```

Insight 3:

```Stable training and clean temporal representation are often more valuable than adding multiple advanced modules.```

---

# Results:

The final pipeline achieved:

- stable foreground detection
- scene-independent motion understanding
- meaningful object localization

while maintaining a relatively lightweight architecture.

# Sample Results

Input Frame
- video scene frame

Ground Truth
- binary foreground mask

Prediction
- predicted foreground motion mask

Example outputs are available in:

```sample_results/```

---


# Repository Architecture:

```commandline

Temporal-Change-Detection/
│
├── models/
│   └── model.py
│
├── utils/
│   └── dataset.py
│
├── experiments/
│   └── pretrain.py
│   └── dataset_verify.py
│
├── sample_results/
│
├── train.py
├── evaluate.py
├── requirements.txt
└── README.md
```

---

# Installation

## Clone Repository
```
git clone <your_repo_link>
cd Temporal-Change-Detection
```
## Install Dependencies

```pip install -r requirements.txt```

## Training

```Python train.py```

## Evaluation

```Python evaluate.py```

---

# Future Work

Possible future extensions include:

- transformer-based temporal modeling
- optical flow integration
- multi-scale feature fusion
- temporal consistency constraints
- lightweight real-time deployment
- unsupervised motion segmentation


---

# Key Learning Outcomes

Through this project, the following practical insights were gained:

- temporal representation design
- segmentation model behavior
- effects of supervision strategies
- instability caused by aggressive loss engineering
- importance of evaluation consistency
- iterative experimentation in computer vision research

# Conclusion

This project demonstrates that:

carefully designed temporal motion representation can substantially improve scene-independent change detection performance, often more effectively than simply increasing model complexity.

The work emphasizes experimentation, reasoning, and architectural understanding rather than focusing purely on benchmark optimization.

