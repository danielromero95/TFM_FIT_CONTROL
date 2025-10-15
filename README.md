# FIT CONTROL

**Repository:** `TFM_FIT_CONTROL`  
**Author:** Daniel Romero  
**Last Update:** 15 de Octubre de 2025

---

## Description

This project aims to build a exercise analysis system based on computer vision.
From strength training videos (squat, bench press, deadlift), the app (desktop app and streamlit app) should:
- Detect exercise type.
- Automatically count repetitions.
- Calculate key joint angles (knee, hip, shoulder).
- Detect potential technique errors with ≥90% accuracy.
- Offer a minimal web demo to view results.


The general workflow includes:
- Video preprocessing: frame extraction, resizing, normalization, filtering (Gauss, CLAHE), ROI cropping.
- Pose estimation: MediaPipe Pose to extract landmarks and calculate angles, angular velocities, symmetry, etc.


Modeling:
- Repeat counting by peak detection in angle series.
- Fault detection (still to be decided how to do it)


This repository contains the basic structure to begin developing each of the modules in a modular and traceable manner using Git.
