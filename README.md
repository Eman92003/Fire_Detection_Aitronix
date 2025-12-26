# 🔥 Fire & Smoke Detection using YOLOv8

This project implements a **Fire and Smoke Detection system** using **YOLOv8**, designed to detect fire and smoke in images with high accuracy and real-time performance. The system can be applied in surveillance cameras, early fire warning systems, and safety monitoring applications.

---

## 🚀 Project Overview
- Task: Object Detection
- Model: YOLOv8 (pre-trained & fine-tuned)
- Framework: Ultralytics YOLOv8 (PyTorch)
- Classes:
  - Fire
  - Smoke

---

## 📂 Dataset
- Source: Kaggle – Fire & Smoke Detection Dataset
- Total Images: ~21,000+
- Annotation Format: YOLO
- Data Split:
  - Training set
  - Validation set
  - Test set

---

## 🧪 Methodology
1. Dataset preparation and validation
2. Loading YOLOv8 pre-trained weights
3. Fine-tuning the model for fire & smoke detection
4. Model evaluation on validation and test sets
5. Performance analysis using detection metrics

---

## 📊 Evaluation Results (Test Set)


| Class  | Precision | Recall | mAP@0.5 |
|------|-----------|--------|---------|
| Smoke | 83.7% | 81.1% | 84.8% |
| Fire  | 73.4% | 65.9% | 72.8% |
| **Overall** | **78.5%** | **73.5%** | **78.8%** |

- Inference Speed: ~12 ms per image
- Strong generalization on unseen data

---

## 🧠 Results Interpretation
The model achieved strong generalization performance on the unseen test set. Smoke detection outperformed fire detection due to more stable and distinguishable visual features. Fire detection remains challenging because of variations in flame shape, illumination, and background complexity.  
Overall, YOLOv8 demonstrated robustness and efficiency for real-world fire and smoke detection tasks.

---

## 🎥 Video Inference Demo

Below is a real-time inference demo showing fire and smoke detection on video input:

👉 **[Click here to watch the demo video](https://drive.google.com/file/d/1P0MHRBVb0ZhEZaUsD1MSr-4XEy64sqmn/view?usp=sharing)**

*(The model successfully detects fire and smoke instances with bounding boxes and confidence scores in real-time.)*

---

## 📌 Conclusion
- YOLOv8 provides a strong balance between accuracy and speed
- Effective for real-time fire and smoke monitoring
- The project demonstrates the practical application of deep learning in safety-critical systems

---

## 🛠 Technologies Used
- Python
- PyTorch
- Ultralytics YOLOv8
- OpenCV
- Kaggle
