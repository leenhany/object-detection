# 🎯 Object Detection App with Streamlit & MediaPipe

This project is a simple web-based **Object Detection App** built with [Streamlit](https://streamlit.io/) and [MediaPipe](https://mediapipe.dev/). The app allows users to upload an image, and it detects and visualizes objects using the **EfficientDet Lite0** model.

---

## 📌 Features

- Upload images in `.jpg`, `.jpeg`, `.png`, or `.wbmp` formats
- Perform real-time object detection using a pre-trained TFLite model
- Visualize the bounding boxes and class labels with detection confidence
- Simple and responsive UI powered by Streamlit

---

## 🖼 Demo

You can try the app locally by running the code below. Here’s an example of what it looks like after detecting objects in an image:
<img width="1757" height="842" alt="Screenshot 2025-07-26 185421" src="https://github.com/user-attachments/assets/90fdb7aa-ef69-4fed-9d78-032e9d128647" />


---

## 📂 Project Structure

```
.
├── efficientdet_lite0.tflite       # Pre-trained model file (download from MediaPipe)
├── obj_det.ipynb                   # nootbook for app(optional)
├── app.py                         # Main Streamlit app
└── README.md
```

---

## 🛠️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/object-detection-app.git
cd object-detection-app
```

### 2. Install required libraries

install the main libraries:

```bash
pip install streamlit opencv-python mediapipe numpy
```

---

## 🚀 Run the App

```bash
streamlit run app.py
```

Then open your browser and go to [http://localhost:8501](http://localhost:8501)

---

## 📦 Model Used

- **EfficientDet Lite0**: A fast and efficient object detection model trained on the COCO dataset.
- The model can detect common objects like:
  - Person
  - Dog
  - Car
  - Chair
  - Bottle
  - Etc.

> 🛑 Note: Objects like **ball** may not be detected if they are not part of the COCO label set.

---

## 👩‍💻 Author

**Leen Hany**  
GitHub: [@leenhany](https://github.com/leenhany)  
LinkedIn: [Leen Hany](https://www.linkedin.com/in/leen-hany-481850220/)

---


