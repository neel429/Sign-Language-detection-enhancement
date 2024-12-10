# Real-Time Sign Language Recognition and Speech Synthesis

This project provides a system that translates American Sign Language (ASL) gestures into text and speech using computer vision and natural language processing techniques.

---

## Project Overview

The system processes video streams of ASL, classifies hand gestures into alphabets, forms words from recognized letters, and converts them into audible speech. It combines advanced image processing with a Convolutional Neural Network (CNN) for gesture recognition and natural language processing for word and sentence construction.

---

## Methodology

### 1. **Data Collection**
- **Dataset Size**: 13,000 images
- **Classes**: 26 (representing English alphabets)
- **Data Source**: ASL gestures captured via laptop camera
- **Specifications**:
  - Each image: 400x400 pixels
  - Consistent lighting and hand positioning

### 2. **Data Preprocessing**
Steps involved:
1. Resize images to 224x224 resolution.
2. Normalize pixel values to [0, 1].
3. Extract hand landmarks using [MediaPipe](https://google.github.io/mediapipe/), representing skeletal joints of fingers and palms.
4. Create skeletal representations on a black 400x400 background.
5. Resize to 200x200 for efficient training and inference.
6. Use Python-based data generators for real-time data augmentation.

### 3. **Model Training**
- **Architecture**: Custom Convolutional Neural Network (CNN)
  - **Conv2D Layers**:
    - Layer 1: 32 filters, kernel size 3x3
    - Layer 2: 64 filters, kernel size 3x3
  - **MaxPooling2D Layers**: Reduces spatial dimensions.
  - **Flatten Layer**: Converts feature maps into a vector.
  - **Dense Layers**:
    - Layer 1: 128 neurons for feature learning
    - Layer 2: 26 neurons for classification
- **Optimizer**: Adam
- **Loss Function**: Sparse categorical cross-entropy

### 4. **Real-Time Inference**
- **Pipeline**:
  1. Capture live video frames using OpenCV.
  2. Extract hand landmarks with MediaPipe.
  3. Plot landmarks on a black background to generate skeletal images.
  4. Evaluate 20 frames and classify the most frequently occurring alphabet.

### 5. **Word Formation and Speech Synthesis**
- Combine alphabets to form words.
- Preprocess words using Python's [NLTK](https://www.nltk.org/) library.
- Convert text to audible speech.

---

## Results
- **Model Performance**: High classification accuracy achieved using CNN.
- **Real-Time Efficiency**: Processes approximately 20 frames per inference cycle.

---

## Dependencies

### Required Libraries
- OpenCV
- MediaPipe
- TensorFlow/Keras
- NLTK

### Hardware Requirements
- Camera-enabled device for live gesture capture.
- Moderate GPU for model training.

