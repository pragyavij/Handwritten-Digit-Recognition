# AI Handwritten Digit Recognition using CNN

A desktop application built with Python and Tkinter that recognizes handwritten digits using a Convolutional Neural Network (CNN) trained on the MNIST dataset.

## Features

- Draw digits using a mouse
- Real-time digit prediction
- Top 3 prediction probabilities
- Image preprocessing using OpenCV
- Prediction history tracking
- User-friendly Tkinter GUI

## Technologies Used

- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Pillow (PIL)
- Tkinter

## Project Workflow

1. User draws a digit on the canvas.
2. The image is converted to grayscale.
3. The digit is cropped and resized.
4. The CNN model processes the image.
5. The predicted digit and confidence scores are displayed.

## Installation

Clone the repository:

```bash
git clone https://github.com/pragyavij/Handwritten-Digit-Recognition.git
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
python digit_recognition.py
```

## Model Information

Dataset: MNIST

Input Shape:

```text
28 x 28 x 1
```

Output Classes:

```text
Digits 0 - 9
```

## Learning Outcomes

Through this project, I learned:

- Convolutional Neural Networks (CNNs)
- Image preprocessing using OpenCV
- GUI development with Tkinter
- Model training and inference using TensorFlow
- Project organization using Git and GitHub

