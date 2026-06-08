from tensorflow.keras.models import load_model
import cv2
import numpy as np
import matplotlib.pyplot as plt

model = load_model("models/mnist_cnn.h5")

img = cv2.imread("images/Screenshot 2026-06-06 160906.png")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

gray = cv2.resize(gray, (28, 28))

gray = 255 - gray

gray = gray / 255.0

gray = gray.reshape(1, 28, 28, 1)

plt.imshow(gray.reshape(28, 28), cmap='gray')
plt.title("Processed Image")
plt.show()

prediction = model.predict(gray)

digit = np.argmax(prediction)

confidence = np.max(prediction) * 100

print("Digit:", digit)
print("Confidence:", confidence, "%")