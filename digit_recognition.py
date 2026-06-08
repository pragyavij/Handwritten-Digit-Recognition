import tkinter as tk
from PIL import Image, ImageDraw, ImageTk
import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("models/mnist_cnn.keras")

root = tk.Tk()
root.title("Digit Recognition")

canvas = tk.Canvas(root, width=300, height=300, bg="black")
canvas.pack()

image = Image.new("L", (300, 300), "black")
draw_image = ImageDraw.Draw(image)
history = []

def draw(event):
    x = event.x
    y = event.y

    canvas.create_oval(
        x-4, y-4,
        x+4, y+4,
        fill="white",
        outline="white"
    )

    draw_image.ellipse(
        [x-4, y-4, x+4, y+4],
        fill="white"
    )

def clear_canvas():
    canvas.delete("all")

    global image, draw_image

    image = Image.new("L", (300, 300), "black")
    draw_image = ImageDraw.Draw(image)

    result_label.config(
        text="Draw a digit"
    )

def save_image():
    image.save("digit.png")
    print("Image Saved")

def predict_digit():

        image.save("digit.png")

        img = cv2.imread("digit.png")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        _, thresh = cv2.threshold(
        gray,
        50,
        255,
        cv2.THRESH_BINARY
    )

        coords = cv2.findNonZero(thresh)
        x, y, w, h = cv2.boundingRect(coords)
        if coords is None:
            result_label.config(
                text="Please draw a digit"
            )
            return

        digit_roi = gray[y:y+h, x:x+w]
        h, w = digit_roi.shape

        if h > w:

            new_h = 20
            new_w = int(w * (20 / h))

        else:

            new_w = 20
            new_h = int(h * (20 / w))

        digit_roi = cv2.resize(
            digit_roi,
             (new_w, new_h)
        )
        gray = np.zeros((28, 28), dtype=np.uint8)
        x_offset = (28 - new_w) // 2
        y_offset = (28 - new_h) // 2

        gray[
            y_offset:y_offset + new_h,
            x_offset:x_offset + new_w
        ] = digit_roi

        preview_img = (gray.copy()).astype(np.uint8)

        gray = gray / 255.0

        gray = gray.reshape(1, 28, 28, 1)
        display_img = Image.fromarray(preview_img)

        display_img = display_img.resize((140,140))

        photo = ImageTk.PhotoImage(display_img)

        preview_label.config(image=photo)

        preview_label.image = photo
        history_label = tk.Label(
        root,
        text="Prediction History",
        justify="left",
        font=("Arial", 12)
        )

        history_label.pack()

        prediction = model.predict(gray)
        probs = prediction[0]

        top3 = np.argsort(probs)[-3:][::-1]

        text = "Top Predictions\n\n"

        for idx in top3:
            text += f"{idx} → {probs[idx]*100:.2f}%\n"

        result_label.config(text=text)
        history.append(
            f"{top3[0]} ({probs[top3[0]]*100:.1f}%)"
        )
        history_text = "Prediction History\n\n"

        for item in history[-10:][::-1]:
            history_text += item + "\n"

        history_label.config(
        text=history_text
        )
       
canvas.bind("<B1-Motion>", draw)

clear_btn = tk.Button(root, text="Clear", command=clear_canvas)
clear_btn.pack()

save_btn = tk.Button(root, text="Save", command=save_image)
save_btn.pack()
predict_btn = tk.Button(
    root,
    text="Predict",
    command=predict_digit
)
predict_btn.pack()
result_label = tk.Label(
    root,
    text="Draw a digit",
    font=("Arial", 16)
)

result_label.pack()
preview_label = tk.Label(root)
preview_label.pack()


root.mainloop()
