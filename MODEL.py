from keras.models import load_model
import cv2
import numpy as np

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("keras_model.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()

# CAMERA can be 0 or 1 based on default camera of your computer
camera = cv2.VideoCapture(0)

while True:
    # Grab the webcamera's image.
    ret, frame = camera.read()

    # Resize the raw image into (224-height,224-width) pixels
    resized_frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)

    # Normalize the image array
    image = np.asarray(resized_frame, dtype=np.float32).reshape(1, 224, 224, 3)
    image = (image / 127.5) - 1

    # Predicts the model
    prediction = model.predict(image)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()
    confidence_score = prediction[0][index]

    # Determine the color based on confidence score
    if confidence_score > 0.8:
        color = (0, 255, 0)  # Green for confidence > 80%
    elif 0.5 <= confidence_score <= 0.8:
        color = (0, 255, 255)  # Yellow for confidence between 50% and 80%
    else:
        color = (0, 0, 255)  # Red for confidence < 50%

    # Display the prediction and confidence score on the frame
    text = f"Class: {class_name[2:]} | Confidence: {confidence_score * 100:.2f}%"
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

    # Print the same information to the terminal
    print(f"Class: {class_name[2:]} | Confidence: {confidence_score * 100:.2f}%")

    # Show the image in a window
    cv2.imshow("Webcam Image", frame)

    # Listen to the keyboard for presses.
    keyboard_input = cv2.waitKey(1)

    # 27 is the ASCII for the esc key on your keyboard.
    if keyboard_input == 27:
        break

camera.release()
cv2.destroyAllWindows()
