import cv2
import mediapipe as mp
from tensorflow.keras.models import load_model
import tensorflow as tf

# Create Labels And Load Model
labels = ['Go', 'Left', 'Right', 'Stop']
model = load_model('my_hand_rec_model3.h5')


# Function That Predicts The Gesture
def Predict_Gesture():
    # Load The Image That Was Saved From The Camera
    new_data = tf.keras.preprocessing.image.load_img('Final.jpg', target_size=(64, 64))

    # PreProcess The Image
    new_data = tf.keras.preprocessing.image.img_to_array(new_data)
    new_data = new_data / 255.

    new_data = tf.expand_dims(new_data, axis=0)

    # Predict With The Model
    prediction = model.predict(new_data)

    # Get the predicted label
    predicted_class = tf.argmax(prediction, axis=1)
    print(f'Predicted class: {labels[predicted_class[0]]}')


# Set Up OpenCV
mphands = mp.solutions.hands
hands = mphands.Hands()
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

_, frame = cap.read()

h, w, c = frame.shape

# Getting The Image From The Camera and Processing It
while True:
    # Get Image and Convert To OpenCV Colours
    _, frame = cap.read()
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Start OpenCV's Search For Hand In Frame
    result = hands.process(framergb)
    hand_landmarks = result.multi_hand_landmarks
    if hand_landmarks:
        for handLMs in hand_landmarks:
            # Get Hand Position
            x_max = 0
            y_max = 0
            x_min = w
            y_min = h
            for lm in handLMs.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                if x > x_max:
                    x_max = x
                if x < x_min:
                    x_min = x
                if y > y_max:
                    y_max = y
                if y < y_min:
                    y_min = y
            # Crop Image Around Hand
            croppedimage = frame[int(y_min - 19):int(y_max + 19), int(x_min - 19):int(x_max + 19)]
            cropout = cv2.rectangle(frame, (x_min - 20, y_min - 20), (x_max + 20, y_max + 20), (0, 255, 0), 1)

            # Change to Greyscale and Invert Because Colour Is Not Necessary And Inverting With Threshold Creates A
            # Higher Contrast Between Foreground And Background
            croppedimage = cv2.cvtColor(croppedimage, cv2.COLOR_RGB2GRAY)
            croppedimage = inverted_img = cv2.bitwise_not(croppedimage)
            croppedimage = cv2.threshold(croppedimage, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
            # Saves Final Image And Gets Model To Predict
            cv2.imwrite("Final.jpg", croppedimage)
            Predict_Gesture()
    cv2.imshow("Frame", frame)

    cv2.waitKey(1)