import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from PIL import Image
import math
from tensorflow.keras.models import load_model


model = load_model("quickdraw-apple.h5")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils


class_labels = ["apple", "alarm clock", "airplane", "axe", "baseball bat"]

# Initialize drawing variables
draw_color = (0, 255, 0)
draw_radius = 5
last_x, last_y = None, None


def get_index_fingertip_position(hand_landmarks, image_shape):
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    h, w, _ = image_shape
    x = int(index_tip.x * w)
    y = int(index_tip.y * h)
    return x, y


def get_thumb_fingertip_position(hand_landmarks, image_shape):
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    h, w, _ = image_shape
    x = int(thumb_tip.x * w)
    y = int(thumb_tip.y * h)
    return x, y


def preprocess_drawing(canvas):
    drawing_resized = cv2.resize(canvas, (28, 28))
    drawing_gray = cv2.cvtColor(drawing_resized, cv2.COLOR_BGR2GRAY)
    drawing_input = drawing_gray.reshape(1, 28, 28, 1) / 255.0
    return drawing_input


def predict_drawing(canvas, threshold=0.6):

    processed_input = preprocess_drawing(canvas)
    
    prediction = model.predict(processed_input)
    max_prob = np.max(prediction)
    
    predicted_index = np.argmax(prediction)
    
   
    probabilities = {class_labels[i]: round(float(prob), 2) for i, prob in enumerate(prediction[0])}
    st.write("Prediction Probabilities:", probabilities)
    
    
    if max_prob < threshold:
        return "Unknown"
    
    predicted_class_name = class_labels[predicted_index]
    
    return predicted_class_name


st.set_page_config(page_title="DrawIt",layout="wide")

st.markdown(
    """
    <style>
    .title {
        text-align: center;
        font-size: 36px;
        font-weight: bold;
    }
    </style>
    <div class="title">DRAW IT RIGHT by DrawIt!</div>
    """, 
    unsafe_allow_html=True
)


clear_canvas = st.button("Clear Canvas", key="clear_canvas")
classify_drawing = st.button("Classify Drawing", key="classify_button")

col1, col2 = st.columns([1, 1])


frame_placeholder = col1.empty()
canvas_placeholder = col2.empty()


canvas = np.ones((480, 640, 3), dtype=np.uint8) * 255


cap = cv2.VideoCapture(0)

if not cap.isOpened():
   
    st.error("Unable to access the webcam. Please check your camera settings.")


while cap.isOpened():
    
    success, image = cap.read()
    if not success:
    
        continue

    # Flip and process the image
    image = cv2.flip(image, 1)
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

 
    frame_placeholder.image(image_rgb, use_container_width=True)

   
   
    if results.multi_hand_landmarks:
       
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(image_rgb, hand_landmarks, mp_hands.HAND_CONNECTIONS)
         
            x, y = get_index_fingertip_position(hand_landmarks, image.shape)
            thumb_x, thumb_y = get_thumb_fingertip_position(hand_landmarks, image.shape)
         
            distance = math.sqrt((x - thumb_x) ** 2 + (y - thumb_y) ** 2)

            #
           
            drawing_mode = distance < 30
            if drawing_mode:
                if last_x is not None and last_y is not None:
           
                    cv2.line(canvas, (last_x, last_y), (x, y), draw_color, draw_radius)
                last_x, last_y = x, y
            else:
           
                last_x, last_y = None, None


    canvas_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
   
    canvas_placeholder.image(canvas_rgb, caption="Drawing Canvas", use_container_width=True)

    
    if clear_canvas or classify_drawing:
        break

cap.release()

cv2.destroyAllWindows()


if clear_canvas:
    canvas = np.ones((480, 640, 3), dtype=np.uint8) * 255  
   
    canvas_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)  
  
    canvas_placeholder.image(canvas_rgb, caption="Drawing Canvas", use_container_width=True)


if classify_drawing:
    result = predict_drawing(canvas)
    
    st.write(f"Predicted class: {result}")

