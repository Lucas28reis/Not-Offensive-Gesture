from streamlit_webrtc import webrtc_streamer as webrtc
import streamlit as st
import cv2
import mediapipe as mp
import tensorflow as tf
import numpy as np
import av
import time



# html_temp = """
# <div style="background-color:#f4f4f4 ;padding:10px;margin:auto">
# st.image("1.png")
# </div>
# """
# st.markdown(html_temp, unsafe_allow_html=True)
st.image("path1.png")

st.sidebar.title("Menu")
# Add a selectbox to the sidebar:
pages=['HANDY']
add_pages = st.sidebar.selectbox('', pages)

st.sidebar.title("Criadores:")
html_temp6 = """
<ul style="font-weight:bold;">
<li>Gabriel Dias</li>
<li>Isabelle Melo</li>
<li>Lucas Reis </li>
<li>Gustavo Melo</li>
</ul>
"""
st.sidebar.markdown(html_temp6, unsafe_allow_html=True)

if add_pages=='HANDY':
    html_temp2 = """
<body style="background-color:#black ;padding:10px;">
<h3 style="color:white ;text-align:center;">Sobre</h3>
<p style="text-align:justify;">O trabalho visa a implementação de um classificador de gestos ofensivos utilizando a abordagem de aprendizado profundo. Para isso, foi criado um banco de dados de imagens separadas em duas classes: ofensivo e não ofensivo. Este banco de dados alimentou o treinamento dos modelos, respectivamente separados em redes convolucionais [1] e redes pré-treinadas a partir do Xception [2] com dados coloridos e em escala de cinza. Os resultados finais mostram que modelos de aprendizado profundo, principalmente no contexto de imagens,dependem de um grande volume de dados para que a tarefa de compreensão de padrão de dados
seja bem sucedida.</p>
</body>
<div style="background-color:;padding:10px;margin-bottom:10px;">
<h4 style="color:white;">Prepared using:</h4>
<ul style="color:white;">
<li>Deep Learning </li>
<li>Processamento de Imagens</li>
<li>Transfer Learning </li>
<li>Opencv </li>
<li>Keras </li>
<li>Streamlit </li>
<li>PyAutoGui </li>

</ul>
</div>
"""
 
    st.markdown(html_temp2, unsafe_allow_html=True)
# ________________________________________________________________________CODESPACE _______________________________________________________________

# mediapipe init
mp_hands = mp.solutions.hands

# Load the pre-trained gesture classification model
model = tf.keras.models.load_model('./model.h5')  # Substitua pelo caminho para o seu modelo

# Set the desired frame rate
desired_fps = 30

ohm = st.checkbox("Rodar Detector de Gestos Ofensivos")

def transform(img):
    frame = img.to_ndarray(format="bgr24")
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:

        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks and ohm:
            for hand_landmarks in results.multi_hand_landmarks:
                # Hands coordinates
                landmarks = []
                for landmark in hand_landmarks.landmark:
                    h, w, _ = frame.shape
                    x, y = int(landmark.x * w), int(landmark.y * h)
                    landmarks.append((x, y))

                # Bounding box
                x_min = min(landmarks, key=lambda x: x[0])[0]
                x_max = max(landmarks, key=lambda x: x[0])[0]
                y_min = min(landmarks, key=lambda x: x[1])[1]
                y_max = max(landmarks, key=lambda x: x[1])[1]

                # Resize bounding box
                expansion_factor = 1.5
                width = x_max - x_min
                height = y_max - y_min
                x_min = max(0, x_min - int(width * (expansion_factor - 1) / 2))
                x_max = min(frame.shape[1], x_max + int(width * (expansion_factor - 1) / 2))
                y_min = max(0, y_min - int(height * (expansion_factor - 1) / 2))
                y_max = min(frame.shape[0], y_max + int(height * (expansion_factor - 1) / 2))

                # Extract hand gesture image
                hand_gesture = frame[y_min:y_max, x_min:x_max]

                # Preprocess hand gesture image (resize to model input size)
                hand_gesture = cv2.resize(hand_gesture, (150, 150))
                hand_gesture = hand_gesture / 255.0  # Normalize
                hand_gesture = np.expand_dims(hand_gesture, axis=0)  # Add batch dimension

                # Classify hand gesture using the model
                prediction = model.predict(hand_gesture)
                probability_offensive = prediction[0][0]
                probability_non_offensive = 1 - probability_offensive

                # Define the class based on the probability
                if probability_non_offensive > probability_offensive:
                    predicted_class = "Non-offensive"
                else:
                    predicted_class = "Offensive"

            # Draw bounding box (moved outside the loop)
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            # Display class label (moved outside the loop)
            cv2.putText(frame, predicted_class, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            if predicted_class == 'Offensive':
                # Gaussian blur 
                roi = frame[y_min:y_max, x_min:x_max]
                roi = cv2.GaussianBlur(roi, (121, 121), 0)
                frame[y_min:y_max, x_min:x_max] = roi

            output = av.VideoFrame.from_ndarray(frame, format="bgr24")
            return output
        else:
            return img

def main():

    webrtc(key="sample", video_frame_callback=transform, media_stream_constraints={"video": True, "audio": False, "video_options": {"frameRate": desired_fps}})

if __name__ == "__main__":
    main()
