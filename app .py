# import os
# import datetime, time
# from threading import Thread
#
# from flask import Flask, render_template, Response, request
# import cv2
# from keras.preprocessing import image
# import numpy as np
# from keras.models import model_from_json
#
#
#
# app = Flask(__name__)
#
# global capture, rec_frame, grey, switch, ans, camera
#
# capture = 0
#
# switch = 1
#
# app = Flask(__name__, template_folder='./templates')
# camera = cv2.VideoCapture(0)
#
# # make shots directory to save pics
# try:
#     os.mkdir('./shots')
# except OSError as error:
#     pass
#
#
# def gen_frames():
#     # generate frame by frame from camera
#     global out, capture, rec_frame, ans, resized_img
#     json_file = open('E://MajorProjectLatest//test_fol//fer.json', 'r')
#
#     loaded_model_json = json_file.read()
#     json_file.close()
#     model = model_from_json(loaded_model_json)
#
#     # # Load weights and them to model
#     # """ model.load_weights('E:/MajorProjectLatest/fer.h5')  """
#     model.load_weights('E://MajorProjectLatest//test_fol//fer.h5')
#
#     while True:
#         success, frame = camera.read()
#
#         if success:
#
#             face_haar_cascade = cv2.CascadeClassifier('E://MajorProjectLatest//test_fol'
#                                                       '//haarcascade_frontalface_default.xml')
#
#             ret, img = camera.read()
#             if not ret:
#                 break
#
#             gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#             faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.1, 6, minSize=(150, 150))
#             for (x, y, w, h) in faces_detected:
#                 x1, y1 = x + w, y + h
#                 cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 1)
#                 cv2.line(img, (x, y), (x + 30, y), (0, 0, 255), 6)  # Top Left
#                 cv2.line(img, (x, y), (x, y + 30), (0, 0, 255), 6)
#
#                 cv2.line(img, (x1, y), (x1 - 30, y), (0, 0, 255), 6)  # Top Right
#                 cv2.line(img, (x1, y), (x1, y + 30), (0, 0, 255), 6)
#
#                 cv2.line(img, (x, y1), (x + 30, y1), (0, 0, 255), 6)  # Bottom Left
#                 cv2.line(img, (x, y1), (x, y1 - 30), (0, 0, 255), 6)
#
#                 cv2.line(img, (x1, y1), (x1 - 30, y1), (0, 0, 255), 6)  # Bottom right
#                 cv2.line(img, (x1, y1), (x1, y1 - 30), (0, 0, 255), 6)
#                 roi_gray = gray_img[y:y + w, x:x + h]
#                 roi_gray = cv2.resize(roi_gray, (48, 48))
#                 img_pixels = image.img_to_array(roi_gray)
#                 img_pixels = np.expand_dims(img_pixels, axis=0)
#                 img_pixels /= 255.0
#
#                 predictions = model.predict(img_pixels)
#                 max_index = int(np.argmax(predictions))
#                 emotions = ['angry', 'happy', 'neutral', 'sad']
#                 predicted_emotion = emotions[max_index]
#
#
#                 cv2.putText(img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255),
#                             2)
#
#             ret, buffer = cv2.imencode('.jpg', cv2.resize(img, (1000, 700)))
#             img = buffer.tobytes()
#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + img + b'\r\n')
#
#         try:
#
#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
#         except Exception as e:
#             pass
#
#     else:
#         pass
#
#
# @app.route("/")
# def index():
#     return render_template('index.html')
#
#
# @app.route("/home")
# def home():
#     return render_template('index.html')
#
#
# @app.route("/about")
# def about():
#     return render_template('about.html')
#
#
# @app.route("/camera")
# def camera():
#     return render_template('camera.html')
#
#
# @app.route('/requests', methods=['POST', 'GET'])
# def tasks():
#     global switch, camera, loaded_model_json, model, json_file
#     if request.method == 'POST':
#
#         if request.form.get('stop') == 'Stop/Start':
#
#             if switch == 1:
#                 switch = 0
#                 camera.release()
#                 cv2.destroyAllWindows()
#
#             else:
#
#                 camera = cv2.VideoCapture(0)
#
#                 switch = 1
#
#
#
#     elif request.method == 'GET':
#         return render_template('camera.html')
#     return render_template('camera.html')
#
#
# @app.route('/video_feed')
# def video_feed():
#     return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
#
#
# if __name__ == '__main__':
#     app.run(debug=True)


import cv2
import numpy as np
import pyttsx3
import streamlit as st
from keras.models import model_from_json
from keras.preprocessing.image import img_to_array
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# load model
emotion_dict = {0: 'angry', 1: 'happy', 2: 'neutral', 3: 'sad'}
# load json and create model
json_file = open('fer.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
classifier = model_from_json(loaded_model_json)

# load weights into new model
classifier.load_weights("fer.h5")

# load face
try:
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
except Exception:
    st.write("Error loading cascade classifiers")


def text_to_speech(text):
    engine = pyttsx3.init()

    engine.say(text)
    engine.runAndWait()


class VideoTransformer(VideoTransformerBase):

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # image gray
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            image=img_gray, scaleFactor=1.3, minNeighbors=5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img=img, pt1=(x, y), pt2=(
                x + w, y + h), color=(255, 0, 0), thickness=2)
            roi_gray = img_gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)
                prediction = classifier.predict(roi)[0]
                maxindex = int(np.argmax(prediction))
                finalout = emotion_dict[maxindex]
                output = str(finalout)
                text_to_speech(output)
            label_position = (x, y)
            cv2.putText(img, output, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return img


def main():
    # Face Analysis Application #
    st.title("Real Time Face Emotion Detection Application")
    activiteis = ["Home", "Webcam Face Detection", "About"]
    choice = st.sidebar.selectbox("Select Activity", activiteis)
    st.sidebar.markdown(
        """ Developed by Ritik Verma , Aakash Jain , Gurtej Singh    
            Email : varmaritik04@gmail.com
            [LinkedIn] ()""")
    if choice == "Home":
        html_temp_home1 = """<div style="background-color:#6D7B8D;padding:10px">
                                            <h4 style="color:white;text-align:center;">
                                            Face Emotion detection application using OpenCV, Custom CNN model and Streamlit.</h4>
                                            </div>
                                            </br>"""
        st.markdown(html_temp_home1, unsafe_allow_html=True)
        st.write("""
                 The application has two functionalities.
                 1. Real time face detection using web cam feed.
                 2. Real time face emotion recognization.
                 """)
    elif choice == "Webcam Face Detection":
        st.header("Webcam Live Feed")
        st.write("Click on start to use webcam and detect your face emotion")
        webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)

    elif choice == "About":
        st.subheader("About this app")
        html_temp_about1 = """<div style="background-color:#6D7B8D;padding:10px">
                                    <h4 style="color:white;text-align:center;">
                                    Real time face emotion detection application using OpenCV, Custom Trained CNN model and Streamlit.</h4>
                                    </div>
                                    </br>"""
        st.markdown(html_temp_about1, unsafe_allow_html=True)

        html_temp4 = """                             		<div style="background-color:#98AFC7;padding:10px">
                             		<h4 style="color:white;text-align:center;">This Application is developed by Mohammad Juned Khan using Streamlit Framework, Opencv, Tensorflow and Keras library for demonstration purpose. If you're on LinkedIn and want to connect, just click on the link in sidebar and shoot me a request. If you have any suggestion or wnat to comment just write a mail at Mohammad.juned.z.khan@gmail.com. </h4>
                             		<h4 style="color:white;text-align:center;">Thanks for Visiting</h4>
                             		</div>
                             		<br></br>
                             		<br></br>"""

        st.markdown(html_temp4, unsafe_allow_html=True)

    else:
        pass


if __name__ == "__main__":
    main()
