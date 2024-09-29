import cv2
import dlib
from scipy.spatial import distance as dist
import time
import azure.cognitiveservices.speech as speechsdk
import threading

speech_config = speechsdk.SpeechConfig(subscription="0f1530ace5b74b329819de48cc9a1e66", region="eastus")

def speak_alert():
    speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config)
    while True:
        speech_synthesizer.speak_text_async("Alert! Drowsiness detected.").get()
        time.sleep(2)

# ear = eye aspect ratio 
# most every example I saw of tracking eyes closing or opening used this approach
def calculate_ear(eye):
    A = dist.euclidean(eye[1], eye[5])  
    B = dist.euclidean(eye[2], eye[4])  
    C = dist.euclidean(eye[0], eye[3])  
    ear = (A + B) / (2.0 * C)
    return ear

def calculate_mouth_ear(mouth):
    A = dist.euclidean(mouth[1], mouth[9])
    B = dist.euclidean(mouth[2], mouth[10])
    C = dist.euclidean(mouth[0], mouth[6])
    mar = (A + B) / (2.0 * C)
    return mar


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('face_data.dat')

# threshold for detecting closed eyes
EYE_AR_THRESHOLD = 0.23  # this could be fine tuned but chatgpt said .23 works best
WINK_DIFF_THRESHOLD = 0.08  # idk wink feature was cool and easy to add
eye_closed_counter = 0 # un used as of now
eye_closed_time = None
yawn_counter = 0
yawn_status = False
yawn_start_time = None
alert_counter = 0
last_alert_time = 0
ALERT_CUSHION = 10
#MAX_ALERTS = 2
eye_closed_time = None
alert_thread = None
EYES_CLOSED_THRESHOLD = 5 
## mouth threshold 
MOUTH_AR_THRESHOLD = 0.6

cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break


    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

   
    faces = detector(gray_frame)

 
    gray_frame_colored = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)

    for face in faces:
        landmarks = predictor(gray_frame, face)
        #adding mouth landmarks
        mouth = [(landmarks.part(n).x, landmarks.part(n).y) for n in range(48, 68)]
        mar = calculate_mouth_ear(mouth)
        #counting the yawns not sure if will keep
        if mar > MOUTH_AR_THRESHOLD:
            if not yawn_status:
                yawn_status = True
                yawn_start_time = time.time()
            elif time.time() - yawn_start_time >= 1:
                yawn_counter += 1
                yawn_status = False
            mouth_status = "Yawning"
        else:
            yawn_status = False
            mouth_status = "Not Yawning"

        for (x, y) in mouth:
            cv2.circle(gray_frame_colored, (x, y), 2, (0, 255, 255), -1)


        left_eye = [(landmarks.part(n).x, landmarks.part(n).y) for n in range(42, 48)]
        right_eye = [(landmarks.part(n).x, landmarks.part(n).y) for n in range(36, 42)]

        #reminder ear isn't ear its eye to aspect ratio
        left_ear = calculate_ear(left_eye)
        right_ear = calculate_ear(right_eye)

        # Check if both eyes are closed (both EAR values below threshold)
        if left_ear < EYE_AR_THRESHOLD and right_ear < EYE_AR_THRESHOLD:
            eye_status = "Both Closed"
            if eye_closed_time is None:
                eye_closed_time = time.time()
            elif time.time() - eye_closed_time >= EYES_CLOSED_THRESHOLD:
                if alert_thread is None or not alert_thread.is_alive():
                    alert_thread = threading.Thread(target=speak_alert)
                    alert_thread.daemon = True
                    alert_thread.start()
                cv2.putText(gray_frame_colored, "DROWSINESS ALERT!", (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            eye_status = "Open"
            eye_closed_time = None
            if alert_thread and alert_thread.is_alive():
                alert_thread = None

        # draw the landmarks for both eyes in color on the grayscale frame
        for (x, y) in left_eye:
            cv2.circle(gray_frame_colored, (x, y), 2, (0, 255, 0), -1)  
        for (x, y) in right_eye:
            cv2.circle(gray_frame_colored, (x, y), 2, (255, 0, 0), -1)  


        
        #creating to it turns red if signs of sleepiness which will sound alarm in next iteration
        current_time = time.time()
        if (eye_closed_counter % 5 == 0 and eye_closed_counter != 0) or (yawn_counter % 3 == 0 and yawn_counter != 0):
            if current_time - last_alert_time > ALERT_CUSHION:
                cv2.putText(gray_frame_colored, "DROWSINESS ALERT!", (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                threading.Thread(target=speak_alert).start()
                alert_counter += 1
                last_alert_time = current_time
                
        avg_ear = (left_ear + right_ear) / 2.0 # this could removed later it was for debugging
        ##cv2.putText(gray_frame_colored, f"EAR: {avg_ear:.2f}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(gray_frame_colored, f"Eye Status: {eye_status}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(gray_frame_colored, f"Mouth Status: {mouth_status}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(gray_frame_colored, f"Eye Closed Counter: {eye_closed_counter}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(gray_frame_colored, f"Yawn Count: {yawn_counter}", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)


    cv2.imshow('Facial Landmarks', gray_frame_colored) # gray frame so the colors stand out


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
