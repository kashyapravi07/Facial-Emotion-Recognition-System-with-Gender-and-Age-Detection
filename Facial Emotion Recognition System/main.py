import cv2

# Load Haar Cascades for face, eye, and smile detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

# Load gender and age models
age_net = cv2.dnn.readNetFromCaffe('C:/Users/Ravi Shankar Jha/Desktop/opencv project/myenv/model/age_deploy.prototxt', 'C:/Users/Ravi Shankar Jha/Desktop/opencv project/myenv/model/age_net.caffemodel')
gender_net = cv2.dnn.readNetFromCaffe('C:/Users/Ravi Shankar Jha/Desktop/opencv project/myenv/model/gender_deploy.prototxt', 'C:/Users/Ravi Shankar Jha/Desktop/opencv project/myenv/model/gender_net.caffemodel')

# Define model mean values and labels
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
gender_list = ['Male', 'Female']

# Initialize webcam
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Extract the region of interest (ROI)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # Detect eyes
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

        # Detect smile (mouth)
        smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.8, minNeighbors=20)
        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (0, 255, 0), 2)

        # Simple rule-based emotion inference
        if len(smiles) > 0:
            emotion = "Happy"
        elif len(eyes) == 2:  # assuming that if eyes are detected but no smile, it could be neutral or sad
            # Basic heuristic for sad: distance between eyes and bottom of the face
            eye_y = max(ey + eh for (ex, ey, ew, eh) in eyes)
            if (y + h) - eye_y > h * 0.5:  # if lower part of the face is significant compared to the upper part
                emotion = "Sad"
            else:
                emotion = "Neutral"
        else:
            emotion = "Neutral"

        # Predict gender
        face_blob = cv2.dnn.blobFromImage(roi_color, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        gender_net.setInput(face_blob)
        gender_preds = gender_net.forward()
        gender = gender_list[gender_preds[0].argmax()]

        # Predict age
        age_net.setInput(face_blob)
        age_preds = age_net.forward()
        age = age_list[age_preds[0].argmax()]

        # Display emotion, gender, and age on the screen
        cv2.putText(frame, f'{gender}, {age}, {emotion}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Display message for sad emotion
        if emotion == "Sad":
            cv2.putText(frame, "Don't forget to smile", (x, y + h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Display the frame with detection and emotion recognition
    cv2.imshow('Emotion, Gender and Age Recognition', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
