import pandas as pd
import numpy as np
import cv2
from nnet import nn

df_train = pd.read_csv("faces_vs_nonfaces_train_32by32.csv") # shape: (29812, 1025) ;-;
df_test = pd.read_csv("faces_vs_nonfaces_test_32by32.csv") # shape: (4045, 1025) ;-;

model = nn([1024, 32, 16, 8, 1], 0.001, 20, df_train, "relu") # output is sigmoid
model.learn()
model.predict(df_test)

def preprocess_frame(frame, size=32):
    gs = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gs, (size, size))
    n = resized / 255.0
    
    flat = n.flatten().reshape(-1, 1).astype(np.float32)
    
    return flat

def predict_face(model, frame):
    processed = preprocess_frame(frame)
    
    output = processed
    for layer in model.layers:
        output = layer.forward(output)
        
    return int((output > 0.5).item()) # sigmoid

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]
        result = predict_face(model, face_img)

        label = "Face" if result else "Not Face"
        color = (0, 255, 0) if result else (0, 0, 255)

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.imshow("Webcam Feed", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()