import pandas as pd
import numpy as np
import cv2
from nnet import nn


df_train = pd.read_csv("faces_vs_nonfaces_train_32by32.csv") # shape: (29812, 1025) ;-;
df_test = pd.read_csv("faces_vs_nonfaces_test_32by32.csv") # shape: (4045, 1025) ;-;

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

def main():
    model = nn([1024, 256, 128, 64, 1], 0.0005, 50, df_train, "relu") # output is sigmoid
    model.learn()
    model.predict(df_test)

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        height, width = frame.shape[:2]
        min_dim = min(height, width)
        start_x = (width - min_dim) // 2
        start_y = (height - min_dim) // 2
        square_frame = frame[start_y:start_y+min_dim, start_x:start_x+min_dim]

        result = predict_face(model, square_frame)
        
        t = "Face" if result else "Not Face"
        color = (0, 255, 0) if result else (0, 0, 255)
        cv2.putText(square_frame, t, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        cv2.imshow("nn view", cv2.resize(preprocess_frame(square_frame).reshape(32, 32), (200, 200)))
        cv2.imshow("cam", square_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()