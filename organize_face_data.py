import os
import cv2
import pandas as pd

data = []
targets = []

#### IMAGES FED ARE CONDENSED INTO 32X32 GREYSCALED ####

for filename in os.listdir('isFace_data/train/face/'):
    if filename.endswith('.jpg'): # just incase
        img = cv2.imread(f'isFace_data/train/face/{filename}', cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (32, 32))
        img = (img / 255.0).astype('float32')
        data.append(img.flatten())
        targets.append(1)  # 1 = face

for filename in os.listdir('isFace_data/train/nonface'):
    if filename.endswith('.jpg'):
        img = cv2.imread(f'isFace_data/train/nonface/{filename}', cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (32, 32))
        img = (img / 255.0).astype('float32')
        data.append(img.flatten())
        targets.append(0)  # 0 = non-face

df_facenonface = pd.DataFrame(data)
df_facenonface['target'] = targets
df_facenonface = df_facenonface.sample(frac=1).reset_index(drop=True)
df_facenonface.to_csv('faces_vs_nonfaces_train_32by32.csv', index=False)