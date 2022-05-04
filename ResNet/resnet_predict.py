from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os



def predict_one(img,model):

    maskNet = load_model(model)
    img = cv2.imread(img)
    faces = []
    face = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    face = cv2.resize(face, (112, 112))
    face = img_to_array(face)
    face = preprocess_input(face)
    faces.append(face)

    faces = np.array(faces, dtype="float32")

    preds = maskNet.predict(faces, batch_size=32)

    maskIncorrect = preds[0][0]
    mask = preds[0][1]
    withoutMask = preds[0][2]
    a = [maskIncorrect, mask, withoutMask]
    l_arr = ["Mask Incorrect", "Mask", "No Mask"]
    label = l_arr[a.index(max(a))]

    print("PREDICTION:", label, "with", max(a) * 100, "% certainty")


    return label


if __name__ == '__main__':
    mask_count = 0
    no_mask_count = 0
    incorrect_mask_count = 0
    modelFileName = "mask_detector_data1.model"
    for img in range(20):
        R = predict_one("data/with_mask/with_mask_" + str(img+1) +  ".jpg", modelFileName)
        if R == "Mask":
            mask_count+=1
        elif R == "No Mask":
            no_mask_count+=1
        else:
            incorrect_mask_count+=1
    print("Number of MASK:", mask_count, "Number of NO MASK:", no_mask_count, "Number of INCORRECT MASK:", incorrect_mask_count)
