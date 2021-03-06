# import the necessary packages
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import label_binarize, LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
# from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import os

INIT_LR = 1e-4
EPOCHS = 20
BS = 32


def add_labels(inp):
    print("[INFO] Loading images...")
    data = []
    labels = []
    for l in inp:
        print(l)
        # CHANGE here to add local path to data folder in your machine
        p = os.path.join("C:\\Users\\19292\\Documents\\GitHub\\MaskDetectionFinalProject\\data", l)
        print(p)
        for img in os.listdir(p):
            img_path = os.path.join(p, img)
            temp_image = load_img(img_path, target_size=(224, 224))
            temp_image = img_to_array(temp_image)
            data.append(temp_image)
            labels.append(l)
    return data, labels


def runModel(data, labels):
    # perform one-hot encoding on the labels
    lb = LabelBinarizer()
    labels = lb.fit_transform(labels)


    data = np.array(data, dtype="float32")
    labels = np.array(labels)

    ## reshape to pass in correcly
    print(data.shape)
    print(labels.shape)
    (trainX, testX, trainY, testY) = train_test_split(data, labels,
                                                      test_size=0.33, stratify=labels, random_state=42)

    # construct the training image generator for data augmentation
    aug = ImageDataGenerator(
        rotation_range=20,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest")

    # load the MobileNetV2 network, ensuring the head FC layer sets are
    # left off
    baseModel = MobileNetV2(weights="imagenet", include_top=False,
                            input_tensor=Input(shape=(224, 224, 3)))

    # construct the head of the model that will be placed on top of the
    # the base model
    headModel = baseModel.output
    headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
    headModel = Flatten(name="flatten")(headModel)
    headModel = Dense(224, activation="relu")(headModel)
    headModel = Dropout(0.5)(headModel)
    headModel = Dense(3, activation="softmax")(headModel)

    # place the head FC model on top of the base model (this will become
    # the actual model we will train)
    model = Model(inputs=baseModel.input, outputs=headModel)

    # loop over all layers in the base model and freeze them so they will
    # *not* be updated during the first training process
    for layer in baseModel.layers:
        layer.trainable = False

    # compile our model
    print("[INFO] compiling model...")
    opt = Adam(learning_rate=INIT_LR, decay=INIT_LR / EPOCHS)
    model.compile(loss="categorical_crossentropy", optimizer=opt,
                  metrics=["accuracy"])

    # train the head of the network
    print("[INFO] training head...")
    H = model.fit(
        aug.flow(trainX, trainY, batch_size=BS),
        steps_per_epoch=len(trainX) // BS,
        validation_data=(testX, testY),
        validation_steps=len(testX) // BS,
        epochs=EPOCHS)

    # make predictions on the testing set
    print("[INFO] evaluating network...")
    predIdxs = model.predict(testX, batch_size=BS)

    # for each image in the testing set we need to find the index of the
    # label with corresponding largest predicted probability
    predIdxs = np.argmax(predIdxs, axis=1)

    # show a nicely formatted classification report
    print(classification_report(testY.argmax(axis=1), predIdxs,
                                target_names=lb.classes_))

    # serialize the model to disk
    print("[INFO] saving mask detector model...")
    model.save("mask_detector2.model", save_format="h5")

    # plot the training loss and accuracy
    N = EPOCHS
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig("plot.png")


if __name__ == '__main__':
    #, "mask_weared_incorrect"
    data, labels = add_labels(["with_mask", "without_mask", "mask_weared_incorrect"])
    runModel(data, labels)

#(7553, 224, 224, 3)