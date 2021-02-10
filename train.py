import cv2
import numpy as np
import os
import sys
from tensorflow.keras import layers

print("Activating TensorFlow. Please wait...\n")
import tensorflow as tf
from sklearn.model_selection import train_test_split

EPOCHS = 40
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 6
TEST_SIZE = 0.4

def clear():
    if os.name == 'nt':
        os.system("cls")
    else:
        os.system("clear")

def main():

    images, labels = load_data('dataset')

    labels = tf.keras.utils.to_categorical(labels)
    print("\nSplitting images into training & testing sets")
    try:
        x_train, x_test, y_train, y_test = train_test_split(
            np.array(images), np.array(labels), test_size=TEST_SIZE
        )
    except ValueError:
        print("Directory names are conflicting")
        print("\nCheck your subdirectory names once more")
        print("If you can't solve it, try renaming the directories to simple integer values")
        sys.exit()

    model = get_model()

    print("\nFitting data into neural net")
    model.fit(x_train, y_train, epochs=EPOCHS)

    print("\nTesting neural net\n")
    model.evaluate(x_test,  y_test, verbose=2)

    if len(sys.argv) == 2:
        filename = sys.argv[1]
        model.save(filename)
        print(f"Model saved to {filename}.")

def load_data(data_dir):
    try:
        images = []
        labels = []
        subdirs = os.listdir(data_dir)
        print("Loading Image files for training. Please wait. This might take a while")
        for each in sorted(subdirs):
            image = os.listdir(data_dir+'/'+each)
            for x in image:
                img = cv2.imread(data_dir+'/'+each+'/'+x, cv2.IMREAD_UNCHANGED)
                resized = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
                labels.append(each)
                images.append(resized)
                clear()
                print("Loading the image files")
                print("Please be patient. This might take a while\n")
                print("Loading: "+data_dir+'/'+each+'/'+x)
        clear()
        print("Finished loading")
        return (images, labels)
    except:
        clear()
        print("Error reading image files. Make sure your dataset is structured according the logic")
        print("For more on how to structure your dataset, visit: https://github.com/vishalkrishnads \n")
        print("QUITTING")
        sys.exit()

def get_model():
    print("\nSetting up the neural network")
    model = tf.keras.models.Sequential([
        layers.Conv2D(8, (3,3), input_shape=(IMG_WIDTH, IMG_HEIGHT, 3), padding='same', activation='relu'),
        layers.MaxPooling2D(pool_size=(2,2)),
        layers.Conv2D(16, (3,3), padding='same', activation='relu'),
        layers.MaxPooling2D(pool_size=(2,2)),
        layers.Conv2D(32, (3,3), padding='same', activation='relu'),
        layers.MaxPooling2D(pool_size=(2,2)),
        layers.Conv2D(64, (3,3), padding='same', activation='relu'),
        layers.MaxPooling2D(pool_size=(2,2)),
        layers.Conv2D(128, (3,3), padding='same', activation='relu'),
        layers.Dropout(0.5),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(NUM_CATEGORIES, activation='softmax')
    ])
    clear()
    print("\nBuilding the neural network")
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

if __name__ == "__main__":
    main()