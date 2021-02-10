import numpy as np
import cv2
import tensorflow as tf
import sys
import os

def main():
    cap = cv2.VideoCapture(0)
    try:
        model = tf.keras.models.load_model(sys.argv[1])
        count = 0
        while(True):
            cap.set(cv2.CAP_PROP_POS_MSEC,(count*1000))
            ret, frame = cap.read()
            cv2.namedWindow("Finger Counter", cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty("Finger Counter",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
            cv2.imshow("Finger Counter", frame)
            frame = cv2.resize(frame, (30, 30))
            label = model.predict(
                    [np.array(frame).reshape(1, 30, 30, 3)]
                ).argmax()
                
            if label is not None:
                os.system("cls")
                print("Number: "+str(label))
            else:
                print("Sorry. I'm not old enough to know that. I'm still a baby")
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            else:
                count = count+1
    except KeyboardInterrupt:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("\nPlease specify a saved model to use for predicting. \n")
        print("QUITTING")
        sys.exit()
    else:
        main()