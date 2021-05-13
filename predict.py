import numpy as np
import tensorflow as tf
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import cv2

bg=None

def main():
    # initialize weight for running average
    aWeight = 0.5

    # get the reference to the webcam
    camera = cv2.VideoCapture(0)

    # region of interest (ROI) coordinates
    top, right, bottom, left = 10, 350, 225, 590

    # initialize num of frames
    num_frames = 0
    start_recording = True

    # keep looping, until interrupted
    while(True):
        # get the current frame
        (grabbed, frame) = camera.read()

        # resize the frame
#        frame = imutils.resize(frame, width = 700)

        # flip the frame so that it is not the mirror view
        frame = cv2.flip(frame, 1)

        # clone the frame
        clone = frame.copy()

        # get the height and width of the frame
        (height, width) = frame.shape[:2]

        # get the ROI
        roi = frame[top:bottom, right:left]

        # convert the roi to grayscale and blur it
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        # to get the background, keep looking till a threshold is reached
        # so that our running average model gets calibrated
        if num_frames < 30:
            run_avg(gray, aWeight)
        else:
            # segment the hand region
            hand = segment(gray)

            # check whether hand region is segmented
            if hand is not None:
                # if hand is detected, unpack the thresholded image and
                # segmented region
                (thresholded, segmented) = hand

                # draw the segmented region and display the frame
                cv2.drawContours(clone, [segmented + (right, top)], -1, (0, 0, 255))
                if start_recording:
#                    gray_image = cv2.cvtColor(thresholded, cv2.COLOR_BGR2GRAY)
                    cv2.imwrite('Temp.png', thresholded)
                    predictedClass = getPredictedClass()
                    print(predictedClass)
                    print(1)
                    showStatistics(predictedClass,clone)
                cv2.imshow("Thesholded", thresholded)

        # draw the segmented hand
        cv2.rectangle(clone, (left, top), (right, bottom), (0,255,0), 2)

        # increment the number of frames
        num_frames += 1

        # display the frame with segmented hand
        cv2.imshow("Video Feed", clone)

        # observe the keypress by the user
        keypress = cv2.waitKey(1) & 0xFF

        # if the user pressed "q", then stop looping
        if keypress == ord("q"):
            cv2.destroyAllWindows()        
            break
        
        #if keypress == ord("s"):
         #   start_recording = True
    

def run_avg(image, aWeight):
    global bg
    # initialize the background
    if bg is None:
        bg = image.copy().astype("float")
        return

    # compute weighted average, accumulate it and update the background
    cv2.accumulateWeighted(image, bg, aWeight)

def segment(image, threshold=25):
    global bg
    # find the absolute difference between background and current frame
    diff = cv2.absdiff(bg.astype("uint8"), image)

    # threshold the diff image so that we get the foreground
    thresholded = cv2.threshold(diff,
                                threshold,
                                255,
                                cv2.THRESH_BINARY)[1]

    # get the contours in the thresholded image
    (cnts, _) = cv2.findContours(thresholded.copy(),
                                    cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)

    # return None, if no contours detected
    if len(cnts) == 0:
        return
    else:
        # based on contour area, get the maximum contour which is the hand
        segmented = max(cnts, key=cv2.contourArea)
        return (thresholded, segmented)


def getPredictedClass():
    # Predict
    
    test_image = image.load_img('temp.png',target_size=(64,64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image,axis=0)
    prediction = cnn.predict(test_image)
    print (1)
    return np.argmax(prediction)

def showStatistics(predictedClass,clone):

    className = ""
    dis={'fist': 0, 'palm': 1, 'swing': 2}
    #    if predictedClass == 0:
    #        className = "Swing"
    #    elif predictedClass == 1:
    #        className = "Palm"
    #    elif predictedClass == 2:
    #        className = "Fist"
    for key,val in dis.items():
        if val == predictedClass:
            className=key
            break

    cv2.putText(clone,"Predicted Class: " + className,(30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)


cnn  = tf.keras.models.load_model('model.h5')
main()
