import cv2

bg = None


def run_avg(image, aWeight):
    global bg
    # initialising the background image
    if bg is None:
        bg = image.copy().astype("float")
        return

    # update background with accumulated weight
    cv2.accumulateWeighted(image, bg, aWeight)



def segment(image, threshold=25):
    global bg
    # finding the absolute difference between background and current frame
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



def main():

    aWeight = 0.5

    video = cv2.VideoCapture(0)

    top,right,bottom,left=10,350,225,590

    num_frames=0
    image_num=0

    start = False

    while(True):
        #Current Frame
        success , frame = video.read()
        if(success == True):
            frame = cv2.flip(frame,1)

            clone = frame.copy()

            height,width = frame.shape[:2]

            #get only desired region
            roi = frame[top:bottom,right:left]

            #convert image
            gray = cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (7,7), 0)
            if(num_frames < 30):
                run_avg(gray,aWeight)
                #print(num_frames)
            else:
                
                hand = segment(gray)

                #if hand is detected in the region
                if hand is not None:
                    thresholded, segmented = hand
                    cv2.drawContours(clone, [segmented + (right, top)], -1,(0, 0, 255))
                    if start:

                        # Mention the directory in which you wanna store the images followed by the image name
                        cv2.imwrite("Dataset/test_set/left/left_" +
                                    str(image_num) + '.png', thresholded)
                        image_num += 1
                    cv2.imshow("Thesholded", thresholded)

            # draw the segmented hand
            cv2.rectangle(clone, (left, top), (right, bottom), (0, 255, 0), 2)

            # increment the number of frames
            num_frames += 1

            # display the frame with segmented hand
            cv2.imshow("Video Feed", clone)

            # observe the keypress by the user
            keypress = cv2.waitKey(1) & 0xFF

            # if the user pressed "q", then stop looping
            if keypress == ord("q") or image_num > 100:
                break

            if keypress == ord("s"):
                start = True

        else:
            print("[Warning!] Error input, Please check your(camera Or video)")
            break
    video.release()


main()

# free up memory

cv2.destroyAllWindows()



