import cv2
import mediapipe as mp
import time

# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
cap = cv2.VideoCapture(0)

hand = mp.solutions.hands
hands = hand.Hands(static_image_mode = True)
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    landMarkList = []
    handNumber = 0
    if results.multi_handedness:
        label = results.multi_handedness[handNumber].classification[0].label  # label gives if hand is left or right
        #account for inversion in webcams
        if label == "Left":
            label = "Right"
        elif label == "Right":
            label = "Left"
    if results.multi_hand_landmarks:
        handAll = results.multi_hand_landmarks
        handMulti = results.multi_hand_landmarks[handNumber]
        for id, landMark in enumerate(handMulti.landmark):
            imgH, imgW, imgC = img.shape  # height, width, channel for image
            xPos, yPos = int(landMark.x * imgW), int(landMark.y * imgH)
            landMarkList.append([id, xPos, yPos, label])

        for lmrk in handAll:
            mpDraw.draw_landmarks(img, lmrk, hand.HAND_CONNECTIONS)
   
    count = 0
    if(len(landMarkList) != 0):
        #Getting y-coordinate of tip of each finger and checking if its greater than y-coordinate of base. Exception for thumb, which is horizontal. 

        if landMarkList[4][3] == "Right" and landMarkList[4][1] > landMarkList[3][1]:       #Right Thumb
            count = count+1
        elif landMarkList[4][3] == "Left" and landMarkList[4][1] < landMarkList[3][1]:       #Left Thumb
            count = count+1
        if landMarkList[8][2] < landMarkList[6][2]:       #Index finger
            count = count+1
        if landMarkList[12][2] < landMarkList[10][2]:     #Middle finger
            count = count+1
        if landMarkList[16][2] < landMarkList[14][2]:     #Ring finger
            count = count+1
        if landMarkList[20][2] < landMarkList[18][2]:     #Little finger
            count = count+1

    cv2.putText(img, str(count), (45, 375), cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 0, 0), 25)
    cv2.imshow("Volume", img)
    cv2.waitKey(1)
            
    if cv2.waitKey(25) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            print('Exited')
            break
    cTime  = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    