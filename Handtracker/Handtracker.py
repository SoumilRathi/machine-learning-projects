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
    
    if results.multi_hand_landmarks:
        for handLmks in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, handLmks, hand.HAND_CONNECTIONS)
            
    if cv2.waitKey(25) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            print('Exited')
            break
    cTime  = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 255, 255), 2,  )
    cv2.imshow('Image', img)
    cv2.waitKey(1)
