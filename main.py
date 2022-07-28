import itertools
import numpy as np
import cv2
import mediapipe as mp
from ketboard import motions, starter
import time

color= (0,0, 255)
base = []
RECORDS =[]
side = ""
start = 0
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh


drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
cap = cv2.VideoCapture(0)
currentFrame = 0

vid_cod = cv2.VideoWriter_fourcc(*'XVID')
output = cv2.VideoWriter(f"cam_video{time.time()}.mp4", vid_cod, 20.0, (640,480))

with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:
    while cap.isOpened():
        success, image = cap.read()
        im_height, im_width, _ = image.shape

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image)

        # Draw the face mesh annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_face_landmarks:
            POSITIONS = mp_face_mesh.FACEMESH_IRISES
            POSITIONS_LIST = list(itertools.chain(*POSITIONS))
            LEFT_EYE = mp_face_mesh.FACEMESH_LEFT_EYE
            LEFT_EYE_POS = list(itertools.chain(*LEFT_EYE))
            for face_landmarks in results.multi_face_landmarks:
                for index in POSITIONS_LIST[:4]:
                    X = int(face_landmarks.landmark[index].x * im_width)
                    Y = int(face_landmarks.landmark[index].y * im_height)
                    base.append([X,Y])
                    #cv2.circle(image, (X,Y), 1, (0,255,0),1)
                for n, index in enumerate(LEFT_EYE_POS):
                    if n in [9,10,13,15]: #10 <==> 15, upper13 => 8 lower
                        X = int(face_landmarks.landmark[index].x * im_width)
                        Y = int(face_landmarks.landmark[index].y * im_height)
                        RECORDS.append([X,Y])
                        # cv2.circle(image, (X, Y), 1, (0, 0,255), 1)
                        # cv2.putText(image, str(n), (X,Y), 1,1,(0,0,255),1)
            X1,Y1 = base[1]
            X2,Y2 = base[3]

            if side == "neutral":
                color=(0,0,255)
            elif side =="left":
                color = (0,255,0)
            elif side =="right":
                color = (0,255,0)
            midpoint_eye =((X1+X2)//2, (Y1+Y2)//2)
            midpoint_center = ((RECORDS[0][0]+RECORDS[2][0])//2, (RECORDS[0][1]+RECORDS[2][1])//2)
            cv2.circle(image, midpoint_eye, 1,color, 2)
            image = cv2.flip(image, 1)
            if side == 'left':
                cv2.putText(image, "Left", (20, 100), 3, 3, color, 5)
            elif side =='right':
                cv2.putText(image, "Right", (350, 100), 3, 3, color, 5)


            #cv2.circle(image, midpoint_center, 1, (0, 255, 0), 1)

            left_midpoint = ((RECORDS[1][0]+3*midpoint_center[0])//4, (RECORDS[1][1]+2*midpoint_center[1])//3)
            right_midpoint = ((RECORDS[3][0] +2* midpoint_center[0]) // 3, (RECORDS[3][1] + 2*midpoint_center[1]) // 3)
            upper_midpoint = ((RECORDS[2][0] + 2*midpoint_center[0]) // 3, (RECORDS[2][1] + 2*midpoint_center[1]) // 3)
            down_midpoint = ((RECORDS[0][0] + 2*midpoint_center[0]) // 3, (RECORDS[0][1] + 2*midpoint_center[1]) // 3)
            base.clear()
            RECORDS.clear()
            X,Y = midpoint_eye[0], midpoint_eye[1]
            if X >= left_midpoint[0]:
                side = "left"
            elif X <= right_midpoint[0]:
                side = "right"
            else:
                side="neutral"
            motions(side)
        if start ==0:
            starter()
            start+=1
        #output.write(image)
        cv2.imshow('MediaPipe Face Mesh', image)

        cv2.waitKey(1)

cap.release()