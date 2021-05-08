# Computer Vision Helper Packages

---

- Face Detection
- Face Mesh

## Install the package
```sh
pip install alvonCV
```

## Demo Code for Face Detection

```c
import alvonCV
import cv2
import time

cap = cv2.VideoCapture(0)
pTime = 0
# here you use the alvonCV package
faceDetectorObj = alvonCV.FaceDetector()

while True:
    success, img = cap.read()
    img, bboxs = faceDetectorObj.findFaces(img, draw=True)
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)
    cv2.imshow("Image", img)
    cv2.waitKey(1)
```

## Demo Code for Face Mesh

```c
import alvonCV
import cv2
import time

cap = cv2.VideoCapture(0)
pTime = 0
faceDetectorObj = alvonCV.FaceMeshDetector()

while True:
    success, img = cap.read()
    img = faceDetectorObj.findFaceMesh(img, draw=True)
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)
    cv2.imshow("Image", img)
    cv2.waitKey(1)

```

## Demo Code for Hand Detector

```c
import alvonCV
import cv2
import time

cap = cv2.VideoCapture(0)
detector = alvonCV.HandDetector(detectionCon=0.8, maxHands=1)
while True:
    # Get image frame
    success, img = cap.read()
    
    # Find the hand and its landmarks
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)
    fingersUp = detector.fingersUp()
    print(fingersUp)

    # Display
    cv2.imshow("Image", img)
    cv2.waitKey(1)
```