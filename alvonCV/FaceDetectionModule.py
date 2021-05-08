import cv2
import time
import mediapipe as mp


class FaceDetector:
    def __init__(self, minDetectionConf=0.5):
        self.minDetectionConf = minDetectionConf
        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection(min_detection_confidence=self.minDetectionConf)

    def findFaces(self, img, draw=False):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(imgRGB)
        bboxs = []
        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                       int(bboxC.width * iw), int(bboxC.height * ih)
                bboxs.append([id, bbox, detection.score])

                right_eye = (detection.location_data.relative_keypoints[0].x * iw, detection.location_data.relative_keypoints[0].y*ih)
                if draw:
                    self.fancyBBoxDraw(img, bbox)
                    cv2.circle(img, (int(right_eye[0]), int(right_eye[1])), 1, (0, 0, 255), 1)
                    cv2.putText(img, f'Score: {int(detection.score[0] * 100)}', (bbox[0], bbox[1] - 20),
                                cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

        return img, bboxs

    def fancyBBoxDraw(self, img, bbox, l=20, t=6):
        x, y, w, h = bbox
        x1, y1 = x + w, y + h

        cv2.rectangle(img, bbox, (255, 0, 255), 2)

        # top left x,y
        cv2.line(img, (x, y), (x + l, y), (255, 0, 255), t)
        cv2.line(img, (x, y), (x, y + l), (255, 0, 255), t)

        # top right x1,y
        cv2.line(img, (x1, y), (x1 - l, y), (255, 0, 255), t)
        cv2.line(img, (x1, y), (x1, y + l), (255, 0, 255), t)

        # bottom left x,y1
        cv2.line(img, (x, y1), (x + l, y1), (255, 0, 255), t)
        cv2.line(img, (x, y1), (x, y1 - l), (255, 0, 255), t)

        # bottom right x1,y1
        cv2.line(img, (x1, y1), (x1 - l, y1), (255, 0, 255), t)
        cv2.line(img, (x1, y1), (x1, y1 - l), (255, 0, 255), t)

        return img


def main():
    cap = cv2.VideoCapture(0)
    pTime = 0
    faceDetectorObj = FaceDetector()

    while True:
        success, img = cap.read()
        img, bboxs = faceDetectorObj.findFaces(img, draw=True)
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)
        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
