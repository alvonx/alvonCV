import cv2
import mediapipe as mp
import time


class FaceMeshDetector:
    def __init__(self, staticMode=False, maxFaces=5, minDetectionConf=0.5, minTrackConf=0.5):
        self.staticMode = staticMode
        self.maxFaces = maxFaces
        self.minDetectionConf = minDetectionConf
        self.minTrackConf = minTrackConf

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.facemesh = self.mpFaceMesh.FaceMesh(self.staticMode, self.maxFaces,
                                                 self.minDetectionConf, self.minTrackConf)
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=1)

    def findFaceMesh(self, img, draw=True):
        self.imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.facemesh.process(self.imgRGB)
        faces = []
        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMesh.FACE_CONNECTIONS, self.drawSpec, self.drawSpec)

                face = []
                for id, lm in enumerate(faceLms.landmark):
                    ih, iw, ic = img.shape
                    x, y = int(lm.x*iw), int(lm.y*ih)
                    face.append([x, y])
                faces.append(face)

        return img


def main():
    cap = cv2.VideoCapture(0)
    pTime = 0
    detector = FaceMeshDetector()
    while True:
        success, img = cap.read()

        img, faces = detector.findFaceMesh(img)
        if len(faces):
            print(len(faces))

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
        cv2.imshow('Image', img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
