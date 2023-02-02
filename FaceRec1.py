import cv2
import numpy as np

class FaceRecognitionRL:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.label_dict = {}
        self.current_id = 0

    def train_model(self, frames, labels):
        self.recognizer.train(frames, np.array(labels))

    def predict(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            label, confidence = self.recognizer.predict(roi_gray)
            return self.label_dict[label], confidence

    def add_face(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            self.label_dict[self.current_id] = roi_gray
            self.current_id += 1

    def train_from_camera(self):
        cap = cv2.VideoCapture(0)
        frames = []
        labels = []
        while True:
            ret, frame = cap.read()
            cv2.imshow("Training", frame)
            key = cv2.waitKey(1)
            if key == ord("q"):
                break
            elif key == ord("a"):
                self.add_face(frame)
            elif key == ord("t"):
                self.train_model(frames, labels)
        cap.release()
        cv2.destroyAllWindows()
