import gym
import cv2
import numpy as np

class FaceRecognitionRL:
    def __init__(self):
        self.env = gym.make("FaceRecognition-v0")
        self.env.reset()
        self.face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        self.data = []
        self.labels = []

    def capture_faces(self):
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
            for (x, y, w, h) in faces:
                roi_gray = gray[y:y+h, x:x+w]
                self.data.append(roi_gray)
                self.labels.append(0)
                cv2.imshow("Face", roi_gray)
                cv2.rectangle(frame, (x,y), (x+w, y+h), (255, 0, 0), 2)
            cv2.imshow("Frame", frame)
            if cv2.waitKey(20) & 0xFF == ord("q"):
                break
        cap.release()
        cv2.destroyAllWindows()

    def train_model(self):
        self.data = np.array(self.data, dtype="uint8")
        self.labels = np.array(self.labels, dtype="int32")
        model = cv2.face.LBPHFaceRecognizer_create()
        model.train(self.data, self.labels)
        return model

    def run_rl(self, model):
        while True:
            observation = self.env.reset()
            total_reward = 0
            while True:
                gray = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
                if len(faces) == 0:
                    action = 0
                else:
                    (x, y, w, h) = faces[0]
                    roi_gray = gray[y:y+h, x:x+w]
                    label, confidence = model.predict(roi_gray)
                    if confidence < 50:
                        action = 1
                    else:
                        action = 0
                observation, reward, done, info = self.env.step(action)
                total_reward += reward
                if done:
                    break
            print("Total Reward:", total_reward)

if __name__ == "__main__":
    frl = FaceRecognitionRL()
    frl.capture_faces()
    model = frl.train_model()
    frl.run_rl(model
