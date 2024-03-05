import cv2

class FaceRecognizer:
    def __init__(self, camera=0, face_cascade_file='haarcascade_frontalface_default.xml', training_file='data/ridwaanhall_training.xml'):
        self.camera = camera
        self.video = cv2.VideoCapture(camera, cv2.CAP_DSHOW)
        self.face_cascade = cv2.CascadeClassifier(face_cascade_file)
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.recognizer.read(training_file)

    def recognize_faces(self):
        a = 0
        while True:
            a += 1
            check, frame = self.video.read()
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray_frame, 1.3, 5)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                id, conf = self.recognizer.predict(gray_frame[y:y+h, x:x+w])
                if id == 1:
                    id = 'ridwaanhall'
                elif id == 2:
                    id = 'uden'
                else:
                    id = 'Lu gak dianggep'
                cv2.putText(frame, str(id), (x+40, y-10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0))
            cv2.imshow("Face Recognition - ridwaanhall", frame)
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
        self.video.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    recognizer = FaceRecognizer()
    recognizer.recognize_faces()
