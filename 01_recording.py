import cv2

class FaceRecognition:
    def __init__(self, camera=0, cascade_file='haarcascade_frontalface_default.xml'):
        self.camera = camera
        self.video = cv2.VideoCapture(camera, cv2.CAP_DSHOW)
        self.face_cascade = cv2.CascadeClassifier(cascade_file)

    def capture_faces(self, user_id, max_images=30):
        count = 0
        while True:
            count += 1
            check, frame = self.video.read()
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray_frame, 1.3, 5)
            print(faces)
            for (x, y, w, h) in faces:
                cv2.imwrite(f'data/user.{user_id}.{count}.jpg', gray_frame[y:y+h, x:x+w])
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.imshow("Face Recognition Window", frame)
            if count >= max_images:
                break
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self.video.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    camera_index = 0
    user_id = input('Id: ')
    recognizer = FaceRecognition(camera_index)
    recognizer.capture_faces(user_id)
