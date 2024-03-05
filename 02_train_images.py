import cv2
import os
import numpy as np
from PIL import Image

'''
train 30 images for each user and save the model to data/ridwaanhall_training.xml
'''

class FaceTrainer:
    def __init__(self, cascade_file="haarcascade_frontalface_default.xml"):
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.detector = cv2.CascadeClassifier(cascade_file)

    def get_images_with_labels(self, path):
        image_paths = [os.path.join(path, f) for f in os.listdir(path)]
        face_samples = []
        ids = []
        for image_path in image_paths:
            pil_image = Image.open(image_path).convert('L')
            image_np = np.array(pil_image, 'uint8')
            identity = int(os.path.split(image_path)[-1].split(".")[1])
            faces = self.detector.detectMultiScale(image_np)
            for (x, y, w, h) in faces:
                face_samples.append(image_np[y:y+h, x:x+w])
                ids.append(identity)
        return face_samples, ids

    def train_and_save(self, data_path, output_path):
        faces, ids = self.get_images_with_labels(data_path)
        self.recognizer.train(faces, np.array(ids))
        self.recognizer.save(output_path)

if __name__ == "__main__":
    trainer = FaceTrainer()
    trainer.train_and_save('data', 'data/ridwaanhall_training.xml')
