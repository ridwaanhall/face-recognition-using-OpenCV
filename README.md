# Face Recognition with OpenCV

This project uses OpenCV to detect and recognize faces in an image with Haar Cascade Classifiers and LBPH Face Recognizer.

## Installation

### 1. Install OpenCV (See [Installation](https://opencv.org/install/) for details)

```bash
pip install opencv-python opencv-contrib-python 
```

### 2. Install PIL (See [Installation](https://pillow.readthedocs.io/en/stable/installation.html) for details)

```bash
pip install pillow
```

### 3. Install numpy (See [Installation](https://numpy.org/install/) for details)

```bash
pip install numpy
```

### 4. or install all at once

```bash
pip install opencv-python opencv-contrib-python pillow numpy
```

## Usage

### 1. Clone this repository

```bash
git clone https://github.com/ridwaanhall/face-recognition-with-OpenCV.git
```

### 2. Navigate to the directory

```bash
cd face-recognition
```

### 3. Run the script

run `01_recording.py` to get 30 photo of your face and insert ID (1, 2, 3, etc.)

run `02_training.py` to train the model with your photos using LBPH and Haar Cascade Classifiers

run `03_scanning.py` to scan your face and show the result.

## References

- [OpenCV](https://opencv.org/)
- [Pillow](https://pillow.readthedocs.io/)
- [Numpy](https://numpy.org/)
- [bogotobogo](https://www.bogotobogo.com/python/OpenCV_Python/python_opencv3_Image_Object_Detection_Face_Detection_Haar_Cascade_Classifiers.php)
