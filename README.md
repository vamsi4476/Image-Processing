# Face Detection and Image Filters with OpenCV

This project demonstrates real-time face and eye detection using OpenCV, along with various image filters. It uses `haarcascade_frontalface_default.xml` and `haarcascade_eye.xml` for detecting faces and eyes, and applies multiple filters, such as grayscale, thresholding, skin detection (HSV-based), sharpening, blurring, edge detection, and more.

## Features

- **Face and Eye Detection**: Detects faces and eyes in real-time from video input using Haar cascades.
- **Multiple Filters**:
  - **Grayscale**: Converts the image to grayscale.
  - **Threshold**: Applies a binary threshold filter.
  - **Skin Detection**: Filters out skin regions using HSV color space.
  - **Sharpening**: Enhances image sharpness.
  - **Blurring**: Adds a Gaussian blur to the image.
  - **Edge Detection**: Applies Sobel edge detection.

## Demo

A video feed is captured using the default camera (or an external video feed), and each filter can be applied by pressing the corresponding key. Press `Q` to quit the application.

## Installation and Usage

git clone https://github.com/yourusername/your-repo-name.git
pip install opencv-python numpy matplotlib
python face_detection_filters.py

- Ensure `haarcascade_frontalface_default.xml` and `haarcascade_eye.xml` are in the appropriate directory (adjust the path in the code if necessary).
- Press the following keys to switch between filters:
  - `P` - Preview (no filter)
  - `G` - Grayscale
  - `T` - Threshold
  - `H` - HSV Skin Detection
  - `S` - Sharpening
  - `B` - Blur
  - `E` - Edge Detection
  - `F` - Face Recognition
  - `Q` - Quit the application

## Code Structure

- **Threshold Filter**: Binary thresholding with a manually set pixel threshold.
- **HSV Skin Filter**: Detects skin regions by applying a mask on the HSV color space.
- **Sharpening**: Enhances details using a custom kernel.
- **Edge Detection**: Sobel-based edge detection to identify edges in grayscale images.
- **Face and Eye Detection**: Uses Haar cascades for detecting faces and eyes in each frame.

## Code Example

Below is a snippet from the code to demonstrate face detection:

def face_recognition_filter(frame):
    face_cascade = cv2.CascadeClassifier('path/to/haarcascade_frontalface_default.xml')
    eyes_cascade = cv2.CascadeClassifier('path/to/haarcascade_eye.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)

    for (x, y, w, h) in faces:
        center = (x + w // 2, y + h // 2)
        frame = cv2.circle(frame, center, int(0.5 * (w + h)), (255, 0, 0), 2)
        faceROI = gray[y:y + h, x:x + w]
        eyes = eyes_cascade.detectMultiScale(faceROI)
        for (x2, y2, w2, h2) in eyes:
            eye_center = (x + x2 + w2 // 2, y + y2 + h2 // 2)
            frame = cv2.circle(frame, eye_center, int(0.25 * (w2 + h2)), (0, 0, 255), 2)

    return frame

## Requirements

- Python 3.x
- OpenCV
- NumPy
- Matplotlib (for additional visualizations)

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- OpenCV documentation for guidance on Haar cascades and image filters.
- Python libraries: [OpenCV](https://opencv.org/), [NumPy](https://numpy.org/), and [Matplotlib](https://matplotlib.org/).

## Contact

For questions, feel free to reach out at [gundavamsi01@gmail.com](mailto:gundavamsi01@gmail.com).
