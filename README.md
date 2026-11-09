# TrackVision AI



## Real-Time Object Detection and Tracking Using YOLO and Deep SORT



TrackVision AI is a Python-based computer vision project that detects and tracks objects using YOLOv8 and Deep SORT.



## Features



- Real-time object detection using YOLOv8

- Object tracking using Deep SORT

- Unique tracking IDs for detected objects

- Bounding boxes around detected objects

- Webcam input support

- Video file input support

- Saves processed video output



## Technologies Used



- Python

- YOLOv8

- Deep SORT

- OpenCV

- PyTorch

- NumPy



## Project Structure



```text

TrackVision AI/

|-- videos/

|-- output/

|-- main.py

|-- detector.py

|-- tracker.py

|-- requirements.txt

|-- README.md

|-- .gitignore

```



## Installation



1. Create a virtual environment:



   `python -m venv venv`



2. Activate it:



   `.\venv\Scripts\Activate.ps1`



3. Install dependencies:



   `pip install -r requirements.txt`



## Run the Project



Run:



`python main.py`



Follow the terminal instructions to select your input source.



Press **Q** to stop tracking.



## Model



The project uses the pretrained YOLOv8 Nano model (`yolov8n.pt`).



## Future Improvements



- Object counting

- Tracking statistics

- Interactive dashboard

- Improved tracking performance



## Project Status



Under development.


