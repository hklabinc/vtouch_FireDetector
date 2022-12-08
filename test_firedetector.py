from urllib.parse import _ResultMixinStr
import cv2
import base64

from vtouch_mec_ai_data import CameraId, DetectionBox, VTouchLabel, VTouchMecAiData

from vtouch_mec_comm import VTouchMecComm

import time
import torch
from vtouch_firedetector import VTouchFireDetector

import platform, pathlib
if platform.system() == 'Linux':        # For Linux
    pathlib.WindowsPath = pathlib.PosixPath

# Set source
# url = 'rtsp://'
url = 0
cap = cv2.VideoCapture(url)

comm = VTouchMecComm()

# Setup FireDetector
weights = 'weights/od_fire_smoke.pt'    # Yolov7 Detection model
weights_c = 'weights/ic_default_fire_smoke.pt'    # Yolov5 Classify model
fd = VTouchFireDetector(weights, weights_c, classify=True)      # Set classify=True if want to use second-stage classification

past = time.time()
while True :
    ret, frame_iamge = cap.read()
    cv2.waitKey(1)
    
    now = time.time()        
    if now - past > 0.1:      # every 0.1 second
        past = now
        
        with torch.no_grad():       
            result, frame_det = fd.detect(frame_iamge, conf_thres=0.25, draw_box=True)     # Inference
              
        cv2.imshow("Video_detected", frame_det)
    
        if len(result) > 0:     # Only if detected, send to server
            ret, jpg_image = cv2.imencode('.jpg', frame_det)
            base64_image = base64.b64encode(jpg_image)

            data = VTouchMecAiData(CameraId.GUNPOWDER_HOUSE, str(base64_image, 'utf-8'), result)
            comm.send(data.toJson())
        