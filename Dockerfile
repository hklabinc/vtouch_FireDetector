FROM python:3.9-slim
WORKDIR /app/vtouch_FireDetector
COPY . ./
RUN pip install -r requirements.txt
RUN apt-get update
RUN apt-get -y install libgl1-mesa-glx
RUN apt-get install libglib2.0-0
CMD ["python", "test_firedetector.py"]
