FROM python:3.9-slim
WORKDIR /app/vtouch_FireDetector
COPY . ./
RUN pip install -r requirements.txt
CMD ["python", "test_firedetector.py"]
