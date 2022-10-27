# vtouch_FireDetector
2022 vtouch FireDetector

test_firedetector.py로 테스트 
1초 정도 간격으로 OD -> IC 두 단계 동작 

- 환경
  - Object detection은 yolov7, Image classification은 yolov5를 기반
  - 윈도우 환경, Python 3.9.12에서 개발
  - pip install -r requirements.txt 먼저 실행 필요

- 추가 파일
  - test_firedetector : fire detector 실행
  - vtouch_firedetector : 메인 fire detector 동작
  - Yolov7 및 v5 소스코드들이 data, models, utils 폴더에 있음 (몇몇 파일 수정됨)
  - weights 폴더에 개발한 od와 ic 모델 pt 파일이 있음
     - od_fire_smoke.pt : yolov7 기반 obeject detection (fire, smoke 검출)
     - ic_default_fire_smoke.pt: yolov5m 기반 image classification(0: default, 1: fire, 2: smoke)
  - vtouch_mec_wsserver_test: 박스를 그리기 위해 코드 약간 수정

- 옵션
  - classify : 두번째 IC 사용 여부 선택
  - draw_box : box 그리기 여부 선택
  - conf_thres : confidence 값
