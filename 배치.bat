@echo on
START /d "C:\Program Files\Mosquitto\" mosquitto -c mosquitto.conf -v
START /d %cd%\������\ node route

timeout 1

START python cam.py

START python fileopenBboxYOLO.py
START python fileopenMakePatch.py
START python fileopenTrain.py

