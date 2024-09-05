@echo off
:: Guarda el script y los argumentos en una variable
set script_path=C:\ruta\al\directorio\del\script\DataCollector.py
set args=--width 1920 --height 1080 --full_screen True

:: Ejecuta el script como administrador
powershell -Command "Start-Process python -ArgumentList '%script_path% %args%' -Verb RunAs"