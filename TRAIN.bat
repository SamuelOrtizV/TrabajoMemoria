@echo off
:: Guarda el script y los argumentos en una variable
set script_path=C:\Users\PC\Documents\GitHub\TrabajoMemoria\SAC_Train.py

:: Ejecuta el script como administrador
powershell -Command "Start-Process python -ArgumentList '%script_path% %args%' -Verb RunAs"