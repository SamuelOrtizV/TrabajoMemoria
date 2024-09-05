@echo off
:: Guarda el script en una variable
set script_path=C:\Users\PC\Documents\GitHub\TrabajoMemoria\getkeys2.py

:: Ejecuta el script como administrador
powershell -Command "Start-Process python -ArgumentList '%script_path%' -Verb RunAs"