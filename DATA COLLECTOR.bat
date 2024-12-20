@echo off
:: Guarda el script y los argumentos en una variable
set script_path=%~dp0DataCollector.py
:: set args=--width 1920 --height 1080 --full_screen True

:: Ejecuta el script como administrador
:: powershell -Command "Start-Process python -ArgumentList '%script_path% %args%' -Verb RunAs"
powershell -Command "Start-Process python -ArgumentList '%script_path%' -Verb RunAs"

:: Verifica el código de salida
if %ERRORLEVEL% neq 0 (
    echo Ocurrió un error. Presione cualquier tecla para continuar...
    pause > nul
)