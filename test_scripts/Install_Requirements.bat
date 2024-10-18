@echo off
REM Verificar si pip está instalado
pip --version
IF %ERRORLEVEL% NEQ 0 (
    echo Pip no está instalado. Por favor, instala pip primero.
    exit /b 1
)

REM Instalar los paquetes desde requirements.txt
pip install -r requirements.txt

REM Verificar si la instalación fue exitosa
IF %ERRORLEVEL% EQU 0 (
    echo Instalación completada exitosamente.
) ELSE (
    echo Hubo un error durante la instalación.
    exit /b 1
)

pause