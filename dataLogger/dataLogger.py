import sys
import os.path
import platform
import ac
import acsys


# import libraries
if platform.architecture()[0] == "64bit":
    sysdir=os.path.dirname(__file__)+'/stdlib64'
else:
    sysdir=os.path.dirname(__file__)+'/stdlib'

sys.path.insert(0, sysdir)
os.environ['PATH'] = os.environ['PATH'] + ";."

import ctypes
import socket
from ctypes import *
from sim_info import *

# Configuración del socket
UDP_IP = "127.0.0.1"  # Dirección IP del servidor receptor
UDP_PORT = 5005       # Puerto del servidor receptor
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Reunir datos de la simulación
# Se necesita: Cantidad de vueltas completadas, posición normalizada de la pista, cantidad de ruedas fuera de la pista, daño del auto, velocidad

# Crear indicadores de velocidad, vueltas, posición en la pista, ruedas fuera de la pista y daño del auto

appWindow = 0
carSpeed = 0
rpms = 0
lapCount = 0
trackPosition = 0
tyresOut = 0
carDamage = 0
filter = 0.2

class SpeedIndicator:
    def __init__(self, app, x, y, name):
        self.xPosition = x
        self.yPosition = y
        self.currentValue = 0
        self.oldValue = 0
        
        ac.setPosition(ac.addLabel(appWindow, name), x, y)
        self.currentValueLabel = ac.addLabel(appWindow, "0 km/h")
        ac.setPosition(self.currentValueLabel, x + 50, y)

    def setCurrentValue(self, value):
        global filter
        self.currentValue = self.oldValue * filter + value * (1 - filter)
        self.currentValue = round(self.currentValue)
        ac.setText(self.currentValueLabel, "{0} km/h".format(abs(self.currentValue)))
        self.oldValue = self.currentValue

class LapIndicator:
    def __init__(self, app, x, y, name):
        self.xPosition = x
        self.yPosition = y
        
        ac.setPosition(ac.addLabel(appWindow, name), x, y)
        self.currentValueLabel = ac.addLabel(appWindow, "0")
        ac.setPosition(self.currentValueLabel, x + 50, y)
        
    def setCurrentValue(self, value):
        ac.setText(self.currentValueLabel, "{0}".format(value))

class TrackPositionIndicator:
    def __init__(self, app, x, y, name):
        self.xPosition = x
        self.yPosition = y
        
        ac.setPosition(ac.addLabel(appWindow, name), x, y)
        self.currentValueLabel = ac.addLabel(appWindow, "-")
        ac.setPosition(self.currentValueLabel, x + 100, y)
        
    def setTrackPositionValue(self, value):
        ac.setText(self.currentValueLabel, "{:.2f}".format(value))

class TyresOutIndicator:
    def __init__(self, app, x, y, name):
        self.xPosition = x
        self.yPosition = y
        
        ac.setPosition(ac.addLabel(appWindow, name), x, y)
        self.currentValueLabel = ac.addLabel(appWindow, "-")
        ac.setPosition(self.currentValueLabel, x + 100, y)
        
    def setTyresOutValue(self, value):
        ac.setText(self.currentValueLabel, "{:.2f}".format(value))

class CarDamageIndicator: # Recibe un array de length 5 con los valores de daño de cada parte del auto, retorna el mayor de ellos
    def __init__(self, app, x, y, name):
        self.xPosition = x
        self.yPosition = y
        
        ac.setPosition(ac.addLabel(appWindow, name), x, y)
        self.currentValueLabel = ac.addLabel(appWindow, "-")
        ac.setPosition(self.currentValueLabel, x + 100, y)
        
    def setCarDamageValue(self, value):
        ac.setText(self.currentValueLabel, "{:.2f}".format(max(value)))

def acMain(ac_version):

    global appWindow, carSpeed, rpms, lapCount, trackPosition, tyresOut, carDamage

    appWindow = ac.newApp(" ")
    #ac.setSize(appWindow, 300, 240)
    ac.setSize(appWindow, 0, 0)
    ac.drawBorder(appWindow, 1)
    ac.setBackgroundOpacity(appWindow, 0)

    # Ocultar el logo de la aplicación
    ac.setIconPosition(appWindow, -10000, -10000)

    # Datos de la simulación
    """ carSpeed = SpeedIndicator(appWindow, 20, 40, "Speed:")
    lapCount = LapIndicator(appWindow, 20, 80, "Laps:")
    trackPosition = TrackPositionIndicator(appWindow, 20, 120, "Track Position:")
    tyresOut = TyresOutIndicator(appWindow, 20, 160, "Tyres Out:")
    carDamage = CarDamageIndicator(appWindow, 20, 200, "Car Damage:") """
    
    ac.log("Hello from Python!")
    log_message = "Car Damage: {}, RPMs: {}, TyresOut: {}".format(max(info.physics.carDamage), info.physics.rpms, info.physics.numberOfTyresOut)
    ac.log(log_message)
    ac.addRenderCallback(appWindow, onFormRender)
    return " "

def onFormRender(deltaT):
    global carSpeed, rpms, lapCount, trackPosition, tyresOut, carDamage

    # Obtener la velocidad en km/h
    velocidad = ac.getCarState(0, acsys.CS.SpeedKMH)
    #carSpeed.setCurrentValue(velocidad)

    # Obtener las revoluciones por minuto
    rpms = ac.getCarState(0, acsys.CS.RPM)

    # Obtener el número de vueltas completadas
    vueltas = ac.getCarState(0, acsys.CS.LapCount)
    #lapCount.setCurrentValue(vueltas)

    # Obtener la posición normalizada en la pista
    posicion = ac.getCarState(0, acsys.CS.NormalizedSplinePosition)
    #trackPosition.setTrackPositionValue(posicion)

    # Obtener la cantidad de ruedas fuera de la pista
    ruedas_fuera = info.physics.numberOfTyresOut
    #tyresOut.setTyresOutValue(ruedas_fuera)

    # Obtener el daño del auto    
    if hasattr(info.physics, 'carDamage'):
        damage = info.physics.carDamage
        try:
            #carDamage.setCarDamageValue(damage)
            pass
        except Exception as e:
            ac.log("Error al obtener el daño del auto: {}".format(e))

    # Crear el mensaje con los datos
    message = "Speed: {}, RPMs: {}, Laps: {}, Track Position: {}, Tyres Out: {}, Car Damage: {}".format(
        velocidad, rpms, vueltas, posicion, ruedas_fuera, max(damage)
    )

    # Enviar el mensaje a través del socket
    sock.sendto(message.encode(), (UDP_IP, UDP_PORT))