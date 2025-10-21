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

# UDP socket configuration (receiver is assumed to be local)
UDP_IP = "127.0.0.1"
UDP_PORT = 5005
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Collect simulation data from shared memory.
# Indicators below are kept for potential on-screen display (currently hidden).

appWindow = 0
carSpeed = 0
rpms = 0
gear = 0
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

class GearIndicator:
    def __init__(self, app, x, y, name):
        self.xPosition = x
        self.yPosition = y
        
        ac.setPosition(ac.addLabel(appWindow, name), x, y)
        self.currentValueLabel = ac.addLabel(appWindow, "0")
        ac.setPosition(self.currentValueLabel, x + 50, y)
        
    def setCurrentValue(self, value):
        ac.setText(self.currentValueLabel, "{0}".format(value))
   
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

    global appWindow, carSpeed, rpms, gear, lapCount, trackPosition, tyresOut, carDamage

    appWindow = ac.newApp(" ")
    #ac.setSize(appWindow, 300, 240)
    ac.setSize(appWindow, 0, 0)
    ac.drawBorder(appWindow, 1)
    ac.setBackgroundOpacity(appWindow, 0)

    # Hide the app logo
    ac.setIconPosition(appWindow, -10000, -10000)

    # Simulation data labels (currently disabled)
    """ carSpeed = SpeedIndicator(appWindow, 20, 40, "Speed:")
    lapCount = LapIndicator(appWindow, 20, 80, "Laps:")
    trackPosition = TrackPositionIndicator(appWindow, 20, 120, "Track Position:")
    tyresOut = TyresOutIndicator(appWindow, 20, 160, "Tyres Out:")
    carDamage = CarDamageIndicator(appWindow, 20, 200, "Car Damage:") """
    
    ac.log("Hello from Python!")
    log_message = "Car Damage: {}, RPMs: {}, Gear: {}, TyresOut: {}, accG: {}".format(info.physics.carDamage[1], info.physics.rpms, info.physics.gear, info.physics.numberOfTyresOut, [info.physics.accG[0], info.physics.accG[1], info.physics.accG[2]])
    ac.log(log_message)
    ac.addRenderCallback(appWindow, onFormRender)
    return " "

def onFormRender(deltaT):
    global carSpeed, rpms, gear, lapCount, trackPosition, tyresOut, carDamage

    # Speed in km/h
    velocidad = ac.getCarState(0, acsys.CS.SpeedKMH)
    #carSpeed.setCurrentValue(velocidad)

    # Car acceleration in Gs (x: left-right, y: up-down, z: forward-back)
    acc_x = info.physics.accG[0]
    acc_y = info.physics.accG[1]
    acc_z = info.physics.accG[2]

    # RPMs
    rpms = ac.getCarState(0, acsys.CS.RPM)

    # Current gear
    gear = info.physics.gear
    #gear.setCurrentValue(gear)

    # Laps completed
    vueltas = ac.getCarState(0, acsys.CS.LapCount)
    #lapCount.setCurrentValue(vueltas)

    # Normalized position on track
    posicion = ac.getCarState(0, acsys.CS.NormalizedSplinePosition)
    #trackPosition.setTrackPositionValue(posicion)

    # Number of tyres out of track
    ruedas_fuera = info.physics.numberOfTyresOut
    #tyresOut.setTyresOutValue(ruedas_fuera)

    # Car damage (vector of 4 values); take the max for display/logging
    if hasattr(info.physics, 'carDamage'):
        damage = info.physics.carDamage        
        try:
            #carDamage.setCarDamageValue(damage)
            damage = "{}_{}_{}_{}".format(damage[0], damage[1], damage[2], damage[3])
            pass
        except Exception as e:
            ac.log("Error while reading car damage: {}".format(e))
            #damage = [0,0,0,0]

    # Build the UDP payload (simple text for ease of parsing)
    message = "Speed: {}, RPMs: {}, Gear: {}, Laps: {}, Track Position: {}, Tyres Out: {}, Car Damage: {}, Acc X: {}".format(
        velocidad, rpms, gear, vueltas, posicion, ruedas_fuera, damage, acc_x
    )

    # Send over UDP
    sock.sendto(message.encode(), (UDP_IP, UDP_PORT))