# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 09:30:02 2021

@author: bmoor
"""

def writeFlightStreamScript(wing,scene,scriptFile):
    scriptTemplate = open("C:/Users/Bruno-USU/Desktop/Research/FlightStream_Code/scriptTemplate.txt","r")
    scriptFlightStream = open(scriptFile,"w")
    
    for line in scriptTemplate:
        if "simulationTemplate" in line:
            scriptFlightStream.write(scene.simulationTemplate+"\n")
        elif "SET_TRAILING_EDGE_SWEEP_ANGLE angleTE" in line:
            if max(wing.Sweep[0]) < 41:
                scriptFlightStream.write("SET_TRAILING_EDGE_SWEEP_ANGLE "+str(45.0)+"\n")
            else:
                scriptFlightStream.write("SET_TRAILING_EDGE_SWEEP_ANGLE "+str(max(wing.Sweep[0])*1.1)+"\n")
        elif "FILE wingCSVFile" in line:
            scriptFlightStream.write("FILE "+wing.CSVFile+"\n")
        elif "wingTEFile" in line:
            scriptFlightStream.write(wing.TEFile+"\n")
        elif "TREFFTZ trefftzDistance" in line:
            scriptFlightStream.write("TREFFTZ "+str(wing.Trefftz)+"\n")
        elif "ANGLE_OF_ATTACK alpha" in line:
            scriptFlightStream.write("ANGLE_OF_ATTACK "+str(scene.alpha)+"\n")
        elif "FREESTREAM_VELOCITY refVelocity" in line:
            scriptFlightStream.write("FREESTREAM_VELOCITY "+str(scene.V_inf)+"\n")
        elif "REFERENCE_VELOCITY refVelocity" in line:
            scriptFlightStream.write("REFERENCE_VELOCITY "+str(scene.V_inf)+"\n")
        elif "REFERENCE_AREA refArea" in line:
            scriptFlightStream.write("REFERENCE_AREA "+str(wing.S_ref)+"\n")
        elif "REFERENCE_LENGTH refLength" in line:
            scriptFlightStream.write("REFERENCE_LENGTH "+str(wing.MAC)+"\n")
        elif "outputFile" in line:
            scriptFlightStream.write(scene.outputFile+"\n")
        elif "savedRunFile" in line:
            scriptFlightStream.write(scene.savedRunFile+"\n")
        else:
            scriptFlightStream.write(line)

# testWing = wingClass.wing(0.3, 12, 27, '0012')
# testScene = wingClass.scene(5, 10, "outputFile.txt", "savedRun.fsm")