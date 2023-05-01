#!/usr/bin/python3


import sys
import math

file_name = sys.argv[1]
new_file = file_name
hasHoming = False  # Set True to enable homing
upscale = True  # set True to multiply x and y values with factor
factor = 2  # scale factor

pump_distance = 400  # set to 0 to disable; millimeters between pumping action
# line_number = 0
drawing_length = 0.0
drawing_x1 = 0.0
drawing_y1 = 0.0
drawing_x2 = 0.0
drawing_y2 = 0.0
Z = 0.0
Y = 0.0
X = 0.0
first_number = False
drawing = False
forcingLines = False
ignoringForcingLines = False
deleteNextLine = False
usedTool = "none"

with open(file_name, "r+") as f:

    if hasHoming == False:
        sheetZero = "X0 Y0"  # distance from starting zero to 0,0 of paper, needs to be in negative numbers
        # homing = "G92 " + sheetZero + "\n"
        homing = ""

    else:
        sheetZero = "X0 Y0"  # distance from homing zero to 0,0 of paper, needs to be in negative numbers
        homing = (
            "G28 X0 Y0\nG92 " + sheetZero + "\n"
        )  # Add second ZAxis to homing #needs to be tested
    new_code = (
        "G90\nG21\nM201 X200 Y200\n" + homing
    )  # G90..absolute Coords; G21..metric;M201..acceleration
    content = f.readlines()  # gcode as a list where each element is a line

    for line in content:
        #print(line)
        if(deleteNextLine):
            deleteNextLine = False
            continue
        if line.startswith(';HEIGHT:0.4'):
            deleteNextLine = True
            continue

        if line.startswith(";forcelines"):
            forcingLines = True
            
        if line.startswith(";/forcelines"):
            forcingLines = False

        if line.startswith(';Tool_'):
            new_code += f';CureentTool: {usedTool}'
            newCode = line.strip(';')
            new_code += f';newTool: {newCode}'
            if usedTool != 'none' and usedTool != line.strip(';'):
                new_code += f';WRONG TOOL: {line}'
                ignoringForcingLines = True
                continue
            usedTool = line.strip(';')
            new_code += f';Tool: {usedTool}'

        if line.startswith(';/Tool_'):
            endTool = line.split('/')[-1]
            if(usedTool == endTool):
                usedTool = 'none'
            ignoringForcingLines = False
        
        if(forcingLines):
            if(ignoringForcingLines == True):
                new_code += f';SKIPPED: {line}'
                continue
            new_code += line
            continue

        

        pref_list = [
            "G0 ",
            "G1 ",
            "G2 ",
            "G3 ",
        ]  # considered beginnings of line
        if line.startswith(tuple(pref_list)):  #
            contentMove = line.strip(
                "/n"
            ).split()  # Array of line with each axis as one element

            if pump_distance != 0:
                for element in contentMove:
                    if element.startswith("Z"):
                        Z = float(element.strip("Z"))
                    if element.startswith("Y"):
                        Y = float(element.strip("Y"))
                    if element.startswith("X"):
                        X = float(element.strip("X"))
                if Z == 0.4:
                    # print('drawing true')
                    drawing = True
                if Z != 0.4:
                    drawing = False
                    # print('drawing false')
                if drawing == True:
                    if first_number == True:  # first number
                        drawing_y1 = Y
                        # print('drawing_y1', drawing_y1)
                        drawing_x1 = X
                        # print('drawing_x1', drawing_x1)
                        first_number = False
                    else:
                        drawing_y2 = Y
                        # print('drawing_y2', drawing_y2)
                        drawing_x2 = X
                        # print('drawing_x2', drawing_x2)
                        first_number = True
                    drawing_length += math.sqrt(
                        pow((drawing_x2*factor - drawing_x1*factor), 2)
                        + pow((drawing_y2*factor - drawing_y1*factor), 2)
                    )
                    # print('x distance', drawing_x2 - drawing_x1)
                    # print('y distance', drawing_y2 - drawing_y1)
                    # print('powerx2x1', math.sqrt(pow((drawing_x2 - drawing_x1),2)+pow((drawing_x2 - drawing_x1),2)))
                    # print('drawing_length',drawing_length)
                if drawing_length >= pump_distance:
                    drawing_length = 0
                    newLine = "; Pumpen"
                    new_code += newLine + "\n"
                    # newLine = 'G1 Z0'
                    # new_code += newLine + '\n'
                    newLine = "G91 ; Pumpen"
                    new_code += newLine + "\n"
                    newLine = "G1 U11"
                    new_code += newLine + "\n"
                    newLine = "G1 U-11"
                    new_code += newLine + "\n"
                    newLine = "G90 ; Pumpen"
                    new_code += newLine + "\n"
                    # newLine = 'G1 Z10'
                    # new_code += newLine + '\n'

            newLine = ""
            for element in contentMove:
                if (
                    "E" not in element
                ):  # and 'Z' not in element:      #use everthing but ExtruderMoves and Z Axis
                    if upscale == True:
                        if element.startswith("X"):
                            element = "X" + str(
                                float(element.strip("X")) * factor
                            )  # multiply value with upscaleFactor and convert back to string with axis
                        if element.startswith("Y"):
                            element = "Y" + str(float(element.strip("Y")) * factor)
                    newLine += element + " "

            new_code += newLine + "\n"
        # else:
        #     new_code += line

with open(new_file, "w") as nf:
    nf.seek(0)
    nf.write(new_code)
    # print(new_code)
