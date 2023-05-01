import svg.path
import math

def generate_gcode_from_svg(svg_file, output_width):
    # with open(svg_file, "r") as file:
    #     svg_data = svg_file
    svg_data = svg_file
    print(svg_data)
    print('here')
    # write a function to sort the svg paths!  
    

    paths, attributes = svg.path.parse_path(svg_data)
    scale_factor = output_width / attributes["width"]
    print(paths)
    print(attributes)
    gcode = "G28\n"  # Home all axes
    gcode += "G90\n"  # Use absolute coordinates

    for path in paths:
        gcode += "G0 X{} Y{}\n".format(path.start.real * scale_factor, path.start.imag * scale_factor)  # Move to start point

        for segment in path:
            if isinstance(segment, svg.path.Line):
                x, y = segment.end.real * scale_factor, segment.end.imag * scale_factor
                gcode += "G1 X{} Y{}\n".format(x, y)  # Draw a line to the end point
            elif isinstance(segment, svg.path.Arc):
                cx, cy = segment.center.real * scale_factor, segment.center.imag * scale_factor
                rx, ry = segment.radius.real * scale_factor, segment.radius.imag * scale_factor
                angle1 = math.degrees(segment.start_angle)
                angle2 = math.degrees(segment.end_angle)
                large_arc_flag = int(segment.arc)
                sweep_flag = int(segment.sweep)

                gcode += "G2 X{} Y{} I{} J{} F{}\n".format(
                    segment.end.real * scale_factor,
                    segment.end.imag * scale_factor,
                    cx - path.end.real * scale_factor,
                    cy - path.end.imag * scale_factor,
                    max(rx, ry),
                )  # Draw an arc to the end point

    return gcode
