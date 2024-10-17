import ezdxf
import math

class PolygonGenerator:
    @staticmethod
    def rectangle(msp, width, length, LayDose_dict={"layer": "1"}):
        return msp.add_lwpolyline([(-width/2, -length/2), (width/2, -length/2), (width/2, length/2), (-width/2, length/2), (-width/2, -length/2)], dxfattribs=LayDose_dict)

    @staticmethod
    def square(msp, width, LayDose_dict={"layer": "1"}):
        side_length = width
        return msp.add_lwpolyline([(-side_length/2, -side_length/2), (side_length/2, -side_length/2), (side_length/2, side_length/2), (-side_length/2, side_length/2), (-side_length/2, -side_length/2)], dxfattribs=LayDose_dict)

    @staticmethod
    def circle(msp, width, LayDose_dict={"layer": "1"}):
        radius = width / 2
        return msp.add_circle((0, 0), radius, dxfattribs=LayDose_dict)

    @staticmethod
    def octagon(msp, radius, LayDose_dict={"layer": "1"}):
        points = []
        for i in range(8):
            angle = i * math.pi / 4
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            points.append((x, y))
        points.append(points[0])  # Close the polygon
        return msp.add_lwpolyline(points, dxfattribs=LayDose_dict)

    @staticmethod
    def line(msp, width, length, LayDose_dict={"layer": "1"}):
        return msp.add_line((0, -length/2), (0, length/2), dxfattribs=LayDose_dict)

    @staticmethod
    def diamond(msp, width, length, LayDose_dict={"layer": "1"}):
        points = [(-width/2, 0), (0, length/2), (width/2, 0), (0, -length/2), (-width/2, 0)]
        return msp.add_lwpolyline(points, dxfattribs=LayDose_dict)

    @staticmethod
    def cross(msp, width, length, LayDose_dict={"layer": "1"}):
        points = [(-width/2, length/2), (width/2, length/2), (width/2, width/2), (length/2, width/2),
                  (length/2, -width/2), (width/2, -width/2), (width/2, -length/2), (-width/2, -length/2),
                  (-width/2, -width/2), (-length/2, -width/2), (-length/2, width/2), (-width/2, width/2),
                  (-width/2, length/2)]
        return msp.add_lwpolyline(points, dxfattribs=LayDose_dict)

    @staticmethod
    def ellipse(msp, width, length, LayDose_dict={"layer": "1"}):
        return msp.add_ellipse((0, 0), major_axis=(width/2, 0), ratio=length/width, dxfattribs=LayDose_dict)

    @staticmethod
    def ring(msp, inner_radius, outer_radius, LayDose_dict={"layer": "1"}):
        msp.add_circle((0, 0), outer_radius, dxfattribs=LayDose_dict)
        msp.add_circle((0, 0), inner_radius, dxfattribs=LayDose_dict)

    @staticmethod
    def bowtie(msp, width, length, LayDose_dict={"layer": "1"}):
        points = [(-length/2, 0), (-length/2+width/2, width/2), (length/2-width/2, width/2), (length/2, 0),
                  (length/2-width/2, -width/2), (-length/2+width/2, -width/2), (-length/2, 0)]
        return msp.add_lwpolyline(points, dxfattribs=LayDose_dict)

    @staticmethod
    def bar(msp, width, length, LayDose_dict={"layer": "1"}):
        return msp.add_lwpolyline([(-length/2, -width/2), (length/2, -width/2), (length/2, width/2), (-length/2, width/2), (-length/2, -width/2)], dxfattribs=LayDose_dict)

    @staticmethod
    def curvedbar(msp, width, length, LayDose_dict={"layer": "1"}):
        center_distance = length - width
        msp.add_arc((center_distance/2, 0), width/2, 90, 270, dxfattribs=LayDose_dict)
        msp.add_arc((-center_distance/2, 0), width/2, 270, 90, dxfattribs=LayDose_dict)
        msp.add_line((center_distance/2, width/2), (-center_distance/2, width/2), dxfattribs=LayDose_dict)
        msp.add_line((center_distance/2, -width/2), (-center_distance/2, -width/2), dxfattribs=LayDose_dict)

    @staticmethod
    def clothpeg(msp, width, length, LayDose_dict={"layer": "1"}):
        points = [(-length/2+width, 0), (-length/2+width/2, width/2), (length/2-width/2, width/2),
                  (length/2-width, 0), (length/2-width/2, -width/2), (-length/2+width/2, -width/2),
                  (-length/2+width, 0)]
        return msp.add_lwpolyline(points, dxfattribs=LayDose_dict)

    @staticmethod
    def contact(msp, wire_width, wire_length, taper, distance_pad, pad_size, LayDose_dict={"layer": "1"}):
        points = [(0, 0), (0, wire_length), (taper/2, wire_length+taper),
                  (pad_size/2, wire_length+taper+distance_pad),
                  (pad_size/2, wire_length+taper+distance_pad+pad_size),
                  (-pad_size/2, wire_length+taper+distance_pad+pad_size),
                  (-pad_size/2, wire_length+taper+distance_pad),
                  (-taper/2, wire_length+taper), (0, wire_length), (0, 0)]
        return msp.add_lwpolyline(points, dxfattribs=LayDose_dict)

# Example usage:
def create_example_drawing():
    doc = ezdxf.new('R2010')
    msp = doc.modelspace()

    PolygonGenerator.rectangle(msp, 10, 20)
    PolygonGenerator.square(msp, 15)
    PolygonGenerator.circle(msp, 10)
    PolygonGenerator.octagon(msp, 10)
    PolygonGenerator.line(msp, 1, 20)
    PolygonGenerator.diamond(msp, 10, 20)
    PolygonGenerator.cross(msp, 5, 20)
    PolygonGenerator.ellipse(msp, 15, 10)
    PolygonGenerator.ring(msp, 8, 10)
    PolygonGenerator.bowtie(msp, 5, 20)
    PolygonGenerator.bar(msp, 5, 20)
    PolygonGenerator.curvedbar(msp, 5, 20)
    PolygonGenerator.clothpeg(msp, 5, 20)
    PolygonGenerator.contact(msp, 2, 10, 5, 5, 10)

    doc.saveas("example_drawing.dxf")

create_example_drawing()
