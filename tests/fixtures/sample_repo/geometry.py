def triangle_area(base, height):
    return base * height / 2


def rectangle_area(w, h):
    return w * h


def circle_area(r):
    return 3.14159 * r ** 2


def halve(x):
    return x / 2


def scale(value, factor):
    return value * factor


def describe_shapes(shapes):
    # Nested function to test tree-sitter
    def label(shape):
        return f"{shape.name}: {shape.area():.2f}"

    return [label(s) for s in shapes]


class Rectangle:
    def __init__(self, w, h):
        self.w = w
        self.h = h

    def area(self):
        return self.w * self.h

    def perimeter_parts(self):
        def side_pairs():
            return [(self.w, self.h), (self.w, self.h)]

        return sum(a + b for a, b in side_pairs())


class Circle:
    def __init__(self, r):
        self.r = r

    def area(self):
        return 3.14159 * self.r ** 2
