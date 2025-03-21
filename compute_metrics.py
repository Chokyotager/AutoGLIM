import math

import numpy as np

def calculate_localisation_quotient (x_r, y_r, x_g, y_g, x_b, y_b):

    def get_angle (a, b, c):

        a = np.array(a)
        b = np.array(b)
        c = np.array(c)

        ba = a - b
        bc = c - b

        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(cosine_angle)

        return angle

    d1 = math.sqrt((x_b - x_r) ** 2 + (y_b - y_r) ** 2)

    red = [x_r, y_r]
    green = [x_g, y_g]
    blue = [x_b, y_b]

    # RBG = alpha
    # In radians!
    alpha = get_angle(red, blue, green)
    abs_tan_a = abs(math.tan(alpha))

    beta = get_angle(green, red, blue)
    abs_tan_b = abs(math.tan(beta))

    # TOA CAH SOH
    hypothenus = math.sqrt((x_b - x_g) ** 2 + (y_b - y_g) ** 2)

    dx = hypothenus * math.cos(alpha)

    lq = dx / d1

    return lq, dx, d1, abs_tan_a, abs_tan_b