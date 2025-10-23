import math

def deg(rad):
    return rad * 180 / math.pi

def reim(v):
    return v.real, v.imag

def intreim(v):
    return int(round(v.real)), int(round(v.imag))

def cross(u, v):
    return u.real*v.imag - u.imag*v.real

