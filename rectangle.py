import math

class Point:
    # initialize the point cordinate
    def __init__(self, x=0.0, y=0.0):
        self.x = float(x)
        self.y = float(y)

    # calculate the distance between this point with another point
    def distance(self, p2):
        return math.sqrt((self.x-p2.x)**2+(self.y-p2.y)**2)

    # Check if two points are equal
    def __eq__(self, other):
        return (abs(self.x-other.x) < 1e-9) and (abs(self.y-other.y) < 1e-9)
    
# return a point rotated from the origin of rotation and theta (as degree) 
def rotate(point, theta, origin = Point(0.0, 0.0)):
    rad = (math.pi/180.0)*theta
    if origin == Point(0.0, 0.0):
        return Point(point.x*math.cos(rad) - point.y*math.sin(rad),
                        point.x*math.sin(rad) + point.y*math.cos(rad))
    else:
        p_temp = Point(point.x-origin.x, point.y-origin.y)
        p_temp = rotate(p_temp, theta)
        return Point(origin.x + p_temp.x, origin.y+p_temp.y)

def dist(p1, p2):
    return math.sqrt((p1.x-p2.x)**2+(p1.y-p2.y)**2)

class Line:
    def __init__(self, a=0.0, b=0.0, c=0.0):
        self.a = float(a)
        self.b = float(b)
        self.c = float(c)
        
    def points_to_line(self, p1, p2):
        if abs(p1.x - p2.x) < 1e-9:
            self.a = 1.0
            self.b = 0.0
            self.c = -float(p1.x)
        else:
            self.a = - float(p1.y-p2.y)/float(p1.x-p2.x)
            self.b  = 1.0
            self.c = - float(self.a*p1.x) - p1.y

def points_to_line(p1, p2):
    line = Line(0.0, 0.0, 0.0)
    if abs(p1.x - p2.x) < 1e-9:
        line.a = 1.0
        line.b = 0.0
        line.c = -float(p1.x)
    else:
        line.a = - float(p1.y-p2.y)/float(p1.x-p2.x)
        line.b  = 1.0
        line.c = - float(line.a*p1.x) - p1.y
    return line

def areParallel(l1, l2):
    if l1.b != 0 and l2.b != 0:
        return (abs(l1.a/l1.b - l2.a/l2.b) < 1e-9)
    else:
        return (abs(l2.b - l1.b) < 1e-9) and abs(l1.a - l2.a) < 1e-9

def areSame(l1, l2):
    if areParallel(l1, l2):
        if l1.a != 0:
            return abs(float(l1.c)/l1.a - float(l2.c)/l2.a) < 1e-9
        if l2.b != 0:
            return abs(float(l1.c)/l1.b - float(l2.c)/l2.b) < 1e-9
        return false
    else:
        return false
    
def areIntersect(l1, l2):
    _, intersect = False, Point(1e9, 1e9)
    if areParallel(l1, l2):
        return _, intersect
    else:
        _ = True
        
        intersect.x = (l2.b*l1.c - l1.b*l2.c)/(l2.a*l1.b - l1.a*l2.b)
        if (abs(l1.b)>1e-9):
            intersect.y = -(l1.a * intersect.x + l1.c)
        else:
            intersect.y = -(l2.a * intersect.x + l2.c)
    return _, intersect

class Vect:
    def __init__(x, y):
        self.x = float(x)
        self.y = float(y)

def toVec(a, b):
    return Vect(b.x - a.x, b.y - a.y)

def scale(v, s):
    return Vect(v.x*float(s), v.y*float(s))

def translate(p, v):
    return Point(p.x+v.x, p.y+v.y)

#  Dot product of 2 vector
def dot(a, b):
    return a.x*b.x + a.y*b.y

# Normalize square of vector
def normSquare(v):
    return v.x*v.x + v.y*v.y

# Distance from a point to a line go through 2 point
# Return the intersection point and the distance
def distToLine(p, a, b):
    ap = toVec(a,p)
    ab = toVec(a,b)
    u = float(dot(ap,ab)/norm(ab))
    c = translate(a, scale(ab, u))
    return c, p.distance(c)

class Fourgon:
    def __init__(self,p1 = Point(0, 0), p2 = Point(), p3 = Point(), p4 = Point()):
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.p4 = p4
    
    def translate(self, delta_x, delta_y):
        self.p1.x +=delta_x
        self.p2.x +=delta_x
        self.p3.x +=delta_x
        self.p4.x +=delta_x
        
        self.p1.y +=delta_y
        self.p2.y +=delta_y
        self.p3.y +=delta_y
        self.p4.y +=delta_y

    def rotate(self, origin, theta):
        self.p1 = rotate(self.p1, theta, origin)
        self.p2 = rotate(self.p2, theta, origin)
        self.p3 = rotate(self.p3, theta, origin)
        self.p4 = rotate(self.p4, theta, origin)
    
    def scale(self, origin, k):
        center = Point((self.p1.x+self.p2.x+self.p3.x+self.p4.x)/4, (self.p1.y+self.p2.y+self.p3.y+self.p4.y)/4)
        
        vec1 = toVec(origin, p1)
        vec2 = toVec(origin, p2)
        vec3 = toVec(origin, p3)
        vec4 = toVec(origin, p4)
        
        vec1 = scale(vec1, k)
        vec2 = scale(vec2, k)
        vec3 = scale(vec3, k)
        vec4 = scale(vec4, k)

        self.p1 = translate(center, vec1)
        self.p2 = translate(center, vec2)
        self.p3 = translate(center, vec3)
        self.p4 = translate(center, vec4)
