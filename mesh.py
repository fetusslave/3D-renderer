import numpy as np
from math import pi, sin, cos, acos
from matrix import project, translation_matrix, length, dot_product, normalize, vector_add, cross_product, \
    projection_matrix, x_rotation_matrix, y_rotation_matrix, z_rotation_matrix, point_at, matrix_inverse


# vertices are stored in clockwise order
class Triangle:
    def __init__(self, p1, p2, p3):
        self.p = [np.array(p1), np.array(p2), np.array(p3)]
        self.color = None

    # project the triangle in 3D space and returns the projection
    def project(self, mat):
        return Triangle(
            project(self.p[0], mat),
            project(self.p[1], mat),
            project(self.p[2], mat)
        )

    # set the color of the triangle
    def set_color(self, color):
        self.color = color

    # multiply each point in the triangle by a given matrix
    def multiply_matrix(self, mat):
        return Triangle(
            np.matmul(mat, self.p[0]),
            np.matmul(mat, self.p[1]),
            np.matmul(mat, self.p[2])
        )

    # multiply each point by a view matrix
    def multiply_view_mat(self, mat):
        return Triangle(
            np.matmul(self.p[0], mat),
            np.matmul(self.p[1], mat),
            np.matmul(self.p[2], mat)
        )

    # returns the normal of the triangle
    def normal(self):
        l1 = self.p[1]-self.p[0]
        l2 = self.p[2]-self.p[0]
        n = cross_product(l1, l2)
        return n/length(n)

    def center(self):
        return np.array([
            (self.p[0][0] + self.p[1][0] + self.p[2][0]) / 3,
            (self.p[0][1] + self.p[1][1] + self.p[2][1]) / 3,
            (self.p[0][2] + self.p[1][2] + self.p[2][2]) / 3
        ])

    # reverses the normal
    def flip(self):
        self.p[1], self.p[2] = self.p[2], self.p[1]

    def __str__(self):
        return self.p[0].__str__()+" "+self.p[1].__str__()+" "+self.p[2].__str__()


# translate brightness value between 0 and 1 to RGB
def get_color(lum: float):
    col = lum*255
    return (col, col, col)


class Mesh:
    # loads a mesh from an obj file
    @staticmethod
    def from_obj(filepath):
        vertices = []
        triangles = []

        with open(filepath, "r") as a:
            lines = a.read().strip().split("\n")
            for line in lines:
                if line == "":
                    continue
                # v means line describes a vertex
                if line[0] == "v":
                    split = line.split(" ")
                    vertices.append((
                        float(split[1]),
                        float(split[2]),
                        float(split[3]),
                        1
                    ))

            for line in lines:
                if line == "":
                    continue
                # f means line describes a face
                if line[0] == "f":
                    split = line.split(" ")
                    # indexes start from 1 in obj files
                    p1 = vertices[int(split[1]) - 1]
                    p2 = vertices[int(split[2]) - 1]
                    p3 = vertices[int(split[3]) - 1]
                    triangles.append(Triangle(
                        p1, p2, p3
                    ))
        return Mesh(triangles)

    # initialize mesh with a list of vertices
    def __init__(self, triangles: list, center=(0, 0, 0)):
        self.triangles = triangles
        self.center = center

    def translate(self, x, y, z):
        triangles = [tri.multiply_matrix(translation_matrix(x, y, z)) for tri in self.triangles]
        return Mesh(triangles, (self.center[0]+x, self.center[1]+y, self.center[2]+z))

    def rotate_x(self, angle):
        triangles = [tri.multiply_matrix(x_rotation_matrix(angle, self.center)) for tri in self.triangles]
        return Mesh(triangles, self.center)


class Light:
    def __init__(self, position, direction, brightness):
        self.position = position
        self.direction = direction
        self.brightness = brightness

    def calculate_luminosity(self, triangle):
        l = self.position-triangle.center()
        nl = l / length(l)
        return np.dot(triangle.normal(), nl)*self.brightness






# equation of plane, returns 0 if p lies on plane
def signed_distance(plane, p):
    return np.dot(plane[:3], p[:3])+plane[3]


def clip_triangle_planes(planes, triangle):
    triangles = [triangle]
    for plane in planes:
        for i in range(len(triangles)):
            tri = triangles.pop(0)
            inside = [0, 0, 0]
            inside_count = 0
            outside = [0, 0, 0]
            outside_count = 0

            # get signed distance between each point and the plane
            # negative means point is outside, positive means point is inside
            for p in tri.p:
                dist = signed_distance(plane, p)
                if dist >= 0:
                    inside[inside_count] = tri.p[0]
                    inside_count += 1
                else:
                    outside[outside_count] = tri.p[0]
                    outside_count += 1

            if inside_count == 3:
                triangles.append(tri)
            elif inside_count == 1:
                new_p1 = intersection(inside[0], outside[0], plane)
                new_p2 = intersection(inside[0], outside[1], plane)
                triangles.append(Triangle(inside[0], new_p1, new_p2))
            elif inside_count == 2:
                new_p1 = intersection(outside[0], inside[0], plane)
                new_p2 = intersection(outside[1], inside[1], plane)
                triangles.append(Triangle(outside[0], outside[1], new_p2))


def intersection(start, end, normal):
    # unit vector
    dir_vec = normalize(end-start)
    # angle between line and normal
    cosp = abs(dot_product(normal, dir_vec))/(length(normal))
    # angle between line and plane
    theta = pi/2-acos(cosp)
    # nearest distance between start and plane
    d = signed_distance(normal, start)
    h = d/sin(theta)
    return start+dir_vec*h
















