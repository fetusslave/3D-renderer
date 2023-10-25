import numpy as np
from math import sin, cos, tan


# calculate the projection matrix
def projection_matrix(z_near, z_far, field_of_view, aspect_ratio):
    tangent = 1 / tan(field_of_view / 2)
    q = z_far / (z_far - z_near)

    mat_proj = np.zeros((4, 4), dtype=np.float32)

    mat_proj[0][0] = aspect_ratio * tangent
    mat_proj[1][1] = tangent
    mat_proj[2][2] = q
    mat_proj[3][2] = -q * z_near
    mat_proj[2][3] = 1
    return mat_proj


# project a point in 3D space and  returns the projected point
def project(point, mat) -> np.array:
    res = np.matmul(mat, point)
    if res[3] != 0:
        res[0] = res[0]/res[3]
        res[1] = res[1] / res[3]
    return res


def x_rotation_matrix(angle, center=(0, 0, 0)):
    s = sin(angle)
    c = cos(angle)

    y = center[1]
    z = center[2]

    return np.array([[1, 0, 0, 0],
                     [0, c, -s, z * s-y * c + y],
                     [0, s, c, -y * s - z * c + z],
                     [0, 0, 0, 1]])


def y_rotation_matrix(angle, center=(0, 0, 0)):
    s = sin(angle)
    c = cos(angle)

    x = center[0]
    z = center[2]

    return np.array([[c, 0, s, -x*c-z*s+x],
                     [0, 1, 0, 0],
                     [-s, 0, c, x*s-z*c+z],
                     [0, 0, 0, 1]])


def z_rotation_matrix(angle, center=(0, 0, 0)):
    s = sin(angle)
    c = cos(angle)

    x = center[0]
    y = center[1]

    return np.array([[c, -s, 0, y*s-c*x+x],
                     [s, c, 0, -x*s-y*c+y],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])


def translation_matrix(x, y, z):
    return np.array([[1, 0, 0, x],
                     [0, 1, 0, y],
                     [0, 0, 1, z],
                     [0, 0, 0, 1]])


# makes the camera point at the target point
def point_at(p, target, up):
    forward = target[:3]-p[:3]
    forward = forward/length(forward)

    dp = dot_product(forward, up)

    new_up = up[:3]-(forward[:3]*dp)
    new_up = new_up/length(new_up)

    right = np.cross(new_up, forward)

    return np.array([
        [right[0], right[1], right[2], 0],
        [new_up[0], new_up[1], new_up[2], 0],
        [forward[0], forward[1], forward[2], 0],
        [p[0], p[1], p[2], 1]
    ])


# inverse a given matrix
def matrix_inverse(m):
    a = m[0, :3]
    b = m[1, :3]
    c = m[2, :3]
    t = m[3, :3]
    return np.array([
        [m[0][0], m[1][0], m[2][0], 0],
        [m[0][1], m[1][1], m[2][1], 0],
        [m[0][2], m[1][2], m[2][2], 0],
        [-np.dot(t, a), -np.dot(t, b), -np.dot(t, c), 1]
    ])


# returns the cross product of 2 vectors
# only considers the first 3 values in the given vectors
def cross_product(v1, v2):
    return np.cross(v1[:3], v2[:3])


# returns the dot product of 2 vectors
def dot_product(p1, p2):
    return np.dot(p1[:3], p2[:3])


# add 2 vectors
def vector_add(v1, v2):
    v = v1+v2
    v[3] -= v2[3]
    return v


# returns a unit vector
def normalize(vector):
    normalized = np.zeros(4)
    normalized[:3] = vector[:3]/length(vector)
    normalized[3] = vector[3]
    return normalized


# returns the length of a vector
def length(line):
    return (line[0]**2+line[1]**2+line[2]**2)**0.5
