import numpy as np
import pygame
from pygame.locals import KEYDOWN, K_ESCAPE, K_SPACE, K_UP, K_DOWN, K_LEFT, K_RIGHT, QUIT
from math import pi, sin, cos, tan
import time
from matrix import *
from mesh import *


# draws a filled triangle on the screen
def draw_fill(screen, triangle):
    #print(triangle.color)
    pygame.draw.polygon(screen, triangle.color, get_xy(triangle))


# normalize x and y coordinates of a triangle to the screen size
def get_xy(triangle):
    global WIDTH, HEIGHT
    return [np.round((point[:2]+1)*0.5*[WIDTH, HEIGHT]) for point in triangle.p]


pygame.init()

clock = pygame.time.Clock()

fps = 60


WIDTH = 1280
HEIGHT = 720

z_n = 0.1   # nearest from screen
z_f = 8000  # furthest from screen
fov = pi/2  # field of view
aspect_ratio = HEIGHT/WIDTH

mat_proj = projection_matrix(z_n, z_f, fov, aspect_ratio)

light_direction = np.array([0, 0, -1])
nl = light_direction/length(light_direction)

screen = pygame.display.set_mode((WIDTH, HEIGHT))
screen.fill((0, 0, 0))


# load an obj file
a = Mesh.from_obj("meshes/cube.obj").translate(0, 0, 20)


# camera position
camera = np.array([0, 0, 0, 1], dtype=np.float16)


# direction camera is pointing, unit vector
camera_direction = np.array([0, 0, 1, 1])

v_up = np.array([0, 1, 0, 1])

v_target = camera+camera_direction


start = time.time()

running = True


while running:
    for event in pygame.event.get():
        if event.type == QUIT:
            running = False
        elif event.type == KEYDOWN:
            if event.key == K_ESCAPE:
                running = False

    keys = pygame.key.get_pressed()

    # move camera position
    if keys[K_UP]:
        camera[2] += 0.1
    elif keys[K_DOWN]:
        camera[2] -= 0.1
    elif keys[K_RIGHT]:
        camera[0] -= 0.1
    elif keys[K_LEFT]:
        camera[0] += 0.1

    screen.fill((0, 0, 0))

    t = time.time()-start

    # rotation matrix for the mesh
    rotation_mat = y_rotation_matrix(t/2, (0, 0, 20))

    # time is in milliseconds
    cam_rotation = y_rotation_matrix(t, (0, 0, 20))
    dir_rotation = y_rotation_matrix(t)

    # rotate camera
    cam = np.matmul(cam_rotation, camera)
    look_dir = np.matmul(dir_rotation, camera_direction)

    v_target = camera + camera_direction
    camera_mat = point_at(camera, v_target, v_up)
    view_mat = matrix_inverse(camera_mat)

    triangles = []

    # calculate the projected triangles
    for tri in a.triangles:
        # rotate the triangle
        rotated = tri.multiply_matrix(rotation_mat)

        normal = rotated.normal()

        # check if the triangle is visible
        if np.dot(normal, rotated.p[0][:3]-camera[:3]) < 0:
            viewed = rotated.multiply_view_mat(view_mat)
            projected = viewed.project(mat_proj)

            # luminosity of the triangle
            # dot product of the rotated triangle's normal and the light source's normal
            dp = np.dot(rotated.normal(), nl)
            # clip the value to be greater or equal to 0
            projected.set_color(get_color(max(0, dp)))
            # add the triangle to the projected triangle list
            triangles.append(projected)

    # sort triangles by average z value
    triangles.sort(key=lambda t: (t.p[0][2]+t.p[1][2]+t.p[2][2])/3, reverse=True)

    # draw triangles in order of distance to camera
    for tri in triangles:
        draw_fill(screen, tri)

    pygame.display.flip()
    clock.tick(fps)
