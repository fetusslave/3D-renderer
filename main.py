import numpy as np
import pygame
from pygame.locals import KEYDOWN, K_ESCAPE, K_SPACE, K_UP, K_DOWN, K_LEFT, K_RIGHT, K_w, K_a, K_s, K_d, QUIT
from math import pi, sin, cos, tan
import time
from matrix import *
from mesh import *


# draws a filled triangle on the screen
def draw_fill(screen, triangle):
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


class Scene:
    z_n = 0.1  # nearest from screen
    z_f = 8000  # furthest from screen
    fov = pi / 2  # field of view

    def __init__(self, width, height):
        self.WIDTH = width
        self.HEIGHT = height
        self.aspect_ratio = self.HEIGHT / self.WIDTH
        self.mat_proj = projection_matrix(self.z_n, self.z_f, self.fov, self.aspect_ratio)

        # camera position
        self.camera = np.array([0, 0, 0, 1], dtype=np.float16)
        # direction camera is pointing, unit vector
        self.camera_direction = np.array([0, 0, 1, 1], dtype=np.float16)
        self.v_up = np.array([0, 1, 0, 1])

        self.light_direction = np.array([0, 0, -1])
        self.nl = self.light_direction / length(self.light_direction)

        self.light_sources = {}

        self.meshes = {}

        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.screen.fill((0, 0, 0))

    def add_mesh(self, name, mesh):
        self.meshes.update({name: mesh})

    def add_light_source(self, name, light):
        self.light_sources.update({name: light})

    def draw(self):
        self.screen.fill((0, 0, 0))

        v_target = self.camera + self.camera_direction
        camera_mat = point_at(self.camera, v_target, self.v_up)
        view_mat = matrix_inverse(camera_mat)

        all_triangles = []
        for m in self.meshes.values():
            all_triangles += m.triangles

        triangles = []

        # calculate the projected triangles
        for tri in all_triangles:

            normal = tri.normal()

            # check if the triangle is visible
            if np.dot(normal, tri.p[0][:3] - self.camera[:3]) < 0:
                viewed = tri.multiply_view_mat(view_mat)
                projected = viewed.project(self.mat_proj)

                # find luminosity for all light sources
                lum = 0
                for light in self.light_sources.values():
                    lum += light.calculate_luminosity(tri)

                # clip the value to be between 0 and 1
                projected.set_color(get_color(min(1, max(0, lum))))
                # add the triangle to the projected triangle list
                triangles.append(projected)

        # sort triangles by average z value
        triangles.sort(key=lambda t: (t.p[0][2] + t.p[1][2] + t.p[2][2]) / 3, reverse=True)

        # draw triangles in order of distance to camera
        for tri in triangles:
            draw_fill(self.screen, tri)

    def get_xy(self, triangle):
        return [np.round((point[:2] + 1) * 0.5 * [self.WIDTH, self.HEIGHT]) for point in triangle.p]

    def move_camera(self, x, y, z):
        self.camera[0] += x
        self.camera[1] += y
        self.camera[2] += z

    def rotate_camera_x(self, angle):
        self.camera_direction = normalize(np.matmul(x_rotation_matrix(angle, self.camera), self.camera_direction))

    def rotate_camera_y(self, angle):
        self.camera_direction = normalize(np.matmul(y_rotation_matrix(angle, self.camera), self.camera_direction))

    def rotate_camera_z(self, angle):
        self.camera_direction = normalize(np.matmul(z_rotation_matrix(angle, self.camera), self.camera_direction))


# load an obj file
mesh = Mesh.from_obj("meshes/cube.obj").translate(0, 0, 40)

l1 = Light(np.array([0., 0., 0.]), np.array([0., 0., -1.]), 0.6)
l2 = Light(np.array([10., 0., 20.]), np.array([-1., 0., 0.]), 0.3)

scene = Scene(WIDTH, HEIGHT)
scene.add_mesh("mesh1", mesh)
scene.add_light_source('light1', l1)
scene.add_light_source('light2', l2)


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

    # rotate camera with up/down/left/right keys
    if keys[K_UP]:
        scene.rotate_camera_x(-0.001)
    elif keys[K_DOWN]:
        scene.rotate_camera_x(0.001)
    elif keys[K_RIGHT]:
        scene.rotate_camera_y(0.001)
    elif keys[K_LEFT]:
        scene.rotate_camera_y(-0.001)

    # move camera position with WASD keys
    if keys[K_w]:
        scene.move_camera(0, -0.1, 0)
    elif keys[K_s]:
        scene.move_camera(0, 0.1, 0)
    elif keys[K_d]:
        scene.move_camera(-0.1, 0, 0)
    elif keys[K_a]:
        scene.move_camera(0.1, 0, 0)

    scene.draw()

    pygame.display.flip()
    clock.tick(fps)
