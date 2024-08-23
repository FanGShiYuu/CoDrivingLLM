import itertools
from typing import List, Tuple, TYPE_CHECKING

import numpy as np
import pygame

from highway_env.types import Vector
from highway_env.vehicle.dynamics import BicycleVehicle
from highway_env.vehicle.kinematics import Vehicle
from highway_env.vehicle.controller import ControlledVehicle, MDPVehicle
from highway_env.vehicle.behavior import IDMVehicle, LinearVehicle

if TYPE_CHECKING:
    from highway_env.road.graphics import WorldSurface


class VehicleGraphics(object):
    # RED = (255, 100, 100)
    # GREEN = (50, 200, 0)
    # BLUE = (100, 200, 255)
    # YELLOW = (200, 200, 0)
    # BLACK = (60, 60, 60)
    # PURPLE = (200, 0, 150)
    # DEFAULT_COLOR = YELLOW
    # EGO_COLOR = BLUE

    RED = (183, 62, 62)  # # Red(255, 100, 100)
    RED_PINK = (137, 55, 95)
    GREEN = (188, 226, 158)  # origin (50, 200, 0)
    GREEN_LIGHT = (144, 238, 144)
    ORANGE = (255, 181, 98)  # (248, 116, 116)
    BLUE = (154, 220, 255)  # (85, 73, 148)  #origin (100, 200, 255)
    BLUE_LIGHT = (135, 206, 250)
    # ORANGE = (255, 153, 0) #(248, 116, 116)
    # BLUE = (0, 51, 204)
    YELLOW = (200, 200, 0)
    BLACK = (128, 138, 135)  # 冷灰 origin (60, 60, 60)
    PURPLE = (200, 0, 150)
    DEFAULT_COLOR = YELLOW
    GREY_SHADOW = (154, 154, 152)  # (200, 200, 200)
    PURPLE_LIGHT = (221, 160, 221)
    # EGO_COLOR = GREEN
    EGO_COLOR = GREEN_LIGHT
    IDM_COLOR = GREY_SHADOW
    IDM_COLOR_Aggre = (226, 104, 104)

    @classmethod
    def display(cls, vehicle: Vehicle, surface: "WorldSurface", transparent: bool = False, offscreen: bool = False,
                label: bool = False) -> None:
        """
        Display a vehicle on a pygame surface.

        The vehicle is represented as a colored rotated rectangle.

        :param vehicle: the vehicle to be drawn
        :param surface: the surface to draw the vehicle on
        :param transparent: whether the vehicle should be drawn slightly transparent
        :param offscreen: whether the rendering should be done offscreen or not
        :param label: whether a text label should be rendered
        """
        if not surface.is_visible(vehicle.position):
            return

        v = vehicle
        tire_length, tire_width = 1, 0.3

        # Vehicle rectangle
        length = v.LENGTH + 2 * tire_length
        vehicle_surface = pygame.Surface((surface.pix(length), surface.pix(length)), flags=pygame.SRCALPHA)  # per-pixel alpha
        rect = (surface.pix(tire_length), surface.pix(length / 2 - v.WIDTH / 2), surface.pix(v.LENGTH), surface.pix(v.WIDTH))
        pygame.draw.rect(vehicle_surface, cls.get_color(v, transparent), rect, 0)
        pygame.draw.rect(vehicle_surface, cls.BLACK, rect, 1)

        # Tires
        if type(vehicle) in [Vehicle, BicycleVehicle]:
            tire_positions = [[surface.pix(tire_length), surface.pix(length / 2 - v.WIDTH / 2)],
                              [surface.pix(tire_length), surface.pix(length / 2 + v.WIDTH / 2)],
                              [surface.pix(length - tire_length), surface.pix(length / 2 - v.WIDTH / 2)],
                              [surface.pix(length - tire_length), surface.pix(length / 2 + v.WIDTH / 2)]]
            tire_angles = [0, 0, v.action["steering"], v.action["steering"]]
            for tire_position, tire_angle in zip(tire_positions, tire_angles):
                tire_surface = pygame.Surface((surface.pix(tire_length), surface.pix(tire_length)), pygame.SRCALPHA)
                rect = (0, surface.pix(tire_length/2-tire_width/2), surface.pix(tire_length), surface.pix(tire_width))
                pygame.draw.rect(tire_surface, cls.BLACK, rect, 0)
                cls.blit_rotate(vehicle_surface, tire_surface, tire_position, np.rad2deg(-tire_angle))

        # Centered rotation
        h = v.heading if abs(v.heading) > 2 * np.pi / 180 else 0
        position = [*surface.pos2pix(v.position[0], v.position[1])]
        if not offscreen:
            # convert_alpha throws errors in offscreen mode
            # see https://stackoverflow.com/a/19057853
            vehicle_surface = pygame.Surface.convert_alpha(vehicle_surface)
        cls.blit_rotate(surface, vehicle_surface, position, np.rad2deg(-h))

        # Label
        if label:
            font = pygame.font.Font(None, 20)
            # text = "#{}".format(id(v) % 1000)
            text = "#{}".format(v.id)
            text = font.render(text, 1, (10, 10, 10), (255, 255, 255))
            surface.blit(text, position)

    @staticmethod
    def blit_rotate(surf: pygame.SurfaceType, image: pygame.SurfaceType, pos: Vector, angle: float,
                    origin_pos: Vector = None, show_rect: bool = False) -> None:
        """Many thanks to https://stackoverflow.com/a/54714144."""
        # calculate the axis aligned bounding box of the rotated image
        w, h = image.get_size()
        box = [pygame.math.Vector2(p) for p in [(0, 0), (w, 0), (w, -h), (0, -h)]]
        box_rotate = [p.rotate(angle) for p in box]
        min_box = (min(box_rotate, key=lambda p: p[0])[0], min(box_rotate, key=lambda p: p[1])[1])
        max_box = (max(box_rotate, key=lambda p: p[0])[0], max(box_rotate, key=lambda p: p[1])[1])

        # calculate the translation of the pivot
        if origin_pos is None:
            origin_pos = w / 2, h / 2
        pivot = pygame.math.Vector2(origin_pos[0], -origin_pos[1])
        pivot_rotate = pivot.rotate(angle)
        pivot_move = pivot_rotate - pivot

        # calculate the upper left origin of the rotated image
        origin = (pos[0] - origin_pos[0] + min_box[0] - pivot_move[0], pos[1] - origin_pos[1] - max_box[1] + pivot_move[1])
        # get a rotated image
        rotated_image = pygame.transform.rotate(image, angle)
        # rotate and blit the image
        surf.blit(rotated_image, origin)
        # draw rectangle around the image
        if show_rect:
            pygame.draw.rect(surf, (255, 0, 0), (*origin, *rotated_image.get_size()), 2)

    @classmethod
    def display_trajectory(cls, states: List[Vehicle], surface: "WorldSurface", offscreen: bool = False) -> None:
        """
        Display the whole trajectory of a vehicle on a pygame surface.

        :param states: the list of vehicle states within the trajectory to be displayed
        :param surface: the surface to draw the vehicle future states on
        :param offscreen: whether the rendering should be done offscreen or not
        """
        for vehicle in states:
            cls.display(vehicle, surface, transparent=True, offscreen=offscreen)

    @classmethod
    def display_history(cls, vehicle: Vehicle, surface: "WorldSurface", frequency: float = 3, duration: float = 2,
                        simulation: int = 15, offscreen: bool = False) -> None:
        """
        Display the whole trajectory of a vehicle on a pygame surface.

        :param vehicle: the vehicle states within the trajectory to be displayed
        :param surface: the surface to draw the vehicle future states on
        :param frequency: frequency of displayed positions in history
        :param duration: length of displayed history
        :param simulation: simulation frequency
        :param offscreen: whether the rendering should be done offscreen or not
        """
        for v in itertools.islice(vehicle.history,
                                  None,
                                  int(simulation * duration),
                                  int(simulation / frequency)):
            cls.display(v, surface, transparent=True, offscreen=offscreen)

    @classmethod
    def get_color(cls, vehicle: Vehicle, transparent: bool = False) -> Tuple[int]:
        color = cls.DEFAULT_COLOR
        if getattr(vehicle, "color", None):
            color = vehicle.color
        elif vehicle.crashed:
            color = cls.RED
        elif isinstance(vehicle, LinearVehicle):
            color = cls.YELLOW
        elif isinstance(vehicle, IDMVehicle):
            color = cls.IDM_COLOR
        elif isinstance(vehicle, MDPVehicle):
            color = cls.EGO_COLOR
        if transparent:
            color = (color[0], color[1], color[2], 30)
        return color
