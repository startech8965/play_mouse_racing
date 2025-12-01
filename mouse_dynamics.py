import math

import Box2D
import numpy as np

from gymnasium.error import DependencyNotInstalled


try:
    from Box2D.b2 import fixtureDef, polygonShape, revoluteJointDef
except ImportError as e:
    raise DependencyNotInstalled(
        'Box2D is not installed, you can install it by run `pip install swig` followed by `pip install "gymnasium[box2d]"`'
    ) from e


SIZE = 0.02
ENGINE_POWER = 100000000 * SIZE * SIZE
WHEEL_MOMENT_OF_INERTIA = 4000 * SIZE * SIZE
FRICTION_LIMIT = (
    1000000 * SIZE * SIZE
)  # friction ~= mass ~= size^2 (calculated implicitly using density)
WHEEL_R = 27
WHEEL_W = 14
WHEELPOS = [(-55, +80), (+55, +80), (-55, -82), (+55, -82)]
WHEEL_EXTENT_X = max(abs(x) for x, y in WHEELPOS)
WHEEL_EXTENT_Y = max(abs(y) for x, y in WHEELPOS)
# Rectangle just beyond the wheels in length, keep width
RECT_PAD = 10  # extra length beyond wheels
HULL_RECT = [
    (-WHEEL_EXTENT_X, WHEEL_EXTENT_Y + RECT_PAD),
    (WHEEL_EXTENT_X, WHEEL_EXTENT_Y + RECT_PAD),
    (WHEEL_EXTENT_X, -WHEEL_EXTENT_Y - RECT_PAD),
    (-WHEEL_EXTENT_X, -WHEEL_EXTENT_Y - RECT_PAD),
]
# Small rectangle at the front (same width, short height)
FRONT_RECT_HEIGHT = 20
FRONT_RECT_OFFSET = WHEEL_EXTENT_Y + RECT_PAD + FRONT_RECT_HEIGHT // 2
HULL_FRONT_RECT = [
    (-WHEEL_EXTENT_X, WHEEL_EXTENT_Y + RECT_PAD + FRONT_RECT_HEIGHT),
    (WHEEL_EXTENT_X, WHEEL_EXTENT_Y + RECT_PAD + FRONT_RECT_HEIGHT),
    (WHEEL_EXTENT_X, WHEEL_EXTENT_Y + RECT_PAD),
    (-WHEEL_EXTENT_X, WHEEL_EXTENT_Y + RECT_PAD),
]
# Grey connector rectangle (long, thin, same width, between main and front rect)
CONNECTOR_HEIGHT = 8
CONNECTOR_LENGTH = 30
CONNECTOR_Y1 = WHEEL_EXTENT_Y + RECT_PAD
CONNECTOR_Y2 = CONNECTOR_Y1 + CONNECTOR_LENGTH
HULL_CONNECTOR_RECT = [
    (-WHEEL_EXTENT_X, CONNECTOR_Y2),
    (WHEEL_EXTENT_X, CONNECTOR_Y2),
    (WHEEL_EXTENT_X, CONNECTOR_Y1),
    (-WHEEL_EXTENT_X, CONNECTOR_Y1),
]
WHEEL_COLOR = (0, 0, 0)
WHEEL_WHITE = (77, 77, 77)
MUD_COLOR = (102, 102, 0)


class Mouse:
    def __init__(self, world, init_angle, init_x, init_y):
        self.world: Box2D.b2World = world
        # Define darker green color
        dark_green = (0.0, 0.39, 0.0)
        # Main body fixture (darker green)
        main_fixture = fixtureDef(
            shape=polygonShape(
                vertices=[(x * SIZE, y * SIZE) for x, y in HULL_RECT]
            ),
            density=1.0,
        )
        # Grey connector square (width = length, both much smaller than body width)
        square_size = WHEEL_EXTENT_X * 0.5  # much smaller than body width
        square_center_y = WHEEL_EXTENT_Y + RECT_PAD + FRONT_RECT_HEIGHT + square_size / 2
        connector_rect = [
            (-square_size/2, square_center_y - square_size/2),
            (square_size/2, square_center_y - square_size/2),
            (square_size/2, square_center_y + square_size/2),
            (-square_size/2, square_center_y + square_size/2)
        ]
        connector_fixture = fixtureDef(
            shape=polygonShape(
                vertices=[(x * SIZE, y * SIZE) for x, y in connector_rect]
            ),
            density=1.0,
        )
        # Front small rectangle (darker green)
        front_fixture = fixtureDef(
            shape=polygonShape(
                vertices=[(x * SIZE, y * SIZE) for x, y in HULL_FRONT_RECT]
            ),
            density=1.0,
        )


        # Red rectangle in front of the grey square (width 5x green, 1/9 the green height)
        green_height = FRONT_RECT_HEIGHT
        red_height = (green_height / 9.0) * 10  # 10x the previous red rectangle height
        # The grey square top y is square_center_y + square_size/2
        red_y1 = square_center_y + square_size/2
        red_y2 = red_y1 + red_height
        red_width = WHEEL_EXTENT_X  # same as the green rectangle width
        red_rect = [
            (-red_width, red_y2),
            (red_width, red_y2),
            (red_width, red_y1),
            (-red_width, red_y1),
        ]
        red_fixture = fixtureDef(
            shape=polygonShape(
                vertices=[(x * SIZE, y * SIZE) for x, y in red_rect]
            ),
            density=1.0,
        )

        self.hull: Box2D.b2Body = self.world.CreateDynamicBody(
            position=(init_x, init_y),
            angle=init_angle,
            fixtures=[main_fixture, connector_fixture, front_fixture, red_fixture],
        )
        self.hull.color = dark_green  # Dark green rectangle body
        # Store colors for each fixture: main (dark green), connector (grey), front (dark green), red (red)
        self.hull.fixture_colors = [
            dark_green,           # main body (dark green)
            (0.5, 0.5, 0.5),     # connector (grey)
            dark_green,           # front (dark green)
            (1.0, 0.0, 0.0),     # red (red)
        ]
        self.wheels = []
        self.fuel_spent = 0.0
        WHEEL_POLY = [
            (-WHEEL_W, +WHEEL_R),
            (+WHEEL_W, +WHEEL_R),
            (+WHEEL_W, -WHEEL_R),
            (-WHEEL_W, -WHEEL_R),
        ]
        for wx, wy in WHEELPOS:
            front_k = 1.0 if wy > 0 else 1.0
            w = self.world.CreateDynamicBody(
                position=(init_x + wx * SIZE, init_y + wy * SIZE),
                angle=init_angle,
                fixtures=fixtureDef(
                    shape=polygonShape(
                        vertices=[
                            (x * front_k * SIZE, y * front_k * SIZE)
                            for x, y in WHEEL_POLY
                        ]
                    ),
                    density=0.1,
                    categoryBits=0x0020,
                    maskBits=0x001,
                    restitution=0.0,
                ),
            )
            w.wheel_rad = front_k * WHEEL_R * SIZE
            w.color = (1.0, 0.0, 0.0)  # Red wheels
            w.gas = 0.0
            w.brake = 0.0
            w.steer = 0.0
            w.phase = 0.0  # wheel angle
            w.omega = 0.0  # angular velocity
            w.skid_start = None
            w.skid_particle = None
            rjd = revoluteJointDef(
                bodyA=self.hull,
                bodyB=w,
                localAnchorA=(wx * SIZE, wy * SIZE),
                localAnchorB=(0, 0),
                enableMotor=True,
                enableLimit=True,
                maxMotorTorque=180 * 900 * SIZE * SIZE,
                motorSpeed=0,
                lowerAngle=-0.4,
                upperAngle=+0.4,
            )
            w.joint = self.world.CreateJoint(rjd)
            w.tiles = set()
            w.userData = w
            self.wheels.append(w)
        self.drawlist = self.wheels + [self.hull]
        self.particles = []

    def gas(self, gas):
        """control: rear wheel drive

        Args:
            gas (float): How much gas gets applied. Gets clipped between -1 and 1. Negative for reverse.
        """
        gas = np.clip(gas, -1, 1)
        for w in self.wheels[2:4]:
            diff = gas - w.gas
            if diff > 0.1:
                diff = 0.1  # gradually increase, but stop immediately
            if diff < -0.1:
                diff = -0.1  # gradually decrease for reverse as well
            w.gas += diff

    def brake(self, b):
        """control: brake

        Args:
            b (0..1): Degree to which the brakes are applied. More than 0.9 blocks the wheels to zero rotation
        """
        for w in self.wheels:
            w.brake = b

    def steer(self, s):
        """control: steer

        Args:
            s (-1..1): target position, it takes time to rotate steering wheel from side-to-side
        """
        self.wheels[0].steer = s
        self.wheels[1].steer = s

    def step(self, dt):
        for w in self.wheels:
            # Steer each wheel
            dir = np.sign(w.steer - w.joint.angle)
            val = abs(w.steer - w.joint.angle)
            w.joint.motorSpeed = dir * min(50.0 * val, 3.0)

            # Position => friction_limit
            grass = True
            friction_limit = FRICTION_LIMIT * 1.0 # Grass friction if no tile
            for tile in w.tiles:
                friction_limit = max(
                    friction_limit, FRICTION_LIMIT * tile.road_friction
                )
                grass = False

            # Force
            forw = w.GetWorldVector((0, 1))
            side = w.GetWorldVector((1, 0))
            v = w.linearVelocity
            vf = forw[0] * v[0] + forw[1] * v[1]  # forward speed
            vs = side[0] * v[0] + side[1] * v[1]  # side speed

            # WHEEL_MOMENT_OF_INERTIA*np.square(w.omega)/2 = E -- energy
            # WHEEL_MOMENT_OF_INERTIA*w.omega * domega/dt = dE/dt = W -- power
            # domega = dt*W/WHEEL_MOMENT_OF_INERTIA/w.omega

            # add small coef not to divide by zero
            w.omega += (
                dt
                * ENGINE_POWER
                * w.gas
                / WHEEL_MOMENT_OF_INERTIA
                / (abs(w.omega) + 5.0)
            )
            self.fuel_spent += dt * ENGINE_POWER * w.gas

            if w.brake >= 0.9:
                w.omega = 0
            elif w.brake > 0:
                BRAKE_FORCE = 15  # radians per second
                dir = -np.sign(w.omega)
                val = BRAKE_FORCE * w.brake
                if abs(val) > abs(w.omega):
                    val = abs(w.omega)  # low speed => same as = 0
                w.omega += dir * val
            w.phase += w.omega * dt

            vr = w.omega * w.wheel_rad  # rotating wheel speed
            f_force = -vf + vr  # force direction is direction of speed difference
            p_force = -vs

            # Physically correct is to always apply friction_limit until speed is equal.
            # But dt is finite, that will lead to oscillations if difference is already near zero.

            # Random coefficient to cut oscillations in few steps (have no effect on friction_limit)
            f_force *= 205000 * SIZE * SIZE
            p_force *= 205000 * SIZE * SIZE
            force = np.sqrt(np.square(f_force) + np.square(p_force))
            force *= np.sign(w.gas)

            # Skid trace
            if abs(force) > 2.0 * friction_limit:
                if (
                    w.skid_particle
                    and w.skid_particle.grass == grass
                    and len(w.skid_particle.poly) < 30
                ):
                    w.skid_particle.poly.append((w.position[0], w.position[1]))
                elif w.skid_start is None:
                    w.skid_start = w.position
                else:
                    w.skid_particle = self._create_particle(
                        w.skid_start, w.position, grass
                    )
                    w.skid_start = None
            else:
                w.skid_start = None
                w.skid_particle = None

            if abs(force) > friction_limit:
                f_force /= force
                p_force /= force
                force = friction_limit  # Correct physics here
                f_force *= force
                p_force *= force

            w.omega -= dt * f_force * w.wheel_rad / WHEEL_MOMENT_OF_INERTIA

            w.ApplyForceToCenter(
                (
                    p_force * side[0] + f_force * forw[0],
                    p_force * side[1] + f_force * forw[1],
                ),
                True,
            )

    def draw(self, surface, zoom, translation, angle, draw_particles=True):
        import pygame.draw

        if draw_particles:
            for p in self.particles:
                poly = [pygame.math.Vector2(c).rotate_rad(angle) for c in p.poly]
                poly = [
                    (
                        coords[0] * zoom + translation[0],
                        coords[1] * zoom + translation[1],
                    )
                    for coords in poly
                ]
                pygame.draw.lines(
                    surface, color=p.color, points=poly, width=2, closed=False
                )

        for obj in self.drawlist:
            # START
            # If this object is a wheel, skip drawing the physical wheel polygon
            # and instead draw a decorative leg that connects the hull to the
            # wheel position (quadruped look). Physics bodies remain unchanged.
            if obj in self.wheels:
                w = obj
                # wheel world position and hull center
                try:
                    wheel_pos = (w.position[0], w.position[1])
                except Exception:
                    continue
                hull_pos = (self.hull.position[0], self.hull.position[1])

                # leg root is a point on the hull closer to the wheel (interpolated)
                # reduce the root factor so legs appear longer visually
                root_factor = 0.12
                root_x = hull_pos[0] + root_factor * (wheel_pos[0] - hull_pos[0])
                root_y = hull_pos[1] + root_factor * (wheel_pos[1] - hull_pos[1])

                # animation offset from wheel phase for a simple gait
                phase = getattr(w, "phase", 0.0)
                swing = math.sin(phase) * 0.35

                # target foot: extend beyond the wheel position for longer legs
                foot_factor = 1.20
                foot_x = hull_pos[0] + foot_factor * (wheel_pos[0] - hull_pos[0]) + 0.02 * math.cos(swing)
                foot_y = hull_pos[1] + foot_factor * (wheel_pos[1] - hull_pos[1]) + 0.02 * math.sin(swing)

                # compute rectangle around the segment root->foot
                dx = foot_x - root_x
                dy = foot_y - root_y
                seg_len = math.hypot(dx, dy) + 1e-8
                # increase thickness so legs are clearly visible
                leg_thickness = max(0.45 * getattr(w, "wheel_rad", 0.02), 0.02)
                nx = -dy / seg_len * leg_thickness
                ny = dx / seg_len * leg_thickness

                poly_world = [
                    (root_x + nx, root_y + ny),
                    (root_x - nx, root_y - ny),
                    (foot_x - nx, foot_y - ny),
                    (foot_x + nx, foot_y + ny),
                ]

                poly_screen = [pygame.math.Vector2(c).rotate_rad(angle) for c in poly_world]
                poly_screen = [
                    (coords[0] * zoom + translation[0], coords[1] * zoom + translation[1])
                    for coords in poly_screen
                ]
                # leg color (darker brown) and draw
                leg_color = (120, 80, 40)
                pygame.draw.polygon(surface, color=leg_color, points=poly_screen)

                # draw a larger foot circle at the foot position for visibility
                foot_screen = pygame.math.Vector2((foot_x, foot_y)).rotate_rad(angle)
                foot_screen = (foot_screen[0] * zoom + translation[0], foot_screen[1] * zoom + translation[1])
                foot_radius = max(2, int(6 * zoom * getattr(w, "wheel_rad", 0.02)))
                pygame.draw.circle(surface, (80, 40, 20), (int(foot_screen[0]), int(foot_screen[1])), foot_radius)
                continue
            # END

            for i, f in enumerate(obj.fixtures):
                trans = f.body.transform
                path = [trans * v for v in f.shape.vertices]
                path = [(coords[0], coords[1]) for coords in path]
                path = [pygame.math.Vector2(c).rotate_rad(angle) for c in path]
                path = [
                    (
                        coords[0] * zoom + translation[0],
                        coords[1] * zoom + translation[1],
                    )
                    for coords in path
                ]
                # Use per-fixture color for hull, else default
                if hasattr(obj, 'fixture_colors') and i < len(obj.fixture_colors):
                    color = [int(c * 255) for c in obj.fixture_colors[i]]
                else:
                    color = [int(c * 255) for c in obj.color]

                pygame.draw.polygon(surface, color=color, points=path)

                if "phase" not in obj.__dict__:
                    continue
                a1 = obj.phase
                a2 = obj.phase + 1.2  # radians
                s1 = math.sin(a1)
                s2 = math.sin(a2)
                c1 = math.cos(a1)
                c2 = math.cos(a2)
                if s1 > 0 and s2 > 0:
                    continue
                if s1 > 0:
                    c1 = np.sign(c1)
                if s2 > 0:
                    c2 = np.sign(c2)
                white_poly = [
                    (-WHEEL_W * SIZE, +WHEEL_R * c1 * SIZE),
                    (+WHEEL_W * SIZE, +WHEEL_R * c1 * SIZE),
                    (+WHEEL_W * SIZE, +WHEEL_R * c2 * SIZE),
                    (-WHEEL_W * SIZE, +WHEEL_R * c2 * SIZE),
                ]
                white_poly = [trans * v for v in white_poly]

                white_poly = [(coords[0], coords[1]) for coords in white_poly]
                white_poly = [
                    pygame.math.Vector2(c).rotate_rad(angle) for c in white_poly
                ]
                white_poly = [
                    (
                        coords[0] * zoom + translation[0],
                        coords[1] * zoom + translation[1],
                    )
                    for coords in white_poly
                ]
                pygame.draw.polygon(surface, color=WHEEL_WHITE, points=white_poly)

        

    def _create_particle(self, point1, point2, grass):
        class Particle:
            pass

        p = Particle()
        p.color = WHEEL_COLOR if not grass else MUD_COLOR
        p.ttl = 1
        p.poly = [(point1[0], point1[1]), (point2[0], point2[1])]
        p.grass = grass
        self.particles.append(p)
        while len(self.particles) > 30:
            self.particles.pop(0)
        return p

    def destroy(self):
        self.world.DestroyBody(self.hull)
        self.hull = None
        for w in self.wheels:
            self.world.DestroyBody(w)
        self.wheels = []