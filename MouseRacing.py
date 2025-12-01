import math
import random

import numpy as np

import gymnasium as gym
from gymnasium import spaces
from mouse_dynamics import Mouse
from gymnasium.error import DependencyNotInstalled, InvalidAction
from gymnasium.utils import EzPickle

import Box2D
from Box2D.b2 import contactListener, fixtureDef, polygonShape

import pygame
from pygame import gfxdraw

STATE_W = 96  # less than Atari 160x192
STATE_H = 96
VIDEO_W = 600
VIDEO_H = 400
WINDOW_W = 1000
WINDOW_H = 800

SCALE = 6.0  # Track scale
TRACK_RAD = 900 / SCALE  # Track is heavily morphed circle with this radius
PLAYFIELD = 2000 / SCALE  # Game over boundary
FPS = 50  # Frames per second
ZOOM = 2.7  # Camera zoom
ZOOM_FOLLOW = True  # Set to False for fixed view (don't use zoom)


TRACK_DETAIL_STEP = 21 / SCALE
TRACK_TURN_RATE = math.pi/2 #~1.57
TRACK_WIDTH = (60 / 2) / SCALE  # 2 times smaller than original
BORDER = 8 / SCALE
BORDER_MIN_COUNT = 4
GRASS_DIM = PLAYFIELD / 20.0
MAX_SHAPE_DIM = (
    max(GRASS_DIM, TRACK_WIDTH, TRACK_DETAIL_STEP) * math.sqrt(2) * ZOOM * SCALE
)
LEAK_SCALE = 1.0


class FrictionDetector(contactListener):
    def __init__(self, env, lap_complete_percent):
        contactListener.__init__(self)
        self.env = env
        self.lap_complete_percent = lap_complete_percent

    def BeginContact(self, contact):
        self._contact(contact, True)


    def EndContact(self, contact):
        self._contact(contact, False)

    def _contact(self, contact, begin):
        tile = None
        obj = None
        u1 = contact.fixtureA.body.userData
        u2 = contact.fixtureB.body.userData
        # Wall contact detection: if either body is marked as a wall, penalize the other
        wall1 = u1 if (u1 and getattr(u1, "is_wall", False)) else None
        wall2 = u2 if (u2 and getattr(u2, "is_wall", False)) else None
        if wall1 or wall2:
            wall = wall1 if wall1 is not None else wall2
            other = u2 if wall is u1 else u1
            # Penalize the agent (other) when it begins contact with a wall
            if begin and other is not None:
                try:
                    # Small penalty for scraping the wall; tune as needed
                    self.env.reward -= 0.5
                except Exception:
                    pass
            # We still continue to check for tile contact below, but return early
            # if there is no tile involved in this contact
            # (wall vs tile collisions don't need further handling here).
            if not (hasattr(other, "tiles") or (hasattr(other, "userData") and other.userData)):
                return
        if u1 and "road_friction" in u1.__dict__:
            tile = u1
            obj = u2
        if u2 and "road_friction" in u2.__dict__:
            tile = u2
            obj = u1
        if not tile:
            return

        # inherit tile color from env
        tile.color[:] = self.env.road_color
        if not obj or "tiles" not in obj.__dict__:
            return
        if begin:
            obj.tiles.add(tile)
            if not tile.road_visited:
                tile.road_visited = True
                self.env.reward += 1000.0 / len(self.env.track)
                self.env.tile_visited_count += 1

                # Lap is considered completed if enough % of the track was covered
                if (
                    tile.idx == 0
                    and self.env.tile_visited_count / len(self.env.track)
                    > self.lap_complete_percent
                ):
                    self.env.new_lap = True
        else:
            obj.tiles.remove(tile)


class MouseRacing(gym.Env, EzPickle):
    """
    ## Description
    The easiest control task to learn from pixels - a top-down
    racing environment. The generated track is random every episode.

    Some indicators are shown at the bottom of the window along with the
    state RGB buffer. From left to right: true speed, four ABS sensors,
    steering wheel position, and gyroscope.
    To play yourself (it's rather fast for humans), type:
    ```shell
    python gymnasium/envs/box2d/car_racing.py
    ```
    Remember: it's a powerful rear-wheel drive car - don't press the accelerator
    and turn at the same time.

    ## Action Space
    If continuous there are 3 actions :
    - 0: steering, -1 is full left, +1 is full right
    - 1: gas
    - 2: braking

    If discrete there are 5 actions:
    - 0: do nothing
    - 1: steer right
    - 2: steer left
    - 3: gas
    - 4: brake

    ## Observation Space

    A top-down 96x96 RGB image of the car and race track.

    ## Rewards
    The reward is -0.1 every frame and +1000/N for every track tile visited, where N is the total number of tiles
     visited in the track. For example, if you have finished in 732 frames, your reward is 1000 - 0.1*732 = 926.8 points.

    ## Starting State
    The car starts at rest in the center of the road.

    ## Episode Termination
    The episode finishes when all the tiles are visited. The car can also go outside the playfield -
     that is, far off the track, in which case it will receive -100 reward and die.

    ## Arguments

    ```python
    >>> import gymnasium as gym
    >>> env = gym.make("CarRacing-v3", render_mode="rgb_array", lap_complete_percent=0.95, domain_randomize=False, continuous=False)
    >>> env
    <TimeLimit<OrderEnforcing<PassiveEnvChecker<CarRacing<CarRacing-v3>>>>>

    ```

    * `lap_complete_percent=0.95` dictates the percentage of tiles that must be visited by
     the agent before a lap is considered complete.

    * `domain_randomize=False` enables the domain randomized variant of the environment.
     In this scenario, the background and track colours are different on every reset.

    * `continuous=True` specifies if the agent has continuous (true) or discrete (false) actions.
     See action space section for a description of each.

    ## Reset Arguments

    Passing the option `options["randomize"] = True` will change the current colour of the environment on demand.
    Correspondingly, passing the option `options["randomize"] = False` will not change the current colour of the environment.
    `domain_randomize` must be `True` on init for this argument to work.

    ```python
    >>> import gymnasium as gym
    >>> env = gym.make("CarRacing-v3", domain_randomize=True)

    # normal reset, this changes the colour scheme by default
    >>> obs, _ = env.reset()

    # reset with colour scheme change
    >>> randomize_obs, _ = env.reset(options={"randomize": True})

    # reset with no colour scheme change
    >>> non_random_obs, _ = env.reset(options={"randomize": False})

    ```

    ## Version History
    - v2: Change truncation to termination when finishing the lap (1.0.0)
    - v1: Change track completion logic and add domain randomization (0.24.0)
    - v0: Original version

    ## References
    - Chris Campbell (2014), http://www.iforce2d.net/b2dtut/top-down-car.

    ## Credits
    Created by Oleg Klimov
    """

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "state_pixels",
        ],
        "render_fps": FPS,
    }

    def __init__(
        self,
        render_mode: str = "rgb_array",
        verbose: bool = False,
        lap_complete_percent: float = 0.95,
        domain_randomize: bool = False,
        continuous: bool = True,
    ):
        EzPickle.__init__(
            self,
            render_mode,
            verbose,
            lap_complete_percent,
            domain_randomize,
            continuous,
        )
        self.continuous = continuous
        self.domain_randomize = domain_randomize
        self.lap_complete_percent = lap_complete_percent
        self._init_colors()

        self.contactListener_keepref = FrictionDetector(self, self.lap_complete_percent)
        self.world = Box2D.b2World((0, 0), contactListener=self.contactListener_keepref)
        self.screen: pygame.Surface = None
        self.surf = None
        self.clock = None
        self.isopen = True
        self.invisible_state_window = None
        self.invisible_video_window = None
        self.road = None
        self.car: Mouse = None
        self.reward = 0.0
        self.prev_reward = 0.0
        self.verbose = verbose
        self.new_lap = False
        self.fd_tile = fixtureDef(
            shape=polygonShape(vertices=[(0, 0), (1, 0), (1, -1), (0, -1)])
        )

        # This will throw a warning in tests/envs/test_envs in utils/env_checker.py as the space is not symmetric
        #   or normalised however this is not possible here so ignore
        if self.continuous:
    # Flattened continuous action space: [steer, gas, brake, detect_leak]
            self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )
        else:
            # Discrete action space: do nothing, right, left, gas, brake
            self.action_space = spaces.Discrete(5)


        self.observation_space = spaces.Box(
            low=0, high=255, shape=(STATE_H, STATE_W, 3), dtype=np.uint8
        )

        self.render_mode = render_mode
        self.tracks = []
        self.track = []
        self.info = None

    def _destroy(self):
        if not self.road:
            return
        for t in self.road:
            self.world.DestroyBody(t)
        for w in self.walls:
            self.world.DestroyBody(w)
        self.road = []
        assert self.car is not None
        self.car.destroy()

    def _init_colors(self):
        if self.domain_randomize:
            # domain randomize the bg and grass colour
            self.road_color = self.np_random.uniform(0, 210, size=3)

            self.bg_color = self.np_random.uniform(0, 210, size=3)

            self.grass_color = np.copy(self.bg_color)
            idx = self.np_random.integers(3)
            self.grass_color[idx] += 20
        else:
            # default colours
            self.road_color = np.array([102, 102, 102])
            self.bg_color = np.array([0,0,0])
            self.grass_color = np.array([0,0,0])

    def _reinit_colors(self, randomize):
        assert (
            self.domain_randomize
        ), "domain_randomize must be True to use this function."

        if randomize:
            # domain randomize the bg and grass colour
            self.road_color = self.np_random.uniform(0, 210, size=3)

            self.bg_color = self.np_random.uniform(0, 210, size=3)

            self.grass_color = np.copy(self.bg_color)
            idx = self.np_random.integers(3)
            self.grass_color[idx] += 20

    def reset(
        self,
        *,
        seed: int = None,
        options: dict = None,
    ):
        super().reset(seed=seed)
        self._destroy()
        self.world.contactListener_bug_workaround = FrictionDetector(
            self, self.lap_complete_percent
        )
        self.world.contactListener = self.world.contactListener_bug_workaround
        self.reward = 0.0
        self.prev_reward = 0.0
        self.tile_visited_count = 0
        self.t = 0.0
        self.new_lap = False
        self.road_poly = []

        if self.domain_randomize:
            randomize = True
            if isinstance(options, dict):
                if "randomize" in options:
                    randomize = options["randomize"]

            self._reinit_colors(randomize)

        while True:
            success = self._create_track()
            if success:
                break
            if self.verbose:
                print(
                    "retry to generate track (normal if there are not many"
                    "instances of this message)"
                )
        if (hasattr(self,"num_tracks")):
            init_x = self.track[0][0][2]
            init_y = self.track[0][0][3]
        else: 
            init_x = self.track[0][2]
            init_y = self.track[0][3]
        self.car = Mouse(self.world, 0, init_x, init_y) #to-do: set init_angle to be a parameter

        self.set_leaky_tiles(leak_probability=0.03)

        if self.render_mode == "human":
            self.render()
        return self.step(None)[0], {}

    def _create_track(self):
        CHECKPOINTS = 12

        # Create checkpoints
        checkpoints = []
        for c in range(CHECKPOINTS):
            noise = self.np_random.uniform(0, 2 * math.pi * 1 / CHECKPOINTS)
            alpha = 2 * math.pi * c / CHECKPOINTS + noise
            rad = self.np_random.uniform(TRACK_RAD / 3, TRACK_RAD)

            if c == 0:
                alpha = 0
                rad = 1.5 * TRACK_RAD
            if c == CHECKPOINTS - 1:
                alpha = 2 * math.pi * c / CHECKPOINTS
                self.start_alpha = 2 * math.pi * (-0.5) / CHECKPOINTS
                rad = 1.5 * TRACK_RAD

            checkpoints.append((alpha, rad * math.cos(alpha), rad * math.sin(alpha)))
        self.road = []

        # Go from one checkpoint to another to create track
        x, y, beta = 1.5 * TRACK_RAD, 0, 0
        dest_i = 0
        laps = 0
        track = []
        no_freeze = 2500
        visited_other_side = False
        while True:
            alpha = math.atan2(y, x)
            if visited_other_side and alpha > 0:
                laps += 1
                visited_other_side = False
            if alpha < 0:
                visited_other_side = True
                alpha += 2 * math.pi

            while True:  # Find destination from checkpoints
                failed = True

                while True:
                    dest_alpha, dest_x, dest_y = checkpoints[dest_i % len(checkpoints)]
                    if alpha <= dest_alpha:
                        failed = False
                        break
                    dest_i += 1
                    if dest_i % len(checkpoints) == 0:
                        break

                if not failed:
                    break

                alpha -= 2 * math.pi
                continue

            r1x = math.cos(beta)
            r1y = math.sin(beta)
            p1x = -r1y
            p1y = r1x
            dest_dx = dest_x - x  # vector towards destination
            dest_dy = dest_y - y
            # destination vector projected on rad:
            proj = r1x * dest_dx + r1y * dest_dy
            while beta - alpha > 1.5 * math.pi:
                beta -= 2 * math.pi
            while beta - alpha < -1.5 * math.pi:
                beta += 2 * math.pi
            prev_beta = beta
            proj *= SCALE
            if proj > 0.3:
                beta -= min(TRACK_TURN_RATE, abs(0.001 * proj))
            if proj < -0.3:
                beta += min(TRACK_TURN_RATE, abs(0.001 * proj))
            x += p1x * TRACK_DETAIL_STEP
            y += p1y * TRACK_DETAIL_STEP
            track.append((alpha, prev_beta * 0.5 + beta * 0.5, x, y))
            if laps > 4:
                break
            no_freeze -= 1
            if no_freeze == 0:
                break

        # Find closed loop range i1..i2, first loop should be ignored, second is OK
        i1, i2 = -1, -1
        i = len(track)
        while True:
            i -= 1
            if i == 0:
                return False  # Failed
            pass_through_start = (
                track[i][0] > self.start_alpha and track[i - 1][0] <= self.start_alpha
            )
            if pass_through_start and i2 == -1:
                i2 = i
            elif pass_through_start and i1 == -1:
                i1 = i
                break
        if self.verbose:
            print("Track generation: %i..%i -> %i-tiles track" % (i1, i2, i2 - i1))
        assert i1 != -1
        assert i2 != -1

        track = track[i1 : i2 - 1]

        first_beta = track[0][1]
        first_perp_x = math.cos(first_beta)
        first_perp_y = math.sin(first_beta)
        # Length of perpendicular jump to put together head and tail
        well_glued_together = np.sqrt(
            np.square(first_perp_x * (track[0][2] - track[-1][2]))
            + np.square(first_perp_y * (track[0][3] - track[-1][3]))
        )
        if well_glued_together > TRACK_DETAIL_STEP:
            return False

        # Create tiles (no red-white border blocks)
        self.wall_vertices_left, self.wall_vertices_right = [], []
        for i in range(len(track)):
            alpha1, beta1, x1, y1 = track[i]
            alpha2, beta2, x2, y2 = track[i - 1]
            road1_l = (
                x1 - TRACK_WIDTH * math.cos(beta1),
                y1 - TRACK_WIDTH * math.sin(beta1),
            )
            road1_r = (
                x1 + TRACK_WIDTH * math.cos(beta1),
                y1 + TRACK_WIDTH * math.sin(beta1),
            )
            road2_l = (
                x2 - TRACK_WIDTH * math.cos(beta2),
                y2 - TRACK_WIDTH * math.sin(beta2),
            )
            road2_r = (
                x2 + TRACK_WIDTH * math.cos(beta2),
                y2 + TRACK_WIDTH * math.sin(beta2),
            )
            # Use a slightly offset polygon for the wall (barrier), but do not draw kerbs
            kerb_offset = 1.0  # same as before, but not drawn
            road1_lP1 = (
                x1 - (TRACK_WIDTH + kerb_offset) * math.cos(beta1),
                y1 - (TRACK_WIDTH + kerb_offset) * math.sin(beta1),
            )
            road1_rP1 = (
                x1 + (TRACK_WIDTH + kerb_offset) * math.cos(beta1),
                y1 + (TRACK_WIDTH + kerb_offset) * math.sin(beta1),
            )
            road2_lP1 = (
                x2 - (TRACK_WIDTH + kerb_offset) * math.cos(beta2),
                y2 - (TRACK_WIDTH + kerb_offset) * math.sin(beta2),
            )
            road2_rP1 = (
                x2 + (TRACK_WIDTH + kerb_offset) * math.cos(beta2),
                y2 + (TRACK_WIDTH + kerb_offset) * math.sin(beta2),
            )
            self.wall_vertices_left.append([road1_lP1, road1_l, road2_l, road2_lP1])
            self.wall_vertices_right.append([road1_r, road1_rP1, road2_rP1, road2_r])
            vertices = [road1_l, road1_r, road2_r, road2_l]
            self.fd_tile.shape.vertices = vertices
            t = self.world.CreateStaticBody(fixtures=self.fd_tile)
            t.userData = t
            c = 0.01 * (i % 3) * 255
            t.color = self.road_color + c
            t.road_visited = False
            t.road_friction = 1.0
            t.idx = i
            t.fixtures[0].sensor = True
            self.road_poly.append(([road1_l, road1_r, road2_r, road2_l], t.color))
            self.road.append(t)
        self.track = track
        self._add_track_barriers()
        return True

    def _add_track_barriers(self):
        wall_thickness = 1.0/SCALE  # adjust as needed
        self.walls = []
        tile_radius = 60.0* SCALE # approximate tile size
        for i in range(len(self.wall_vertices_left)):
            # tile_pos = tile[-2:]  # Not needed for barrier placement
            # Create 2 barriers per tile (left and right)
            for j in range(2):
                wall_fixture = Box2D.b2FixtureDef(
                    shape=Box2D.b2PolygonShape(box=(2, wall_thickness)),
                    restitution=0.8  # Set bounce (0.0 = no bounce, 1.0 = very bouncy)
                )
                if j==0:
                    wall_fixture.shape.vertices = self.wall_vertices_left[i]
                elif j==1:
                    wall_fixture.shape.vertices = self.wall_vertices_right[i]
                wall = self.world.CreateStaticBody(
                    fixtures=wall_fixture,
                )
                wall.color = np.array([255, 0, 0], dtype=np.uint8)
                # Mark this body as a wall so contact listener can detect it
                wall.userData = wall
                wall.is_wall = True
                self.walls.append(wall)

    def set_leaky_tiles(self, leak_probability=0.03):
        """
        Randomly mark only the main road tiles (not red/white barriers) as 'leaky' based on a given probability.
        Only tiles in self.road are eligible.
        """
        self.leaky_tiles = {}
        for i, tile in enumerate(self.road):
            if random.random() < leak_probability:
                self.leaky_tiles[i] = {
                    "position": (tile.angle if hasattr(tile, 'angle') else 0, tile.angle if hasattr(tile, 'angle') else 0, tile.position[0], tile.position[1]),
                    "color": (0, 0, 255),
                    "leaky": True,
                    "fixed": False
                }

    def step(self, action):
        step_reward = 0
        assert self.car is not None
        if action is not None:
            if self.continuous:
                # Accept both array and dict for backward compatibility
                if isinstance(action, dict):
                    steer = float(action.get("steer", 0))
                    gas = float(action.get("gas", 0))
                    brake = float(action.get("brake", 0))
                    fix_leak = float(action.get("fix_leak", 0))
                else:
                    # fallback for array input
                    steer, gas, brake, fix_leak = action.astype(np.float64)
                self.car.steer(-steer)
                self.car.gas(gas)
                self.car.brake(brake)
            else:
                if not self.action_space.contains(action):
                    raise InvalidAction(
                        f"you passed the invalid action `{action}`. "
                        f"The supported action_space is `{self.action_space}`"
                    )
                self.car.steer(-0.6 * (action == 1) + 0.6 * (action == 2))
                self.car.gas(0.2 * (action == 3))
                self.car.brake(0.8 * (action == 4))

        # Missed-leak penalty: track if car was ever close to a leaky tile, and penalize if it moves away without fixing
        car_pos = self.car.hull.position
        info = {"leak_tile_stats": "tn"}  # true negative by default
        for idx, tile_info in self.leaky_tiles.items():
            if tile_info.get("fixed", False) or tile_info.get("missed", False):
                continue
            tile_pos = tile_info["position"][-2:]
            dist = np.linalg.norm(np.array(car_pos) - np.array(tile_pos))
            # Use a larger threshold for detection
            close_thresh = TRACK_WIDTH * 2.5
            if dist < close_thresh:
                self.leaky_tiles[idx]["was_close"] = True
            if tile_info.get("was_close", False) and dist > close_thresh:
                print(f"Missed leaky tile at index {idx} (position: {tile_pos})")
                step_reward -= 20 * LEAK_SCALE  # Penalty for missing a leaky tile (20)
                self.leaky_tiles[idx]["missed"] = True
                info["leak_tile_stats"] = "fn" # false negative

        # If the detect_leak action is triggered, try to detect a nearby leaky tile
        if self.continuous and action is not None:
            if isinstance(action, dict):
                detect_leak = float(action.get("detect_leak", 0))
            else:
                detect_leak = float(action[3]) if len(action) > 3 else 0
            if detect_leak > 0.5:
                car_pos = self.car.hull.position
                closest_idx = None
                closest_dist = float('inf')
                # Find the closest undetected leaky tile
                for idx, tile_info in self.leaky_tiles.items():
                    if not tile_info.get("fixed", False):
                        tile_pos = tile_info["position"][-2:]
                        dist = np.linalg.norm(np.array(car_pos) - np.array(tile_pos))
                        if dist < closest_dist:
                            closest_dist = dist
                            closest_idx = idx
                # If close enough, detect the leak and give reward
                if closest_idx is not None and closest_dist < TRACK_WIDTH * 2:
                    if not self.leaky_tiles[closest_idx].get("fixed", False):
                        self.leaky_tiles[closest_idx]["fixed"] = True
                        self.leaky_tiles[closest_idx]["color"] = (0, 255, 0)
                        step_reward += 50 * LEAK_SCALE  # Reward for detecting a leak (50)
                        self.leaky_tiles[closest_idx]["was_close"] = True
                        leak_detected = True
                        info["leak_tile_stats"] = "tp" # true positive
                else:
                    # If not close enough to any leaky tile, penalize for false detect or missed leak
                    missed_any = False
                    for idx, tile_info in self.leaky_tiles.items():
                        if not tile_info.get("fixed", False):
                            tile_pos = tile_info["position"][-2:]
                            dist = np.linalg.norm(np.array(car_pos) - np.array(tile_pos))
                            if dist < TRACK_WIDTH * 4:
                                if not tile_info.get("missed", False):
                                    step_reward -= 20 * LEAK_SCALE # Penalty for missing a leak (20)
                                    self.leaky_tiles[idx]["missed"] = True
                                    missed_any = True
                                    info["leak_tile_stats"] = "fn" # false negative
                    if not missed_any:
                        step_reward -= 2 * LEAK_SCALE  # Penalty for trying to detect where there is no leak (2)
                        info["leak_tile_stats"] = "fp" # false positive
                    leak_detected = False

        self.car.step(1.0 / FPS)
        self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)
        self.t += 1.0 / FPS

        self.state = self._render("state_pixels")

        terminated = False
        truncated = False
        if action is not None:  # First step without action, called from reset()
            self.reward -= 0.1
            self.car.fuel_spent = 0.0
            step_reward += self.reward - self.prev_reward
            self.prev_reward = self.reward
            if self.tile_visited_count == len(self.track) or self.new_lap:
                terminated = True
                info["lap_finished"] = True
            x, y = self.car.hull.position
            if abs(x) > PLAYFIELD or abs(y) > PLAYFIELD:
                terminated = True
                info["lap_finished"] = False
                step_reward = -100

        if self.render_mode == "human":
            self.render()
        return self.state, step_reward, terminated, truncated, info

    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return
        else:
            return self._render(self.render_mode)

    def _render(self, mode: str):
        assert mode in self.metadata["render_modes"]

        pygame.font.init()
        if self.screen is None and mode == "human":
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        if "t" not in self.__dict__:
            return  # reset() not called yet


        # Make the canvas larger to ensure all overlays are visible
        self.surf = pygame.Surface((WINDOW_W, WINDOW_H))

        assert self.car is not None
        # computing transformations
        angle = -self.car.hull.angle
        # Animating first second zoom.
        zoom = 0.1 * SCALE * max(1 - self.t, 0) + ZOOM * SCALE * min(self.t, 1)
        scroll_x = -(self.car.hull.position[0]) * zoom
        scroll_y = -(self.car.hull.position[1]) * zoom
        trans = pygame.math.Vector2((scroll_x, scroll_y)).rotate_rad(angle)
        trans = (WINDOW_W / 2 + trans[0], WINDOW_H / 4 + trans[1])

        self._render_road(zoom, trans, angle)
        self.car.draw(
            self.surf,
            zoom,
            trans,
            angle,
            mode not in ["state_pixels_list", "state_pixels"],
        )


        # Flip for display first (restore original order)
        self.surf = pygame.transform.flip(self.surf, False, True)

        # Draw overlays (score and indicators) AFTER flipping (original behavior)
        # Draw score at bottom left, just above the indicator bar
        font = pygame.font.Font(pygame.font.get_default_font(), 42)
        text = font.render("%04i" % self.reward, True, (255, 255, 255), (0, 0, 0))
        text_rect = text.get_rect()
        # Place score 10px from left, 5*indicator height from bottom
        indicator_h = WINDOW_H / 40.0
        text_rect.topleft = (10, WINDOW_H - 5 * indicator_h - text_rect.height - 5)
        self.surf.blit(text, text_rect)

        # Draw indicators bar at the bottom (after score, so both overlays are on top)
        self._render_indicators(WINDOW_W, WINDOW_H)

        if mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            assert self.screen is not None
            self.screen.fill(0)
            self.screen.blit(self.surf, (0, 0))
            pygame.display.flip()
        elif mode == "rgb_array":
            return self._create_image_array(self.surf, (VIDEO_W, VIDEO_H))
        elif mode == "state_pixels":
            return self._create_image_array(self.surf, (STATE_W, STATE_H))
        else:
            return self.isopen


    def _render_road(self, zoom, translation, angle):
        bounds = PLAYFIELD
        field = [
            (bounds, bounds),
            (bounds, -bounds),
            (-bounds, -bounds),
            (-bounds, bounds),
        ]

        # draw background
        self._draw_colored_polygon(
            self.surf, field, self.bg_color, zoom, translation, angle, clip=False
        )

        # draw grass patches
        grass = []
        for x in range(-20, 20, 2):
            for y in range(-20, 20, 2):
                grass.append(
                    [
                        (GRASS_DIM * x + GRASS_DIM, GRASS_DIM * y + 0),
                        (GRASS_DIM * x + 0, GRASS_DIM * y + 0),
                        (GRASS_DIM * x + 0, GRASS_DIM * y + GRASS_DIM),
                        (GRASS_DIM * x + GRASS_DIM, GRASS_DIM * y + GRASS_DIM),
                    ]
                )
        for poly in grass:
            self._draw_colored_polygon(
                self.surf, poly, self.grass_color, zoom, translation, angle
            )

        # draw road, highlighting leaky tiles in blue/green only for main road tiles
        leaky_indices = set(self.leaky_tiles.keys()) if hasattr(self, 'leaky_tiles') else set()
        num_main_road = len(self.road)
        for i, (poly, color) in enumerate(self.road_poly):
            poly = [(p[0], p[1]) for p in poly]
            if i < num_main_road and i in leaky_indices:
                tile_info = self.leaky_tiles[i]
                draw_color = tile_info["color"] if "color" in tile_info else (0, 0, 255)
                self._draw_colored_polygon(self.surf, poly, draw_color, zoom, translation, angle)
            else:
                color = [int(c) for c in color]
                self._draw_colored_polygon(self.surf, poly, color, zoom, translation, angle)

    def _render_indicators(self, W, H):
        s = W / 40.0
        h = H / 40.0
        color = (0, 0, 0)
        polygon = [(W, H), (W, H - 5 * h), (0, H - 5 * h), (0, H)]
        pygame.draw.polygon(self.surf, color=color, points=polygon)

        def vertical_ind(place, val):
            return [
                (place * s, H - (h + h * val)),
                ((place + 1) * s, H - (h + h * val)),
                ((place + 1) * s, H - h),
                ((place + 0) * s, H - h),
            ]

        def horiz_ind(place, val):
            return [
                ((place + 0) * s, H - 4 * h),
                ((place + val) * s, H - 4 * h),
                ((place + val) * s, H - 2 * h),
                ((place + 0) * s, H - 2 * h),
            ]

        assert self.car is not None
        true_speed = np.sqrt(
            np.square(self.car.hull.linearVelocity[0])
            + np.square(self.car.hull.linearVelocity[1])
        )

        # simple wrapper to render if the indicator value is above a threshold
        def render_if_min(value, points, color):
            if abs(value) > 1e-4:
                pygame.draw.polygon(self.surf, points=points, color=color)

        render_if_min(true_speed, vertical_ind(5, 0.02 * true_speed), (255, 255, 255))
        # ABS sensors
        render_if_min(
            self.car.wheels[0].omega,
            vertical_ind(7, 0.01 * self.car.wheels[0].omega),
            (0, 0, 255),
        )
        render_if_min(
            self.car.wheels[1].omega,
            vertical_ind(8, 0.01 * self.car.wheels[1].omega),
            (0, 0, 255),
        )
        render_if_min(
            self.car.wheels[2].omega,
            vertical_ind(9, 0.01 * self.car.wheels[2].omega),
            (51, 0, 255),
        )
        render_if_min(
            self.car.wheels[3].omega,
            vertical_ind(10, 0.01 * self.car.wheels[3].omega),
            (51, 0, 255),
        )

        render_if_min(
            self.car.wheels[0].joint.angle,
            horiz_ind(20, -10.0 * self.car.wheels[0].joint.angle),
            (0, 255, 0),
        )
        render_if_min(
            self.car.hull.angularVelocity,
            horiz_ind(30, -0.8 * self.car.hull.angularVelocity),
            (255, 0, 0),
        )

    def _draw_colored_polygon(
        self, surface, poly, color, zoom, translation, angle, clip=True
    ):
        poly = [pygame.math.Vector2(c).rotate_rad(angle) for c in poly]
        poly = [
            (c[0] * zoom + translation[0], c[1] * zoom + translation[1]) for c in poly
        ]
        # This checks if the polygon is out of bounds of the screen, and we skip drawing if so.
        # Instead of calculating exactly if the polygon and screen overlap,
        # we simply check if the polygon is in a larger bounding box whose dimension
        # is greater than the screen by MAX_SHAPE_DIM, which is the maximum
        # diagonal length of an environment object
        if not clip or any(
            (-MAX_SHAPE_DIM <= coord[0] <= WINDOW_W + MAX_SHAPE_DIM)
            and (-MAX_SHAPE_DIM <= coord[1] <= WINDOW_H + MAX_SHAPE_DIM)
            for coord in poly
        ):
            gfxdraw.aapolygon(self.surf, poly, color)
            gfxdraw.filled_polygon(self.surf, poly, color)

    def _create_image_array(self, screen, size):
        scaled_screen = pygame.transform.smoothscale(screen, size)
        return np.transpose(
            np.array(pygame.surfarray.pixels3d(scaled_screen)), axes=(1, 0, 2)
        )

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            self.isopen = False
            pygame.quit()

def register_input():
    global quit, restart
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                a[0] = -1.0
            if event.key == pygame.K_RIGHT:
                a[0] = +1.0
            if event.key == pygame.K_UP:
                a[1] = +1.0
            if event.key == pygame.K_DOWN:
                a[1] = -1.0  # reverse
                # Set brake to +0.8 if moving forward, else 0
                try:
                    car_vel = env.car.hull.linearVelocity[0] * np.cos(env.car.hull.angle) + env.car.hull.linearVelocity[1] * np.sin(env.car.hull.angle)
                    if car_vel > 0.1:
                        a[2] = +0.8  # brake if moving forward
                    else:
                        a[2] = 0.0   # no brake if already moving backward
                except Exception:
                    a[2] = 0.0
            if event.key == pygame.K_SPACE:
                a[2] = +1.0  # full brake
            if event.key == pygame.K_f:
                a[3] = 1.0  # detect_leak action
            if event.key == pygame.K_RETURN:
                restart = True
            if event.key == pygame.K_ESCAPE:
                quit = True

        if event.type == pygame.KEYUP:
            if event.key == pygame.K_LEFT:
                a[0] = 0
            if event.key == pygame.K_RIGHT:
                a[0] = 0
            if event.key == pygame.K_UP:
                a[1] = 0
            if event.key == pygame.K_DOWN:
                a[1] = 0
                a[2] = 0
            if event.key == pygame.K_SPACE:
                a[2] = 0
            if event.key == pygame.K_f:
                a[3] = 0.0  # release detect_leak

        if event.type == pygame.QUIT:
            quit = True

if __name__ == "__main__":
    a = np.array([0.0, 0.0, 0.0, 0.0])

    pygame.init()
    env = MouseRacing(render_mode="rgb_array")

    quit = False
    while not quit:
        env.reset()
        total_reward = 0.0
        steps = 0
        restart = False
        while True:
            register_input()
            s, r, terminated, truncated, info = env.step(a)
            total_reward += r
            # Always print reward for each step
            print(f"step {steps} reward {r:+0.2f} total_reward {total_reward:+0.2f}")
            steps += 1
            if terminated or truncated or restart or quit:
                break
    env.close()