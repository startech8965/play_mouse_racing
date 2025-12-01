import math
import random

import numpy as np

import gymnasium as gym
from gymnasium import spaces

from mouse_dynamics import Mouse
from MouseRacing import MouseRacing,SCALE, TRACK_TURN_RATE, TRACK_DETAIL_STEP, TRACK_WIDTH, PLAYFIELD, register_input, FrictionDetector, GRASS_DIM

from gymnasium.error import DependencyNotInstalled, InvalidAction
from gymnasium.utils import EzPickle

import Box2D
from Box2D.b2 import contactListener, fixtureDef, polygonShape

import pygame
from pygame import gfxdraw
from shapely.geometry import Polygon

SHOW_JOINTS = 0 
SHOW_BETA_PI_ANGLE = 0 


class MultiTrackMouseRacing(MouseRacing):
    def __init__(self, num_tracks: int = 1, num_lanes: int = 1, num_lanes_changes: int = 0, *args, **kwargs):
        #kwargs.setdefault("num_tracks", 2)
        super().__init__(*args, **kwargs)
        self.num_tracks = num_tracks
        self.num_lanes = num_lanes
        self.num_lanes_changes = num_lanes_changes
    
    @staticmethod
    def polygon_area(pts):
        # Shoelace formula for area of a polygon
        x = [p[0] for p in pts]
        y = [p[1] for p in pts]
        return 0.5 * abs(sum(x[i]*y[(i+1)%len(pts)] - x[(i+1)%len(pts)]*y[i] for i in range(len(pts))))
    
    

    def _get_track(self, num_checkpoints, track_rad=900/SCALE, x_bias=0, y_bias=0):

        #num_checkpoints = 12

        # Create checkpoints
        checkpoints = []
        for c in range(num_checkpoints):
            alpha = 2*math.pi*c/num_checkpoints + self.np_random.uniform(0, 2*math.pi*1/num_checkpoints)
            rad = self.np_random.uniform(track_rad/3, track_rad)
            if c==0:
                alpha = 0
                rad = 1.5*track_rad
            if c==num_checkpoints-1:
                alpha = 2*math.pi*c/num_checkpoints
                self.start_alpha = 2*math.pi*(-0.5)/num_checkpoints
                rad = 1.5*track_rad
            checkpoints.append( (alpha, rad*math.cos(alpha), rad*math.sin(alpha)) )

        #print "\n".join(str(h) for h in checkpoints)
        #self.road_poly = [ (    # uncomment this to see checkpoints
        #    [ (tx,ty) for a,tx,ty in checkpoints ],
        #    (0.7,0.7,0.9) ) ]

        # Go from one checkpoint to another to create track
        x, y, beta = 1.5*track_rad, 0, 0
        dest_i = 0
        laps = 0
        track = []
        no_freeze = 2500
        visited_other_side = False
        while 1:
            alpha = math.atan2(y, x)
            if visited_other_side and alpha > 0:
                laps += 1
                visited_other_side = False
            if alpha < 0:
                visited_other_side = True
                alpha += 2*math.pi
            while True: # Find destination from checkpoints
                failed = True
                while True:
                    dest_alpha, dest_x, dest_y = checkpoints[dest_i % len(checkpoints)]
                    if alpha <= dest_alpha:
                        failed = False
                        break
                    dest_i += 1
                    if dest_i % len(checkpoints) == 0: break
                if not failed: break
                alpha -= 2*math.pi
                continue
            r1x = math.cos(beta)
            r1y = math.sin(beta)
            p1x = -r1y
            p1y = r1x
            dest_dx = dest_x - x  # vector towards destination
            dest_dy = dest_y - y
            proj = r1x*dest_dx + r1y*dest_dy  # destination vector projected on rad
            while beta - alpha >  1.5*math.pi: beta -= 2*math.pi
            while beta - alpha < -1.5*math.pi: beta += 2*math.pi
            prev_beta = beta
            proj *= SCALE
            if proj >  0.3: beta -= min(TRACK_TURN_RATE, abs(0.001*proj))
            if proj < -0.3: beta += min(TRACK_TURN_RATE, abs(0.001*proj))
            x += p1x*TRACK_DETAIL_STEP
            y += p1y*TRACK_DETAIL_STEP
            track.append( (alpha,prev_beta*0.5 + beta*0.5,x,y) )
            if laps > 4: break
            no_freeze -= 1
            if no_freeze==0: break
        #print "\n".join([str(t) for t in enumerate(track)])

        # Find closed loop range i1..i2, first loop should be ignored, second is OK
        i1, i2 = -1, -1
        i = len(track)
        while True:
            i -= 1
            if i==0: return False  # Failed
            pass_through_start = track[i][0] > self.start_alpha and track[i-1][0] <= self.start_alpha
            if pass_through_start and i2==-1:
                i2 = i
            elif pass_through_start and i1==-1:
                i1 = i
                break
        if self.verbose > 0:
            print("Track generation: %i..%i -> %i-tiles track" % (i1, i2, i2-i1))
        assert i1!=-1
        assert i2!=-1

        track = track[i1:i2-1]

        first_beta = track[0][1]
        first_perp_x = math.cos(first_beta)
        first_perp_y = math.sin(first_beta)
        # Length of perpendicular jump to put together head and tail
        well_glued_together = np.sqrt(
            np.square( first_perp_x*(track[0][2] - track[-1][2]) ) +
            np.square( first_perp_y*(track[0][3] - track[-1][3]) ))
        if well_glued_together > TRACK_DETAIL_STEP:
            return False

        track = [[a,b,x+x_bias*2,y+y_bias*2] for a,b,x,y in track]
        track = [[track[i-1],track[i]] for i in range(len(track))]
        return track

    def _generate_track(self):
        tracks = []
        info = []
        for track_idx in range(self.num_tracks):
            track = self._get_track(10) #to-do: make num_checkpoints a parameter
            if not track:
                return False
            tracks.append(track)
            # For each segment in this track, create an info dict
            for seg_idx, seg in enumerate(track):
                info_dict = {
                    'track': track_idx,
                    'lanes': [True for _ in range(self.num_lanes)],  # or customize per lane
                    'start': seg_idx == 0,
                    'end': seg_idx == len(track) - 1,
                    # add other keys as needed
                }
                info.append(info_dict)
        self.tracks = tracks
        self.track = np.concatenate(tracks) if self.num_tracks > 1 else tracks[0]
        self.info = info  # <--- update self.info here
        return True
    
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
        for i, road_poly_i in enumerate(self.road_poly):
            if len(road_poly_i) == 4:
                poly, color, _, _ = road_poly_i
            else:
                poly, color = road_poly_i
            poly = [(p[0], p[1]) for p in poly]
            if i < num_main_road and i in leaky_indices:
                tile_info = self.leaky_tiles[i]
                draw_color = tile_info["color"] if "color" in tile_info else (0, 0, 255)
                self._draw_colored_polygon(self.surf, poly, draw_color, zoom, translation, angle)
            else:
                color = [int(c) for c in color]
                self._draw_colored_polygon(self.surf, poly, color, zoom, translation, angle)

    def _create_track(self):

        self.road = []
        self.road_poly = []

        Ok = self._generate_track()
        if Ok is False:
            return False
        
        # Create tiles
        for j in range(len(self.track)):
            alpha1, beta1, x1, y1 = self.track[j][1]
            alpha2, beta2, x2, y2 = self.track[j][0]
            
            # drawing angles of old config, the 
            # black line is the angle (NOT WORKING)
            if SHOW_BETA_PI_ANGLE:
                if self.track_lanes == None: self.track_lanes = []
                p1x = x1+np.cos(beta1)*0.2
                p1y = y1+np.sin(beta1)*0.2
                p2x = x1+np.cos(beta1)*0.2+np.cos(beta1+np.pi/2)*2
                p2y = y1+np.sin(beta1)*0.2+np.sin(beta1+np.pi/2)*2
                p3x = x1-np.cos(beta1)*0.2+np.cos(beta1+np.pi/2)*2
                p3y = y1-np.sin(beta1)*0.2+np.sin(beta1+np.pi/2)*2
                p4x = x1-np.cos(beta1)*0.2
                p4y = y1-np.sin(beta1)*0.2
                self.track_lanes.append([
                    [p1x,p1y],
                    [p2x,p2y],
                    [p3x,p3y],
                    [p4x,p4y]])

            for lane in range(self.num_lanes):
                if self.info[j]['lanes'][lane]:
                    
                    joint = False # to differentiate joints from normal tiles

                    r = 1- ((lane+1)%self.num_lanes)
                    l = 1- ((lane+2)%self.num_lanes)

                    # Get if it is the first or last
                    first = False # first of lane
                    last  = False # last tile of line

                    if self.info[j]['end'] == False and self.info[j]['start'] == False:

                        # Getting if first tile of lane
                        # if last tile was from the same lane
                        current_track = self.info[j]['track']
                        info_track = [info for info in self.info if info['track'] == current_track]
                        j_relative = [idx for idx, info in enumerate(info_track) if info is self.info[j]][0]        
                        
                        if info_track[j_relative-1]['track'] == info_track[j_relative]['track']:
                            # If last tile didnt exist
                            if info_track[j_relative-1]['lanes'][lane] == False:
                                first = True
                        if info_track[(j_relative+1)%len(info_track)]['track'] == info_track[j_relative]['track']:
                            # If last tile didnt exist
                            if info_track[(j_relative+1)%len(info_track)]['lanes'][lane] == False:
                                last = True

                    road1_l = (x1 - (1-last) *l*TRACK_WIDTH*math.cos(beta1), y1 - (1-last) *l*TRACK_WIDTH*math.sin(beta1))
                    road1_r = (x1 + (1-last) *r*TRACK_WIDTH*math.cos(beta1), y1 + (1-last) *r*TRACK_WIDTH*math.sin(beta1))
                    road2_l = (x2 - (1-first)*l*TRACK_WIDTH*math.cos(beta2), y2 - (1-first)*l*TRACK_WIDTH*math.sin(beta2))
                    road2_r = (x2 + (1-first)*r*TRACK_WIDTH*math.cos(beta2), y2 + (1-first)*r*TRACK_WIDTH*math.sin(beta2))

                    vertices = [road1_l, road1_r, road2_r, road2_l]

                    if self.info[j]['end'] == True or self.info[j]['start'] == True:

                        points = [] # to store the new points
                        p3 = [] # in order to save all points 3 to create joints
                        for i in [0,1]: # because there are two point to do
                            # Get the closest point to a line make by the continuing trend of the original road points, the points will be the points 
                            # under a radius r from line to avoid taking points far away in the other extreme of the track
                            # Remember the distance from a point p3 to a line p1,p2 is d = norm(np.cross(p2-p1, p1-p3))/norm(p2-p1)
                            # p1=(x1,y1)+sin/cos, p2=(x2,y2)+sin/cos, p3=points in poly
                            if self.info[j]['end']:
                                p1 = road1_l if i == 0 else road1_r
                                p2 = road2_l if i == 0 else road2_r
                            else:
                                p1 = road1_l if i == 0 else road1_r
                                p2 = road2_l if i == 0 else road2_r

                            if len(p3) == 0:
                                track0_info = [info for info in self.info if info['track'] == 0]
                                max_idx = sum(sum(lanes) for lanes in [info['lanes'] for info in track0_info]) # this will work because only seconday tracks have ends
                                p3_org = sum([x[0] for x in self.road_poly[:max_idx]], [])
                                if len(p3_org) == 0:
                                    continue  # or handle this case appropriately
                                p3_org = np.array(p3_org)
                                distance = TRACK_WIDTH * 2
                                not_too_close = np.where(np.linalg.norm(np.subtract(p3_org, p1), axis=1) >= TRACK_WIDTH / 3)[0]
                                while len(p3) == 0 and distance < PLAYFIELD:
                                    close = np.where(np.linalg.norm(np.subtract(p3_org, p1), axis=1) <= distance)[0]
                                    p3 = [p3_org[i] for i in np.intersect1d(close, not_too_close)]
                                    distance += TRACK_WIDTH

                            if len(p3) == 0:
                                raise RuntimeError('p3 lenght is zero')

                            d = (np.cross(np.subtract(p2,p1),np.subtract(p1,p3)))**2/np.linalg.norm(np.subtract(p2,p1))
                            points.append(p3[d.argmin()])

                        if len(points) < 2:
                            print(f"Warning: Not enough points found for joint at tile {j}, skipping this tile.")
                            continue  # Skip this tile and move to the next

                        if self.info[j]['start']:
                            vertices = [points[0], points[1], road1_r, road1_l]
                        else:
                            vertices = [road2_r, road2_l, points[0], points[1]]
                        joint = True

                        if self.info[j]['start']:
                            vertices = [points[0], points[1], road1_r, road1_l]
                        else:
                            vertices = [road2_r, road2_l, points[0], points[1]]
                        joint = True
                    
                    test_set = set([tuple(p) for p in vertices])
                    if len(test_set) >= 3 and self.polygon_area(vertices):
                        # TODO CHECK IF THIS AVOID THE ERROR OF ASSERTION COUNT >= 3
                        # TODO remove this try and find a way of really catching the errer
                        try:
                            self.fd_tile.shape.vertices = [tuple(np.asarray(v).tolist()) for v in vertices]
                            t = self.world.CreateStaticBody(fixtures=self.fd_tile)
                        except AssertionError as e:
                            print(str(e))
                            print(vertices)
                            return False
                        t.userData = t
                        i = 0
                        # changing the following i for j achives different colors when visited tiles
                        c = 0.01*(i%3) 
                        if joint and SHOW_JOINTS:
                            t.color = [1,1,1]
                        else:
                            #t.color = [ROAD_COLOR[0], ROAD_COLOR[1], ROAD_COLOR[2]] 
                            t.color = [self.road_color[0] + c, self.road_color[1] + c, self.road_color[2] + c]
                        t.road_visited = False
                        #t.typename = TILE_NAME
                        t.road_friction = 1.0
                        t.idx = j
                        t.lane = lane
                        t.fixtures[0].sensor = True
                        self.road_poly.append(( vertices, t.color, t.idx, t.lane ))
                        self.road.append(t)
                    else:
                        print("saved from error")
                        print(f"Skipped degenerate tile at index {j} (area too small or duplicate points)")
        self.wall_vertices_left, self.wall_vertices_right = [], []
        for i in range(len(self.track)):
            alpha1, beta1, x1, y1 = self.track[i][1]
            alpha2, beta2, x2, y2 = self.track[i][0]
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
            self._check_lane_overlaps(track_idx=0)  # Assuming single track for now

    def _check_lane_overlaps(self, track_idx):
        lane_polygons = []
        for info in self.road_poly:
            # Access the polygon directly from the first element of the tuple
            polygon = info[0]  # This gets the polygon coordinates [(x1,y1), (x2,y2), ...]
            lane_polygons.append(polygon)
        
        # Rest of your overlap checking logic
        if len(lane_polygons) > 1:
            overlap_found = False
            for i in range(len(lane_polygons)):
                for j in range(i + 1, len(lane_polygons)):
                    if self._polygons_overlap(lane_polygons[i], lane_polygons[j]):
                        overlap_found = True
                        break
                if overlap_found:
                    break

            if overlap_found:
                # Only walls on the far left and far right
                self._add_walls(track_idx, only_outer=True)
            else:
                # Walls on both sides of each lane
                self._add_walls(track_idx, only_outer=False)
    def _polygons_overlap(self, poly1, poly2):
        # Implement a method to check if two polygons overlap
        # This can be done using a library like Shapely or by implementing a custom algorithm

        polygon1 = Polygon(poly1)
        polygon2 = Polygon(poly2)
        if polygon1.intersects(polygon2):
            #if polygon1.intersection(polygon2).area > 0:
            return True
        return False

    def check_lane_overlaps(self):
        for track_idx in range(len(self.info)):
            lane_polygons = []
            for info in self.info:
                if info['track'] == track_idx:
                    lane_polygons.append(self.road_poly[info['idx']][0])  # Get the polygon for each lane

            if len(lane_polygons) > 1:
                overlap_found = False
                for i in range(len(lane_polygons)):
                    for j in range(i + 1, len(lane_polygons)):
                        if self._polygons_overlap(lane_polygons[i], lane_polygons[j]):
                            overlap_found = True
                            break
                    if overlap_found:
                        break

                if overlap_found:
                    # Only walls on the far left and far right
                    self._add_walls(track_idx, only_outer=True)
                else:
                    # Walls on both sides of each lane
                    self._add_walls(track_idx, only_outer=False)

    def _add_walls(self, track_idx, only_outer):
        # Implement the logic to add walls based on the track index and whether to add only outer walls
        pass

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
    env = MultiTrackMouseRacing(num_tracks = 2, num_lanes = 1, num_lanes_changes = 0, render_mode="human")

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