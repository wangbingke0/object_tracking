#!/usr/bin/env python3
"""
Preprocess nuScenes dataset for IMM-UKF-JPDA tracking evaluation.

Converts nuScenes 3D annotations into a simple text format that can be
read by the C++ nuscenes_tracking node.  Also outputs per-frame LiDAR
and camera metadata so the C++ node can load raw sensor data.

Supports interpolation between keyframes using LiDAR sweeps to increase
the effective frame rate (e.g. from 2Hz to 10Hz or 20Hz).

Usage:
    python3 preprocess_nuscenes.py --dataroot /path/to/nuscenes/v1.0-mini \
                                   --output_dir ./nuscenes_preprocessed

    # With interpolation (10 Hz effective frame rate):
    python3 preprocess_nuscenes.py --dataroot /path/to/nuscenes/v1.0-mini \
                                   --output_dir ./nuscenes_preprocessed \
                                   --interp_hz 10
"""

import json
import numpy as np
import os
import argparse

try:
    from nuscenes.map_expansion.map_api import NuScenesMap
except Exception:
    NuScenesMap = None


def quaternion_to_rotation_matrix(q):
    """Convert quaternion [w, x, y, z] to 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)]
    ])


def quaternion_to_yaw(q):
    """Extract yaw angle from quaternion [w, x, y, z]."""
    w, x, y, z = q
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return np.arctan2(siny_cosp, cosy_cosp)


def make_transform(rotation, translation):
    """Build a 4x4 homogeneous transform from quaternion + translation."""
    R = quaternion_to_rotation_matrix(rotation)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = translation
    return T


def slerp_yaw(yaw0, yaw1, t):
    """Interpolate between two yaw angles, handling wraparound."""
    diff = yaw1 - yaw0
    while diff > np.pi:  diff -= 2 * np.pi
    while diff < -np.pi: diff += 2 * np.pi
    return yaw0 + t * diff


def compute_xy_yaw_kappa(xy):
    """
    Given a polyline [(x, y), ...], return [(x, y, yaw, kappa), ...].
    """
    n = len(xy)
    if n < 2:
        return []

    pts = np.asarray(xy, dtype=float)
    yaw = np.zeros(n, dtype=float)
    kappa = np.zeros(n, dtype=float)

    # 1) tangent yaw for each point
    for i in range(n):
        if i == 0:
            dx = pts[1, 0] - pts[0, 0]
            dy = pts[1, 1] - pts[0, 1]
        elif i == n - 1:
            dx = pts[n - 1, 0] - pts[n - 2, 0]
            dy = pts[n - 1, 1] - pts[n - 2, 1]
        else:
            dx = pts[i + 1, 0] - pts[i - 1, 0]
            dy = pts[i + 1, 1] - pts[i - 1, 1]
        yaw[i] = np.arctan2(dy, dx)

    # 2) curvature kappa by central finite difference
    for i in range(1, n - 1):
        xm, ym = pts[i - 1, 0], pts[i - 1, 1]
        x0, y0 = pts[i, 0], pts[i, 1]
        xp, yp = pts[i + 1, 0], pts[i + 1, 1]

        x1 = 0.5 * (xp - xm)
        y1 = 0.5 * (yp - ym)
        x2 = xp - 2.0 * x0 + xm
        y2 = yp - 2.0 * y0 + ym
        den = np.power(x1 * x1 + y1 * y1, 1.5)
        if den > 1e-9:
            kappa[i] = (x1 * y2 - y1 * x2) / den

    if n >= 3:
        kappa[0] = kappa[1]
        kappa[n - 1] = kappa[n - 2]

    return [(float(pts[i, 0]), float(pts[i, 1]),
             float(yaw[i]), float(kappa[i])) for i in range(n)]


def compute_polyline_length_xyyk(xyyk):
    """Compute polyline length from [(x, y, yaw, kappa), ...]."""
    if len(xyyk) < 2:
        return 0.0
    length = 0.0
    for i in range(1, len(xyyk)):
        dx = xyyk[i][0] - xyyk[i - 1][0]
        dy = xyyk[i][1] - xyyk[i - 1][1]
        length += float(np.hypot(dx, dy))
    return float(length)


def compute_box_corners(translation, size, rotation):
    """
    Compute 3D bounding box corners from nuScenes annotation.

    Args:
        translation: [x, y, z] center of box in global frame
        size: [width, length, height]
        rotation: [w, x, y, z] quaternion

    Returns:
        corners_bottom: (4, 3) array of bottom face corners in global frame
        min_z: minimum z of the box
        max_z: maximum z of the box
    """
    w, l, h = size
    cx, cy, cz = translation

    # nuScenes convention: x=forward(length), y=left(width), z=up(height)
    corners_local_bottom = np.array([
        [-l/2, -w/2, -h/2],
        [ l/2, -w/2, -h/2],
        [ l/2,  w/2, -h/2],
        [-l/2,  w/2, -h/2],
    ])

    corners_local_top = np.array([
        [-l/2, -w/2, h/2],
        [ l/2, -w/2, h/2],
        [ l/2,  w/2, h/2],
        [-l/2,  w/2, h/2],
    ])

    R = quaternion_to_rotation_matrix(rotation)
    center = np.array([cx, cy, cz])

    corners_bottom = (R @ corners_local_bottom.T).T + center
    corners_top = (R @ corners_local_top.T).T + center

    min_z = float(np.min(corners_bottom[:, 2]))
    max_z = float(np.max(corners_top[:, 2]))

    return corners_bottom, min_z, max_z


def mat4_to_str(M):
    """Serialise a 4x4 matrix as 16 space-separated floats (row-major)."""
    return ' '.join(f'{v:.10f}' for v in M.flatten())


class NuScenesParser:
    """Lightweight nuScenes parser that reads JSON tables directly."""

    TRACK_CATEGORIES = [
        'vehicle.car', 'vehicle.truck', 'vehicle.bus.bendy',
        'vehicle.bus.rigid', 'vehicle.construction', 'vehicle.trailer',
        'vehicle.motorcycle', 'vehicle.bicycle',
        'human.pedestrian.adult', 'human.pedestrian.child',
        'human.pedestrian.construction_worker',
        'human.pedestrian.police_officer',
        'vehicle.emergency.ambulance', 'vehicle.emergency.police',
    ]

    def __init__(self, dataroot, version='v1.0-mini'):
        self.dataroot = dataroot
        self.meta_dir = os.path.join(dataroot, version)
        self.nusc_maps = {}

        # Map extraction caches (significantly reduces repeated map API calls).
        self._cache_grid_m = 1.0
        self._lane_centerline_cache = {}   # (map, resolution, token) -> (length_m, [(x,y,yaw,kappa), ...])
        self._lane_radius_cache = {}       # (map, qx, qy, radius, resolution, max_lanes) -> lanes
        self._road_polygon_cache = {}      # (map, layer, token) -> (is_intersection, [(x,y), ...])
        self._road_radius_cache = {}       # (map, qx, qy, radius, max_polys) -> road polygons

        print(f"Loading nuScenes tables from {self.meta_dir} ...")
        self.scenes = self._load('scene.json')
        self.samples = self._load('sample.json')
        self.sample_annotations = self._load('sample_annotation.json')
        self.sample_datas = self._load('sample_data.json')
        self.ego_poses = self._load('ego_pose.json')
        self.calibrated_sensors = self._load('calibrated_sensor.json')
        self.sensors = self._load('sensor.json')
        self.instances = self._load('instance.json')
        self.categories = self._load('category.json')
        self.logs = self._load('log.json')

        self._build_lookups()
        self._init_maps()
        print(f"Loaded {len(self.scenes)} scenes, {len(self.samples)} samples, "
              f"{len(self.sample_annotations)} annotations")

    def _load(self, filename):
        filepath = os.path.join(self.meta_dir, filename)
        with open(filepath) as f:
            return json.load(f)

    def _build_lookups(self):
        self.sample_by_token = {s['token']: s for s in self.samples}
        self.ego_pose_by_token = {e['token']: e for e in self.ego_poses}
        self.instance_by_token = {i['token']: i for i in self.instances}
        self.category_by_token = {c['token']: c for c in self.categories}
        self.log_by_token = {l['token']: l for l in self.logs}
        self.calibrated_sensor_by_token = {
            cs['token']: cs for cs in self.calibrated_sensors
        }
        self.sensor_by_token = {s['token']: s for s in self.sensors}
        self.sd_by_token = {sd['token']: sd for sd in self.sample_datas}

        self.annotations_by_sample = {}
        for anno in self.sample_annotations:
            st = anno['sample_token']
            self.annotations_by_sample.setdefault(st, []).append(anno)

        # Build sample_data lookup by (sample_token, channel)
        self.sd_by_sample_channel = {}
        for sd in self.sample_datas:
            if not sd['is_key_frame']:
                continue
            cs = self.calibrated_sensor_by_token[sd['calibrated_sensor_token']]
            sensor = self.sensor_by_token[cs['sensor_token']]
            channel = sensor['channel']
            self.sd_by_sample_channel[(sd['sample_token'], channel)] = sd

        self.ego_pose_by_sample = {}
        for key, sd in self.sd_by_sample_channel.items():
            sample_token, channel = key
            if channel == 'LIDAR_TOP':
                self.ego_pose_by_sample[sample_token] = \
                    self.ego_pose_by_token[sd['ego_pose_token']]

    def _init_maps(self):
        """Load map-expansion files for all locations present in this split."""
        if NuScenesMap is None:
            print("[WARN] nuscenes-devkit map API not available; skipping map export.")
            return

        map_root = os.path.join(self.dataroot, 'maps', 'expansion')
        if not os.path.isdir(map_root):
            print(f"[WARN] map expansion directory not found: {map_root}")
            return

        map_names = sorted({
            self.log_by_token[s['log_token']]['location']
            for s in self.scenes if s['log_token'] in self.log_by_token
        })
        for map_name in map_names:
            map_file = os.path.join(map_root, f'{map_name}.json')
            if not os.path.isfile(map_file):
                print(f"[WARN] missing map file for {map_name}: {map_file}")
                continue
            try:
                self.nusc_maps[map_name] = NuScenesMap(
                    dataroot=self.dataroot, map_name=map_name)
                print(f"Loaded map: {map_name}")
            except Exception as e:
                print(f"[WARN] failed to load map {map_name}: {e}")

    def _quantize_xy(self, x, y):
        q = max(self._cache_grid_m, 1e-3)
        return int(round(x / q)), int(round(y / q))

    def _extract_lanes(self, map_name, ego_x, ego_y, radius_m=55.0,
                       resolution_m=1.0, max_lanes=120):
        """
        Extract nearby lane/lane_connector centerlines around ego position.
        Returns a list of (token, length_m, [(x, y, yaw, kappa), ...]).
        """
        nusc_map = self.nusc_maps.get(map_name)
        if nusc_map is None:
            return []

        qx, qy = self._quantize_xy(ego_x, ego_y)
        radius_key = (
            map_name, qx, qy,
            float(radius_m), float(resolution_m), int(max_lanes)
        )
        cached_lanes = self._lane_radius_cache.get(radius_key)
        if cached_lanes is not None:
            return cached_lanes

        records = nusc_map.get_records_in_radius(
            ego_x, ego_y, radius_m, ['lane', 'lane_connector'])
        lane_tokens = records.get('lane', []) + records.get('lane_connector', [])
        lane_tokens = list(dict.fromkeys(lane_tokens))
        if not lane_tokens:
            return []
        if len(lane_tokens) > max_lanes:
            lane_tokens = lane_tokens[:max_lanes]

        lanes = []
        for token in lane_tokens:
            centerline_key = (map_name, float(resolution_m), token)
            lane_data = self._lane_centerline_cache.get(centerline_key)
            if lane_data is None:
                centerline = []
                try:
                    lane_points = nusc_map.discretize_lanes([token], resolution_m)
                    points = lane_points.get(token)
                except Exception:
                    points = None

                if points is not None and len(points) >= 2:
                    xy = []
                    for p in points:
                        if len(p) < 2:
                            continue
                        xy.append((float(p[0]), float(p[1])))
                    if len(xy) >= 2:
                        centerline = compute_xy_yaw_kappa(xy)
                lane_data = (compute_polyline_length_xyyk(centerline), centerline)
                self._lane_centerline_cache[centerline_key] = lane_data

            length_m, centerline = lane_data
            if len(centerline) >= 2:
                lanes.append((token, length_m, centerline))

        self._lane_radius_cache[radius_key] = lanes
        return lanes

    @staticmethod
    def _polygon_to_xy(poly):
        """Convert a shapely polygon to a list of (x, y) exterior points."""
        if poly is None or not hasattr(poly, 'exterior') or poly.exterior is None:
            return []
        coords = list(poly.exterior.coords)
        xy = []
        for p in coords:
            if len(p) < 2:
                continue
            xy.append((float(p[0]), float(p[1])))
        if len(xy) >= 2 and np.hypot(xy[0][0] - xy[-1][0], xy[0][1] - xy[-1][1]) < 1e-3:
            xy = xy[:-1]
        return xy

    def _extract_road_polygons(self, map_name, ego_x, ego_y, radius_m=65.0,
                               max_polys=240):
        """
        Extract nearby road_segment / road_block polygons around ego position.
        Returns a list of (tag, token, is_intersection, [(x, y), ...]).
        """
        nusc_map = self.nusc_maps.get(map_name)
        if nusc_map is None:
            return []

        qx, qy = self._quantize_xy(ego_x, ego_y)
        radius_key = (map_name, qx, qy, float(radius_m), int(max_polys))
        cached = self._road_radius_cache.get(radius_key)
        if cached is not None:
            return cached

        records = nusc_map.get_records_in_radius(
            ego_x, ego_y, radius_m, ['road_segment', 'road_block'])
        seg_tokens = list(dict.fromkeys(records.get('road_segment', [])))
        blk_tokens = list(dict.fromkeys(records.get('road_block', [])))

        out = []

        for tok in seg_tokens:
            if len(out) >= max_polys:
                break
            poly_key = (map_name, 'road_segment', tok)
            cached_poly = self._road_polygon_cache.get(poly_key)
            if cached_poly is None:
                is_intersection = 0
                xy = []
                rec = nusc_map.get('road_segment', tok)
                poly_tok = rec.get('polygon_token')
                if poly_tok:
                    try:
                        poly = nusc_map.extract_polygon(poly_tok)
                        xy = self._polygon_to_xy(poly)
                    except Exception:
                        xy = []
                if len(xy) >= 3:
                    is_intersection = 1 if rec.get('is_intersection', False) else 0
                self._road_polygon_cache[poly_key] = (is_intersection, xy)
                cached_poly = self._road_polygon_cache[poly_key]

            is_intersection, xy = cached_poly
            if len(xy) < 3:
                continue
            out.append(('ROADSEG', tok, is_intersection, xy))

        for tok in blk_tokens:
            if len(out) >= max_polys:
                break
            poly_key = (map_name, 'road_block', tok)
            cached_poly = self._road_polygon_cache.get(poly_key)
            if cached_poly is None:
                xy = []
                rec = nusc_map.get('road_block', tok)
                poly_tok = rec.get('polygon_token')
                if poly_tok:
                    try:
                        poly = nusc_map.extract_polygon(poly_tok)
                        xy = self._polygon_to_xy(poly)
                    except Exception:
                        xy = []
                self._road_polygon_cache[poly_key] = (0, xy)
                cached_poly = self._road_polygon_cache[poly_key]

            _, xy = cached_poly
            if len(xy) < 3:
                continue
            out.append(('ROADBLOCK', tok, 0, xy))

        self._road_radius_cache[radius_key] = out
        return out

    def get_scene_samples(self, scene):
        """Get all samples in a scene, ordered chronologically."""
        samples = []
        token = scene['first_sample_token']
        while token:
            sample = self.sample_by_token[token]
            samples.append(sample)
            token = sample['next'] if sample['next'] else None
        return samples

    def get_category_name(self, instance_token):
        instance = self.instance_by_token[instance_token]
        return self.category_by_token[instance['category_token']]['name']

    def _get_sensor_to_global(self, sample_data_entry):
        """Compute the 4x4 sensor-to-global transform for a sample_data record."""
        cs = self.calibrated_sensor_by_token[
            sample_data_entry['calibrated_sensor_token']]
        ego = self.ego_pose_by_token[sample_data_entry['ego_pose_token']]
        T_sensor_ego = make_transform(cs['rotation'], cs['translation'])
        T_ego_global = make_transform(ego['rotation'], ego['translation'])
        return T_ego_global @ T_sensor_ego

    def _get_camera_intrinsic(self, sample_data_entry):
        """Return (fx, fy, cx, cy) from the calibrated sensor intrinsic matrix."""
        cs = self.calibrated_sensor_by_token[
            sample_data_entry['calibrated_sensor_token']]
        K = np.array(cs['camera_intrinsic'])
        return K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    def get_sweeps_between(self, sd_token_start, sd_token_end):
        """Get all sweep sample_data entries between two keyframes (exclusive)."""
        sweeps = []
        token = self.sd_by_token[sd_token_start].get('next')
        while token and token != sd_token_end:
            sd = self.sd_by_token.get(token)
            if sd is None:
                break
            if not sd['is_key_frame']:
                sweeps.append(sd)
            token = sd.get('next')
        return sweeps

    def _find_nearest_sweep(self, sweeps, target_ts):
        """Find the sweep closest to target_ts (in seconds)."""
        best, best_dt = None, 1e9
        for sw in sweeps:
            dt = abs(sw['timestamp'] / 1e6 - target_ts)
            if dt < best_dt:
                best_dt = dt
                best = sw
        return best

    def _interpolate_annotation(self, anno0, anno1, t_frac):
        """Linearly interpolate between two annotations of the same instance."""
        tr0 = np.array(anno0['translation'])
        tr1 = np.array(anno1['translation'])
        tr = tr0 + t_frac * (tr1 - tr0)

        sz0 = np.array(anno0['size'])
        sz1 = np.array(anno1['size'])
        sz = sz0 + t_frac * (sz1 - sz0)

        yaw0 = quaternion_to_yaw(anno0['rotation'])
        yaw1 = quaternion_to_yaw(anno1['rotation'])
        yaw = slerp_yaw(yaw0, yaw1, t_frac)

        q_interp = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]

        return list(tr), list(sz), q_interp

    def _write_frame(self, f, timestamp, ego_x, ego_y, ego_yaw,
                     lidar_sd, camera_sds, annotations, lanes, road_polygons):
        """Write a single frame (keyframe or interpolated) to the output file."""
        f.write(f"FRAME {timestamp:.6f} {ego_x:.6f} {ego_y:.6f} {ego_yaw:.6f}\n")

        if lidar_sd:
            T_lidar_global = self._get_sensor_to_global(lidar_sd)
            f.write(f"LIDAR {lidar_sd['filename']} "
                    f"{mat4_to_str(T_lidar_global)}\n")

        if camera_sds:
            for cam_channel, cam_sd in camera_sds.items():
                T_cam_global = self._get_sensor_to_global(cam_sd)
                T_global_cam = np.linalg.inv(T_cam_global)
                fx, fy, cx, cy = self._get_camera_intrinsic(cam_sd)
                f.write(f"CAMERA {cam_channel} {cam_sd['filename']} "
                        f"{fx:.6f} {fy:.6f} {cx:.6f} {cy:.6f} "
                        f"{mat4_to_str(T_global_cam)}\n")

        frame_boxes = 0
        for anno_info in annotations:
            translation = anno_info['translation']
            size = anno_info['size']
            rotation = anno_info['rotation']
            instance_token = anno_info['instance_token']
            category = anno_info['category']

            corners, min_z, max_z = compute_box_corners(
                translation, size, rotation)

            f.write(
                f"BOX "
                f"{corners[0,0]:.6f} {corners[0,1]:.6f} "
                f"{corners[1,0]:.6f} {corners[1,1]:.6f} "
                f"{corners[2,0]:.6f} {corners[2,1]:.6f} "
                f"{corners[3,0]:.6f} {corners[3,1]:.6f} "
                f"{min_z:.6f} {max_z:.6f} "
                f"{instance_token} {category}\n"
            )
            frame_boxes += 1

        for lane_token, lane_length_m, lane_pts in lanes:
            flat = ' '.join(
                f'{x:.6f} {y:.6f} {yaw:.6f} {kappa:.6f}'
                for x, y, yaw, kappa in lane_pts)
            f.write(f"LANE {lane_token} {len(lane_pts)} {lane_length_m:.6f} {flat}\n")

        for tag, poly_token, is_intersection, poly_pts in road_polygons:
            flat = ' '.join(f'{x:.6f} {y:.6f}' for x, y in poly_pts)
            if tag == 'ROADSEG':
                f.write(f"ROADSEG {poly_token} {is_intersection} {len(poly_pts)} {flat}\n")
            else:
                f.write(f"ROADBLOCK {poly_token} {len(poly_pts)} {flat}\n")

        f.write("ENDFRAME\n\n")
        return frame_boxes

    def process_scene(self, scene, output_file, track_categories=None,
                      interp_hz=0):
        """
        Process one scene and write preprocessed data to a text file.

        Args:
            interp_hz: If > 0, interpolate annotations between keyframes at
                       this rate (Hz). Set to 10 or 20 for higher frame rate.
                       If 0, only output keyframes (original 2Hz).
        """
        if track_categories is None:
            track_categories = self.TRACK_CATEGORIES

        samples = self.get_scene_samples(scene)
        effective_hz = interp_hz if interp_hz > 0 else 2
        map_name = self.log_by_token[scene['log_token']]['location']

        with open(output_file, 'w') as f:
            f.write(f"# Scene: {scene['name']}\n")
            f.write(f"# Description: {scene['description']}\n")
            f.write(f"# Num keyframe samples: {len(samples)}\n")
            f.write(f"# Effective frame rate: ~{effective_hz} Hz"
                    f"{' (interpolated)' if interp_hz > 0 else ''}\n")
            f.write("# Format:\n")
            f.write("#   FRAME timestamp ego_x ego_y ego_yaw\n")
            f.write("#   LIDAR <rel_path> <16 floats: 4x4 sensor_to_global>\n")
            f.write("#   CAMERA <ch> <rel_path> <fx fy cx cy>"
                    " <16 floats: 4x4 global_to_sensor>\n")
            f.write("#   BOX x1 y1 ... minZ maxZ instance_token category\n")
            f.write("#   LANE lane_token n_points length_m x1 y1 yaw1 kappa1 x2 y2 yaw2 kappa2 ...\n")
            f.write("#   ROADSEG token is_intersection n_points x1 y1 ...\n")
            f.write("#   ROADBLOCK token n_points x1 y1 ...\n")
            f.write("#   ENDFRAME\n\n")

            total_boxes = 0
            total_frames = 0

            for si, sample in enumerate(samples):
                ego = self.ego_pose_by_sample.get(sample['token'])
                if ego is None:
                    continue

                ego_x, ego_y, _ = ego['translation']
                ego_yaw = quaternion_to_yaw(ego['rotation'])
                timestamp = sample['timestamp'] / 1e6

                # Get sensor data for this keyframe
                lidar_sd = self.sd_by_sample_channel.get(
                    (sample['token'], 'LIDAR_TOP'))
                camera_sds = {}
                for ch in ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
                           'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']:
                    csd = self.sd_by_sample_channel.get((sample['token'], ch))
                    if csd:
                        camera_sds[ch] = csd

                # Get filtered annotations for this keyframe
                raw_annos = self.annotations_by_sample.get(
                    sample['token'], [])
                kf_annotations = []
                for anno in raw_annos:
                    cat = self.get_category_name(anno['instance_token'])
                    if not any(cat.startswith(tc) for tc in track_categories):
                        continue
                    if anno['num_lidar_pts'] < 1:
                        continue
                    kf_annotations.append({
                        'translation': anno['translation'],
                        'size': anno['size'],
                        'rotation': anno['rotation'],
                        'instance_token': anno['instance_token'],
                        'category': cat,
                    })

                # Write keyframe
                lanes = self._extract_lanes(map_name, ego_x, ego_y)
                road_polygons = self._extract_road_polygons(map_name, ego_x, ego_y)
                nb = self._write_frame(
                    f, timestamp, ego_x, ego_y, ego_yaw,
                    lidar_sd, camera_sds, kf_annotations, lanes, road_polygons)
                total_boxes += nb
                total_frames += 1

                # --- Interpolation between this keyframe and next ---
                if interp_hz <= 0 or si >= len(samples) - 1:
                    continue

                next_sample = samples[si + 1]
                next_ego = self.ego_pose_by_sample.get(next_sample['token'])
                if next_ego is None:
                    continue

                next_timestamp = next_sample['timestamp'] / 1e6
                next_ego_x, next_ego_y, _ = next_ego['translation']
                next_ego_yaw = quaternion_to_yaw(next_ego['rotation'])

                next_raw_annos = self.annotations_by_sample.get(
                    next_sample['token'], [])
                next_kf_by_inst = {}
                for anno in next_raw_annos:
                    cat = self.get_category_name(anno['instance_token'])
                    if not any(cat.startswith(tc) for tc in track_categories):
                        continue
                    next_kf_by_inst[anno['instance_token']] = {
                        'translation': anno['translation'],
                        'size': anno['size'],
                        'rotation': anno['rotation'],
                        'instance_token': anno['instance_token'],
                        'category': cat,
                    }

                curr_kf_by_inst = {a['instance_token']: a
                                   for a in kf_annotations}

                # Get LiDAR sweeps between keyframes
                next_lidar_sd = self.sd_by_sample_channel.get(
                    (next_sample['token'], 'LIDAR_TOP'))
                lidar_sweeps = []
                if lidar_sd and next_lidar_sd:
                    lidar_sweeps = self.get_sweeps_between(
                        lidar_sd['token'], next_lidar_sd['token'])

                # Get camera sweeps between keyframes (per channel)
                cam_sweeps = {}
                for ch in ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
                           'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']:
                    curr_cam_sd = self.sd_by_sample_channel.get(
                        (sample['token'], ch))
                    next_cam_sd = self.sd_by_sample_channel.get(
                        (next_sample['token'], ch))
                    if curr_cam_sd and next_cam_sd:
                        cam_sweeps[ch] = self.get_sweeps_between(
                            curr_cam_sd['token'], next_cam_sd['token'])

                dt_total = next_timestamp - timestamp
                if dt_total <= 0:
                    continue
                interp_dt = 1.0 / interp_hz
                n_interp = int(dt_total / interp_dt)

                for ki in range(1, n_interp):
                    t_frac = ki / n_interp
                    interp_ts = timestamp + t_frac * dt_total

                    ie_x = ego_x + t_frac * (next_ego_x - ego_x)
                    ie_y = ego_y + t_frac * (next_ego_y - ego_y)
                    ie_yaw = slerp_yaw(ego_yaw, next_ego_yaw, t_frac)

                    best_lidar = self._find_nearest_sweep(
                        lidar_sweeps, interp_ts)

                    # Find nearest camera sweep for each channel
                    interp_cam_sds = {}
                    for ch, ch_sweeps in cam_sweeps.items():
                        best_cam = self._find_nearest_sweep(
                            ch_sweeps, interp_ts)
                        if best_cam:
                            interp_cam_sds[ch] = best_cam

                    interp_annos = []
                    for inst_tok, a0 in curr_kf_by_inst.items():
                        a1 = next_kf_by_inst.get(inst_tok)
                        if a1 is None:
                            continue
                        tr, sz, rot = self._interpolate_annotation(
                            a0, a1, t_frac)
                        interp_annos.append({
                            'translation': tr,
                            'size': sz,
                            'rotation': rot,
                            'instance_token': inst_tok,
                            'category': a0['category'],
                        })

                    interp_road_polygons = self._extract_road_polygons(
                        map_name, ie_x, ie_y)
                    nb = self._write_frame(
                        f, interp_ts, ie_x, ie_y, ie_yaw,
                        best_lidar,
                        interp_cam_sds if interp_cam_sds else None,
                        interp_annos,
                        self._extract_lanes(map_name, ie_x, ie_y),
                        interp_road_polygons)
                    total_boxes += nb
                    total_frames += 1

        print(f"  Scene {scene['name']}: {total_frames} frames "
              f"({len(samples)} keyframes"
              f"{f' + {total_frames - len(samples)} interpolated' if interp_hz > 0 else ''})"
              f", {total_boxes} total detections -> {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Preprocess nuScenes dataset for IMM-UKF-JPDA tracking')
    parser.add_argument('--dataroot', type=str, required=True,
                        help='Path to nuScenes dataset root')
    parser.add_argument('--version', type=str, default='v1.0-mini',
                        help='nuScenes version (default: v1.0-mini)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory')
    parser.add_argument('--scene', type=str, default=None,
                        help='Process only this scene (e.g. scene-0061)')
    parser.add_argument('--min_lidar_pts', type=int, default=1,
                        help='Minimum lidar points per annotation (default: 1)')
    parser.add_argument('--interp_hz', type=int, default=0,
                        help='Interpolation frame rate in Hz (0=off, 10 or 20 recommended)')
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(
            os.path.dirname(args.dataroot.rstrip('/')), 'nuscenes_preprocessed')

    os.makedirs(args.output_dir, exist_ok=True)

    ns = NuScenesParser(args.dataroot, args.version)

    for scene in ns.scenes:
        if args.scene and scene['name'] != args.scene:
            continue
        output_file = os.path.join(args.output_dir, f"{scene['name']}.txt")
        ns.process_scene(scene, output_file, interp_hz=args.interp_hz)

    print(f"\nDone. Preprocessed files saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
