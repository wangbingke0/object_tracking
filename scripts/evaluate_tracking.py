#!/usr/bin/env python3
"""
Evaluate IMM-UKF-JPDA tracking results against nuScenes ground truth.

Computes standard MOT metrics:
  - MOTA (Multiple Object Tracking Accuracy)
  - MOTP (Multiple Object Tracking Precision)
  - ID switches, FP, FN, etc.

Usage:
    python3 evaluate_tracking.py \
        --result_file tracking_results.txt \
        --gt_file scene-0061.txt \
        --match_threshold 2.0
"""

import argparse
import numpy as np
from collections import defaultdict


def load_tracking_results(filepath):
    """Load tracking results from the C++ tracker output."""
    frames = []
    current_frame = None

    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            parts = line.split()
            tag = parts[0]

            if tag == 'FRAME':
                if current_frame is not None:
                    frames.append(current_frame)
                current_frame = {
                    'frame_idx': int(parts[1]),
                    'timestamp': float(parts[2]),
                    'tracks': [],
                    'dets': [],
                }
            elif tag == 'TRACK' and current_frame is not None:
                current_frame['tracks'].append({
                    'track_id': int(parts[1]),
                    'x': float(parts[2]),
                    'y': float(parts[3]),
                    'v': float(parts[4]),
                    'yaw': float(parts[5]),
                    'trackManage': int(parts[6]),
                    'isStatic': parts[7] == '1',
                })
            elif tag == 'DET' and current_frame is not None:
                current_frame['dets'].append({
                    'instance_token': parts[1],
                    'category': parts[2],
                    'cx': float(parts[3]),
                    'cy': float(parts[4]),
                })
            elif tag == 'ENDFRAME' and current_frame is not None:
                frames.append(current_frame)
                current_frame = None

    if current_frame is not None:
        frames.append(current_frame)

    return frames


def load_ground_truth(filepath):
    """Load ground truth from the preprocessed nuScenes file."""
    frames = []
    current_frame = None

    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            parts = line.split()
            tag = parts[0]

            if tag == 'FRAME':
                if current_frame is not None:
                    frames.append(current_frame)
                current_frame = {
                    'timestamp': float(parts[1]),
                    'objects': [],
                }
            elif tag == 'BOX' and current_frame is not None:
                x1, y1 = float(parts[1]), float(parts[2])
                x2, y2 = float(parts[3]), float(parts[4])
                x3, y3 = float(parts[5]), float(parts[6])
                x4, y4 = float(parts[7]), float(parts[8])
                cx = (x1 + x2 + x3 + x4) / 4.0
                cy = (y1 + y2 + y3 + y4) / 4.0

                current_frame['objects'].append({
                    'instance_token': parts[11],
                    'category': parts[12],
                    'cx': cx,
                    'cy': cy,
                })
            elif tag == 'ENDFRAME' and current_frame is not None:
                frames.append(current_frame)
                current_frame = None

    if current_frame is not None:
        frames.append(current_frame)

    return frames


def hungarian_match(tracks, gt_objects, threshold):
    """
    Match tracks to GT objects using greedy nearest-neighbor assignment.
    Returns list of (track_idx, gt_idx, distance) and unmatched indices.
    """
    if len(tracks) == 0 or len(gt_objects) == 0:
        return [], list(range(len(tracks))), list(range(len(gt_objects)))

    cost_matrix = np.zeros((len(tracks), len(gt_objects)))
    for i, t in enumerate(tracks):
        for j, g in enumerate(gt_objects):
            dx = t['x'] - g['cx']
            dy = t['y'] - g['cy']
            cost_matrix[i, j] = np.sqrt(dx*dx + dy*dy)

    matches = []
    matched_tracks = set()
    matched_gts = set()

    while True:
        if len(matched_tracks) == len(tracks) or len(matched_gts) == len(gt_objects):
            break

        min_val = np.inf
        min_i, min_j = -1, -1
        for i in range(len(tracks)):
            if i in matched_tracks:
                continue
            for j in range(len(gt_objects)):
                if j in matched_gts:
                    continue
                if cost_matrix[i, j] < min_val:
                    min_val = cost_matrix[i, j]
                    min_i, min_j = i, j

        if min_val > threshold or min_i < 0:
            break

        matches.append((min_i, min_j, min_val))
        matched_tracks.add(min_i)
        matched_gts.add(min_j)

    unmatched_tracks = [i for i in range(len(tracks)) if i not in matched_tracks]
    unmatched_gts = [j for j in range(len(gt_objects)) if j not in matched_gts]

    return matches, unmatched_tracks, unmatched_gts


def evaluate(result_frames, gt_frames, match_threshold=2.0, min_track_manage=5):
    """
    Compute MOT metrics.

    Args:
        result_frames: tracking results
        gt_frames: ground truth
        match_threshold: distance threshold for matching (meters)
        min_track_manage: minimum trackManage to consider a track confirmed
    """
    total_gt = 0
    total_fp = 0
    total_fn = 0
    total_id_sw = 0
    total_dist = 0.0
    total_matches = 0

    prev_match_map = {}

    n_frames = min(len(result_frames), len(gt_frames))

    for fi in range(n_frames):
        rf = result_frames[fi]
        gf = gt_frames[fi]

        confirmed_tracks = [t for t in rf['tracks'] if t['trackManage'] >= min_track_manage]
        gt_objs = gf['objects']

        total_gt += len(gt_objs)

        matches, unmatched_trk, unmatched_gt = hungarian_match(
            confirmed_tracks, gt_objs, match_threshold)

        fp = len(unmatched_trk)
        fn = len(unmatched_gt)

        id_sw = 0
        match_map = {}
        for ti, gi, dist in matches:
            track_id = confirmed_tracks[ti]['track_id']
            gt_id = gt_objs[gi]['instance_token']
            match_map[gt_id] = track_id

            if gt_id in prev_match_map and prev_match_map[gt_id] != track_id:
                id_sw += 1

            total_dist += dist
            total_matches += 1

        total_fp += fp
        total_fn += fn
        total_id_sw += id_sw
        prev_match_map = match_map

    mota = 1.0 - (total_fp + total_fn + total_id_sw) / max(total_gt, 1)
    motp = total_dist / max(total_matches, 1)
    precision = total_matches / max(total_matches + total_fp, 1)
    recall = total_matches / max(total_gt, 1)

    return {
        'num_frames': n_frames,
        'total_gt': total_gt,
        'total_matches': total_matches,
        'total_fp': total_fp,
        'total_fn': total_fn,
        'total_id_sw': total_id_sw,
        'MOTA': mota,
        'MOTP': motp,
        'Precision': precision,
        'Recall': recall,
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate tracking results')
    parser.add_argument('--result_file', type=str, required=True,
                        help='Path to tracking_results.txt')
    parser.add_argument('--gt_file', type=str, required=True,
                        help='Path to preprocessed nuScenes scene file (ground truth)')
    parser.add_argument('--match_threshold', type=float, default=2.0,
                        help='Distance threshold for matching tracks to GT (meters)')
    parser.add_argument('--min_track_manage', type=int, default=5,
                        help='Minimum trackManage value to consider a track confirmed')
    args = parser.parse_args()

    print("Loading tracking results...")
    result_frames = load_tracking_results(args.result_file)
    print(f"  Loaded {len(result_frames)} frames")

    print("Loading ground truth...")
    gt_frames = load_ground_truth(args.gt_file)
    print(f"  Loaded {len(gt_frames)} frames")

    print(f"\nEvaluating (match_threshold={args.match_threshold}m, "
          f"min_track_manage={args.min_track_manage})...\n")

    metrics = evaluate(result_frames, gt_frames,
                       args.match_threshold, args.min_track_manage)

    print("=" * 50)
    print("  MOT Evaluation Results")
    print("=" * 50)
    print(f"  Frames evaluated: {metrics['num_frames']}")
    print(f"  Total GT objects:  {metrics['total_gt']}")
    print(f"  Total matches:     {metrics['total_matches']}")
    print(f"  False Positives:   {metrics['total_fp']}")
    print(f"  False Negatives:   {metrics['total_fn']}")
    print(f"  ID Switches:       {metrics['total_id_sw']}")
    print("-" * 50)
    print(f"  MOTA:      {metrics['MOTA']:.4f}")
    print(f"  MOTP:      {metrics['MOTP']:.4f} m")
    print(f"  Precision: {metrics['Precision']:.4f}")
    print(f"  Recall:    {metrics['Recall']:.4f}")
    print("=" * 50)


if __name__ == '__main__':
    main()
