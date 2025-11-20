from __future__ import annotations

import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from scipy.spatial import KDTree


@dataclass
class PlannerConfig:
    beats_per_shot: List[int]
    ease: str
    lateral_margin_ratio: float
    zoom_margin_ratio: float
    orbit_altitude_ratio: float
    dolly_altitude_ratio: float
    min_duration: float
    max_duration: float
    safety_margin_ratio: float
    clearance_ratio: float
    shot_mix: Dict[str, float] | None = None


class CameraPlanner:
    """Turns beats + scene cues into timed camera shots."""

    def __init__(
        self,
        scene_info: Dict,
        beat_info: Dict,
        config: PlannerConfig,
    ) -> None:
        self.scene = scene_info
        self.beats = beat_info
        self.config = config
        self.bounds_center = np.array(scene_info["bounds"]["center"])
        self.extent = np.array(scene_info["bounds"]["extent"])
        self.axes = np.array(scene_info["principal_axes"])
        self.vertical_axis_index = int(scene_info.get("vertical_axis_index", 1))
        self.clusters = scene_info.get("clusters", [])
        safe_bounds = scene_info.get("safe_bounds") or scene_info["bounds"]
        self.safe_min = np.array(safe_bounds.get("min", scene_info["bounds"]["min"]))
        self.safe_max = np.array(safe_bounds.get("max", scene_info["bounds"]["max"]))
        self.safe_extent = self.safe_max - self.safe_min
        self.safe_center = (self.safe_min + self.safe_max) / 2.0
        tight_bounds = scene_info.get("tight_bounds") or {}
        tight_min = np.array(tight_bounds.get("min", self.safe_min.tolist()))
        tight_max = np.array(tight_bounds.get("max", self.safe_max.tolist()))
        env_info = scene_info.get("environment") or {}
        env_type = env_info.get("type", "ambiguous")
        env_conf = float(env_info.get("confidence", 0.0))
        env_metrics = env_info.get("metrics", {})
        # Relaxed indoor check to ensure we apply enclosure logic even for lower confidence scenes
        self.is_indoor = env_type == "indoor"
        self.is_outdoor = env_type == "outdoor" and env_conf >= 0.5
        self.floor_height = float(env_metrics.get("floor_height", self.safe_min[1]))
        self.ceiling_height = float(env_metrics.get("ceiling_height", self.safe_max[1]))
        self.vertical_span = max(self.ceiling_height - self.floor_height, 1e-3)
        clamp_min = self.safe_min.copy()
        clamp_max = self.safe_max.copy()
        if self.is_indoor:
            clamp_min = np.maximum(clamp_min, tight_min)
            clamp_max = np.minimum(clamp_max, tight_max)
            padding = 0.08 * np.maximum(clamp_max - clamp_min, 1e-4)
            clamp_min = np.minimum(clamp_max - 1e-4, clamp_min + padding)
            clamp_max = np.maximum(clamp_min + 1e-4, clamp_max - padding)
        floor_buffer = 0.05 * self.vertical_span
        ceil_buffer = (0.08 if self.is_indoor else 0.04) * self.vertical_span
        clamp_min[1] = max(clamp_min[1], self.floor_height + floor_buffer)
        clamp_max[1] = min(clamp_max[1], self.ceiling_height - ceil_buffer)
        self.flight_min = np.minimum(clamp_min, clamp_max - 1e-4)
        self.flight_max = np.maximum(clamp_max, self.flight_min + 1e-4)
        self.flight_extent = np.maximum(self.flight_max - self.flight_min, 1e-4)
        self.flight_center = (self.flight_min + self.flight_max) / 2.0
        if self.is_outdoor:
            self.motion_extent = self.safe_extent
            self.motion_center = self.safe_center
        else:
            self.motion_extent = self.flight_extent
            self.motion_center = self.flight_center
            # For indoor scenes, the bounding box center might be in a void (e.g. L-shape).
            # Prefer the largest cluster center if available and reasonably central.
            if self.clusters:
                best_cluster = np.array(self.clusters[0]["center"])
                # Only use it if it's within the flight bounds
                if np.all(best_cluster >= self.flight_min) and np.all(best_cluster <= self.flight_max):
                    self.motion_center = best_cluster

        vertical_span = max(float(self.flight_extent[1]), 1e-3)
        self.clearance_step = 0.05 * vertical_span
        self.max_clearance = self.config.clearance_ratio * vertical_span
        base_ratio = 0.42 if self.is_indoor else 0.32
        self.altitude = self._height_at_ratio(base_ratio)
        
        self.obstacles = np.array(scene_info.get("occupancy", []))
        self.obstacle_tree = KDTree(self.obstacles) if self.obstacles.size > 0 else None
        # Collision radius scales with scene size; smaller for dense indoor scenes to avoid false positives
        diag = float(np.linalg.norm(self.motion_extent))
        self.collision_radius = float(max(0.8, min(2.5, 0.02 * max(diag, 1.0))))
        self.used_focus_points: List[np.ndarray] = []
        self.last_zoom_angle: float | None = None

    def plan(self) -> Dict:
        beat_times = self.beats["beat_times"]
        if len(beat_times) < 2:
            raise ValueError("Beat tracker returned too few beats for planning")

        audio_duration = float(self.beats.get("duration", beat_times[-1]))
        target_duration = min(max(audio_duration, self.config.min_duration), self.config.max_duration)
        shot_generators = self._shot_generators()

        timeline: List[Dict] = []
        beat_index = 0
        shot_counter = 1
        beats_cycle = itertools.cycle(self.config.beats_per_shot)

        while beat_index < len(beat_times) - 1:
            beats_for_shot = next(beats_cycle)
            start_idx = beat_index
            end_idx = min(len(beat_times) - 1, beat_index + beats_for_shot)
            start_time = beat_times[start_idx]
            end_time = beat_times[end_idx]
            if end_time - beat_times[0] > target_duration:
                break
            shot_id, builder = next(shot_generators)
            shot = builder(
                shot_id=f"shot_{shot_counter:02d}",
                start_time=start_time,
                end_time=end_time,
                beat_range=(start_idx, end_idx),
            )
            if shot:
                timeline.append(shot)
                shot_counter += 1
            beat_index = end_idx

        offset = beat_times[0]
        for shot in timeline:
            shot["startTime"] -= offset
            shot["endTime"] -= offset
            for keyframe in shot["keyframes"]:
                keyframe["time"] -= offset

        if not timeline:
            raise ValueError("Unable to create any shots from beat grid")

        schedule = {
            "shots": timeline,
            "timeline_start": 0.0,
            "timeline_end": timeline[-1]["endTime"] if timeline else beat_times[-1] - offset,
            "target_duration": target_duration,
            "tempo": self.beats.get("tempo"),
            "beat_confidence": self.beats.get("confidence"),
            "beat_times": [bt - offset for bt in beat_times],
        }
        return schedule

    def _shot_generators(self) -> Iterable:
        mapping = {
            "wide_lateral": self._build_lateral_shot,
            "zoom_focus": self._build_zoom_shot,
            "orbit": self._build_orbit_shot,
            "elevated_dolly": self._build_dolly_shot,
            "slide_peek": self._build_slide_peek_shot,
            "diagonal_push": self._build_diagonal_push_shot,
            "pan_arc": self._build_pan_arc_shot,
        }
        ordered = self._shot_order_defaults()
        sequence = self._weighted_shot_sequence(ordered)
        return itertools.cycle((sid, mapping[sid]) for sid in sequence)

    def _shot_order_defaults(self) -> List[str]:
        return ["wide_lateral", "zoom_focus"]

    def _weighted_shot_sequence(self, defaults: List[str]) -> List[str]:
        # Force mix to only include allowed shots
        mix = {"wide_lateral": 0.6, "zoom_focus": 0.4}
        
        weights: List[Tuple[str, float]] = []
        for sid in defaults:
            weight = float(max(0.0, mix.get(sid, 0.0)))
            if weight > 0.0:
                weights.append((sid, weight))
        if not weights:
            weights = [(sid, 1.0) for sid in defaults]
        total = sum(weight for _, weight in weights)
        bucket = max(len(weights), 10)
        counts = {sid: max(1, int(round((weight / total) * bucket))) for sid, weight in weights}
        max_count = max(counts.values())
        sequence: List[str] = []
        for i in range(max_count):
            for sid, _ in weights:
                if i < counts[sid]:
                    sequence.append(sid)
        return sequence or defaults

    # Real implementations ------------------------------------------------

    def _build_lateral_shot(self, shot_id: str, start_time: float, end_time: float, beat_range) -> Dict:
        # Try to find a fresh location with random sampling
        collision_rejections = 0
        los_rejections = 0
        path_rejections = 0
        used_rejections = 0
        for attempt in range(500):
            # 1. Pick a potential focus point (what we are looking at)
            if self.clusters and np.random.rand() < 0.7:
                c_idx = np.random.randint(0, len(self.clusters))
                base = np.array(self.clusters[c_idx]["center"])
                # Add noise to vary the specific part of the cluster
                noise = (np.random.rand(3) - 0.5) * 15.0
                candidate_focus = base + noise
            else:
                # Random point in the flight bounds
                # Ensure we don't pick points too close to the walls to avoid staring at them
                margin = self.flight_extent * 0.15
                safe_min = self.flight_min + margin
                safe_max = self.flight_max - margin
                # If bounds are too tight, fall back to full bounds
                if np.any(safe_min >= safe_max):
                    safe_min = self.flight_min
                    safe_max = self.flight_max
                
                r = np.random.rand(3)
                candidate_focus = safe_min + r * (safe_max - safe_min)
            
            # Check if we've used a similar focus point recently (check last 8 shots)
            if any(np.linalg.norm(candidate_focus - p) < 5.0 for p in self.used_focus_points[-8:]):
                used_rejections += 1
                continue

            # 2. Pick a viewing direction (random angle in horizontal plane)
            angle = np.random.rand() * 2 * np.pi
            view_dir = np.array([np.cos(angle), 0, np.sin(angle)])
            
            # 3. Pick a distance from the focus point
            # Use cluster radius if available to keep it relative to the object size
            # But ensure a minimum distance and CAP the maximum distance to avoid going outside
            
            # Cap the base distance based on scene size to avoid going through walls
            max_scene_dim = min(self.flight_extent[0], self.flight_extent[2])
            base_dist_cap = max(4.0, 0.4 * max_scene_dim)
            
            if self.clusters:
                # Find nearest cluster to focus to get a sense of scale
                dists = [np.linalg.norm(candidate_focus - np.array(c["center"])) for c in self.clusters]
                nearest_r = self.clusters[np.argmin(dists)]["radius"]
                # Cap the base distance.
                base_dist = min(max(nearest_r * 0.6, 4.0), base_dist_cap)
            else:
                base_dist = min(0.15 * np.linalg.norm(self.motion_extent), base_dist_cap)
            
            dist = base_dist + np.random.rand() * 1.5
            
            # 4. Determine Camera Center
            cam_center = candidate_focus - view_dir * dist
            
            # 5. Determine Height
            # Vary height between 30% and 70% of vertical span
            h_ratio = 0.3 + np.random.rand() * 0.4
            cam_center[1] = self._height_at_ratio(h_ratio)
            
            # Check if this center is valid (in bounds and not in collision)
            cam_center = self._clamp_point(cam_center)
            if self._is_in_collision(cam_center):
                collision_rejections += 1
                continue
            
            # Check enclosure for indoor scenes (prevents being in the void)
            if self.is_indoor and not self._check_enclosure(cam_center):
                los_rejections += 1
                continue
            
            # CRITICAL: Check line of sight to focus. 
            # If we are outside the room and looking in, we must pass through a wall.
            # So if the path to focus intersects an obstacle, we are likely obstructed or outside.
            # Use a multiplier to "thicken" walls for this check.
            if self._path_intersects_obstacle(cam_center, candidate_focus, radius_multiplier=1.3):
                los_rejections += 1
                continue
                
            # 6. Define Motion Axis (perpendicular to view direction)
            # view_dir is (x, 0, z), up is (0, 1, 0) -> right is (z, 0, -x)
            motion_axis = np.array([view_dir[2], 0, -view_dir[0]])
            
            # 7. Define Travel Path
            scene_diag = np.linalg.norm(self.motion_extent)
            travel_dist = 0.1 * scene_diag + np.random.rand() * 0.15 * scene_diag
            start = cam_center - motion_axis * travel_dist * 0.5
            end = cam_center + motion_axis * travel_dist * 0.5
            
            # Ensure start/end height matches center
            start[1] = end[1] = cam_center[1]
            
            # Check path collision with slightly stricter radius
            if self._path_intersects_obstacle(start, end, radius_multiplier=1.2):
                path_rejections += 1
                continue
                
            # If we are here, we found a valid shot!
            self.used_focus_points.append(candidate_focus)
            
            # 8. Define Targets for Minimal Rotation
            # To have no rotation, the view vector must be constant.
            # View vector = candidate_focus - cam_center
            # Target(t) = Position(t) + ViewVector
            fixed_view_vec = candidate_focus - cam_center
            
            print(f"[Planner] Shot {shot_id} (wide_lateral): Focus={candidate_focus.round(2)}, Dist={dist:.2f}, Height={cam_center[1]:.2f}, Travel={travel_dist:.2f}")

            keyframes = [
                {"time": start_time, "position": start.tolist(), "target": (start + fixed_view_vec).tolist()},
                {"time": end_time, "position": end.tolist(), "target": (end + fixed_view_vec).tolist()},
            ]
            
            return self._package_shot(
                shot_id,
                "wide_lateral",
                start_time,
                end_time,
                beat_range,
                self._finalize_keyframes(keyframes, fix_orientation=True),
                description="Lateral tracking shot with fixed orientation",
            )

        # Fallback if random search fails (use safe defaults)
        print(f"[Planner] Shot {shot_id} (wide_lateral): attempts={attempt+1}, used_rejects={used_rejections}, coll_rejects={collision_rejections}, los_rejects={los_rejections}, path_rejects={path_rejections}")
        center = self.motion_center.copy()
        axis = self._horizontal_axis(self.axes[0])
        travel_span = np.linalg.norm(self.motion_extent) * 0.3
        start = center - axis * travel_span
        end = center + axis * travel_span
        start[1] = end[1] = self.altitude
        
        # Fixed orientation fallback
        view_vec = self._horizontal_axis(self.axes[2]) * 10.0
        
        print(f"[Planner] Shot {shot_id} (wide_lateral - FALLBACK): Center={center.round(2)}")

        keyframes = [
            {"time": start_time, "position": start.tolist(), "target": (start + view_vec).tolist()},
            {"time": end_time, "position": end.tolist(), "target": (end + view_vec).tolist()},
        ]
        return self._package_shot(
            shot_id,
            "wide_lateral",
            start_time,
            end_time,
            beat_range,
            self._finalize_keyframes(keyframes, fix_orientation=True),
            description="Fallback lateral shot",
        )

    def _build_zoom_shot(self, shot_id: str, start_time: float, end_time: float, beat_range) -> Dict:
        shot_num = self._shot_number(shot_id)
        cluster_idx = self._cluster_index_for_shot(shot_id)
        focus = self._clamp_point(np.array(self._choose_focus_cluster(cluster_idx)))

        # Base direction from motion center to focus
        base_dir = focus - self.motion_center
        if np.linalg.norm(base_dir) < 1e-3:
            base_dir = self._horizontal_axis(self.axes[1])

        base_dir = base_dir / np.linalg.norm(base_dir)

        planar_span = float(max(1e-3, min(self.motion_extent[0], self.motion_extent[2])))
        diag = np.linalg.norm(self.motion_extent)
        out_distance_base = 0.32 * planar_span + 0.25 * self.config.zoom_margin_ratio * diag
        if self.is_outdoor:
            out_distance_base = max(out_distance_base, 0.55 * np.linalg.norm(self.extent))
        cluster_radius = 0.08 * diag
        if self.clusters:
            cluster_radius = max(cluster_radius, float(self.clusters[0]["radius"]))
        in_distance_base = max(cluster_radius * 1.1, 0.12 * planar_span)
        if in_distance_base >= out_distance_base:
            in_distance_base = 0.55 * out_distance_base

        # Try multiple approach angles and small height adjustments to find a clear path.
        angle_candidates = list(np.linspace(0, 2 * np.pi, 12, endpoint=False))
        np.random.shuffle(angle_candidates)
        chosen = None
        chosen_angle = None
        chosen_start = None
        chosen_end = None
        chosen_out = None
        chosen_in = None

        for angle in angle_candidates:
            # Enforce angle jump limit relative to previous zoom angle (if set)
            deg = np.degrees(angle)
            if self.last_zoom_angle is not None:
                diff = abs(((deg - self.last_zoom_angle + 180) % 360) - 180)
                if diff > 120:
                    continue

            c, s = np.cos(angle), np.sin(angle)
            x, z = base_dir[0], base_dir[2]
            new_x = x * c - z * s
            new_z = x * s + z * c
            direction = np.array([new_x, base_dir[1], new_z])
            direction = direction / np.linalg.norm(direction)

            # Avoid vertical singularity
            if abs(direction[1]) > 0.75:
                direction[1] = 0.75 * np.sign(direction[1])
                direction = direction / np.linalg.norm(direction)

            # allow small variation in distances
            out_distance = out_distance_base + (np.random.rand() - 0.5) * 0.2 * out_distance_base
            in_distance = in_distance_base + (np.random.rand() - 0.5) * 0.2 * in_distance_base

            start_pos = focus + direction * out_distance
            end_pos = focus + direction * in_distance

            # set heights
            start_pos[1] = self._height_at_ratio(0.48)
            min_focus_height = focus[1] + 0.1 * self.vertical_span
            end_pos[1] = self._height_at_ratio(0.35, minimum=min_focus_height)

            # clamp and check collisions / line of sight
            start_pos = self._clamp_point(start_pos)
            end_pos = self._clamp_point(end_pos)
            if self._is_in_collision(start_pos) or self._is_in_collision(end_pos):
                continue
            if self._path_intersects_obstacle(start_pos, focus, radius_multiplier=1.2) or self._path_intersects_obstacle(start_pos, end_pos, radius_multiplier=1.2):
                # try raising the start height slightly to clear geometry
                for h_try in [0.1, 0.25, 0.5]:
                    s_try = start_pos.copy()
                    s_try[1] = min(self.flight_max[1], s_try[1] + h_try * self.vertical_span)
                    if not (self._is_in_collision(s_try) or self._path_intersects_obstacle(s_try, focus, radius_multiplier=1.2)):
                        start_pos = s_try
                        break
                else:
                    continue

            # passes checks
            chosen = True
            chosen_angle = angle
            chosen_start = start_pos
            chosen_end = end_pos
            chosen_out = out_distance
            chosen_in = in_distance
            break

        if not chosen:
            # Second pass: relax angle jump constraint and allow larger height adjustments
            for angle in angle_candidates:
                c, s = np.cos(angle), np.sin(angle)
                x, z = base_dir[0], base_dir[2]
                new_x = x * c - z * s
                new_z = x * s + z * c
                direction = np.array([new_x, base_dir[1], new_z])
                direction = direction / np.linalg.norm(direction)
                if abs(direction[1]) > 0.9:
                    direction[1] = 0.9 * np.sign(direction[1])
                    direction = direction / np.linalg.norm(direction)

                out_distance = out_distance_base + (np.random.rand() - 0.5) * 0.3 * out_distance_base
                in_distance = in_distance_base + (np.random.rand() - 0.5) * 0.3 * in_distance_base

                start_pos = focus + direction * out_distance
                end_pos = focus + direction * in_distance
                # try raising more aggressively
                for height_frac in [0.0, 0.1, 0.25, 0.5, 0.8]:
                    s_try = start_pos.copy()
                    e_try = end_pos.copy()
                    s_try[1] = min(self.flight_max[1], s_try[1] + height_frac * self.vertical_span)
                    e_try[1] = min(self.flight_max[1], e_try[1] + height_frac * self.vertical_span)
                    s_try = self._clamp_point(s_try)
                    e_try = self._clamp_point(e_try)
                    if self._is_in_collision(s_try) or self._is_in_collision(e_try):
                        continue
                    if self._path_intersects_obstacle(s_try, focus, radius_multiplier=1.2) or self._path_intersects_obstacle(s_try, e_try, radius_multiplier=1.2):
                        continue
                    chosen = True
                    chosen_angle = angle
                    chosen_start = s_try
                    chosen_end = e_try
                    chosen_out = out_distance
                    chosen_in = in_distance
                    break
                if chosen:
                    break

        if not chosen:
            print(f"[Planner] Shot {shot_id} (zoom_focus): no clear angle found after relax, skipping")
            return None

        # remember angle
        self.last_zoom_angle = float(np.degrees(chosen_angle))

        keyframes = [
            {"time": start_time, "position": chosen_start.tolist(), "target": focus.tolist()},
            {"time": end_time, "position": chosen_end.tolist(), "target": focus.tolist()},
        ]

        print(f"[Planner] Shot {shot_id} (zoom_focus): Focus={focus.round(2)}, AngleOffset={self.last_zoom_angle:.1f}deg, Dist={chosen_out:.2f}->{chosen_in:.2f}")

        return self._package_shot(
            shot_id,
            "zoom_focus",
            start_time,
            end_time,
            beat_range,
            self._finalize_keyframes(keyframes),
            description="Push in towards highlighted object",
        )

    def _build_orbit_shot(self, shot_id: str, start_time: float, end_time: float, beat_range) -> Dict:
        if self.is_indoor:
            return None
        cluster_idx = self._cluster_index_for_shot(shot_id, default=1)
        focus = self._clamp_point(np.array(self._choose_focus_cluster(index=cluster_idx)))
        axis_a = self._horizontal_axis(self.axes[0])
        axis_b = self._horizontal_axis(self.axes[2])
        radius = 0.45 * min(self.motion_extent[0], self.motion_extent[2])
        if self.is_outdoor:
            radius = max(radius, 0.6 * max(self.extent[0], self.extent[2]))
        times = np.linspace(start_time, end_time, num=5)
        keyframes = []
        for i, t in enumerate(times):
            angle = 2 * np.pi * (i / (len(times) - 1))
            pos = focus + axis_a * (radius * np.cos(angle)) + axis_b * (radius * np.sin(angle))
            min_height = focus[1] + 0.08 * self.vertical_span
            desired_ratio = 0.55 if self.is_outdoor else 0.5
            pos[1] = self._height_at_ratio(desired_ratio, minimum=min_height)
            keyframes.append({"time": float(t), "position": pos.tolist(), "target": focus.tolist()})
        return self._package_shot(
            shot_id,
            "orbit",
            start_time,
            end_time,
            beat_range,
            self._finalize_keyframes(keyframes),
            description="Orbit move to reveal parallax",
        )

    def _build_dolly_shot(self, shot_id: str, start_time: float, end_time: float, beat_range) -> Dict:
        axis = self._horizontal_axis(self.axes[1])
        center = self.motion_center.copy()
        travel = 0.25 * np.linalg.norm(self.motion_extent)
        if self.is_indoor:
            travel *= 0.6
        elif self.is_outdoor:
            travel = max(travel, 0.4 * np.linalg.norm(self.extent))
        near = center - axis * travel
        far = center + axis * travel
        base_height = 0.46 if self.is_indoor else 0.35
        near[1] = self._height_at_ratio(base_height)
        far[1] = self._height_at_ratio(base_height + 0.06, minimum=near[1] + 0.05 * self.vertical_span)
        keyframes = [
            {"time": start_time, "position": near.tolist(), "target": center.tolist()},
            {"time": end_time, "position": far.tolist(), "target": center.tolist()},
        ]
        return self._package_shot(
            shot_id,
            "elevated_dolly",
            start_time,
            end_time,
            beat_range,
            self._finalize_keyframes(keyframes),
            description="Floating dolly for connective tissue",
        )

    def _build_slide_peek_shot(self, shot_id: str, start_time: float, end_time: float, beat_range) -> Dict:
        cluster_idx = self._cluster_index_for_shot(shot_id)
        focus = self._clamp_point(np.array(self._choose_focus_cluster(cluster_idx)))
        axis = self._horizontal_axis(self.axes[2])
        planar_span = max(1e-3, min(self.motion_extent[0], self.motion_extent[2]))
        offset = 0.3 * planar_span
        side = -1.0 if self._shot_number(shot_id) % 2 == 0 else 1.0
        start = focus + axis * side * offset
        end = focus - axis * 0.4 * side * offset
        min_height = focus[1] + 0.05 * self.vertical_span
        height = self._height_at_ratio(0.4, minimum=min_height)
        start[1] = end[1] = height
        mid = (start + end) / 2.0
        keyframes = [
            {"time": start_time, "position": start.tolist(), "target": focus.tolist()},
            {"time": (start_time + end_time) / 2.0, "position": mid.tolist(), "target": focus.tolist()},
            {"time": end_time, "position": end.tolist(), "target": focus.tolist()},
        ]
        return self._package_shot(
            shot_id,
            "slide_peek",
            start_time,
            end_time,
            beat_range,
            self._finalize_keyframes(keyframes),
            description="Lateral slide that peeks around an interior feature",
        )

    def _build_diagonal_push_shot(self, shot_id: str, start_time: float, end_time: float, beat_range) -> Dict:
        corners = [
            np.array([self.flight_min[0], self.flight_min[1], self.flight_min[2]]),
            np.array([self.flight_max[0], self.flight_min[1], self.flight_min[2]]),
            np.array([self.flight_min[0], self.flight_min[1], self.flight_max[2]]),
            np.array([self.flight_max[0], self.flight_min[1], self.flight_max[2]]),
        ]
        corner = corners[self._shot_number(shot_id) % len(corners)].copy()
        center = self.motion_center.copy()
        corner[1] = self._height_at_ratio(0.48)
        center[1] = self._height_at_ratio(0.5)
        offset_axis = self._horizontal_axis(self.axes[0] + self.axes[2])
        offset = offset_axis * 0.1 * np.linalg.norm(self.motion_extent)
        keyframes = [
            {"time": start_time, "position": corner.tolist(), "target": center.tolist()},
            {"time": (start_time + end_time) / 2.0, "position": (corner + offset).tolist(), "target": center.tolist()},
            {"time": end_time, "position": center.tolist(), "target": center.tolist()},
        ]
        return self._package_shot(
            shot_id,
            "diagonal_push",
            start_time,
            end_time,
            beat_range,
            self._finalize_keyframes(keyframes),
            description="Diagonal push from a corner toward the focal volume",
        )

    def _build_pan_arc_shot(self, shot_id: str, start_time: float, end_time: float, beat_range) -> Dict:
        if not self.is_outdoor:
            return None
        focus = self.motion_center.copy()
        radius = max(0.6 * max(self.extent[0], self.extent[2]), 0.5 * np.linalg.norm(self.motion_extent))
        axis_a = self._horizontal_axis(self.axes[0])
        axis_b = self._horizontal_axis(self.axes[2])
        shot_num = self._shot_number(shot_id)
        base_angle = (shot_num % 4) * (np.pi / 2.0)
        sweep = np.pi / 3.0
        angles = np.linspace(base_angle, base_angle + sweep, num=4)
        altitude = self._height_at_ratio(0.55, minimum=focus[1] + 0.1 * self.vertical_span)
        keyframes = []
        for t, angle in zip(np.linspace(start_time, end_time, num=len(angles)), angles):
            pos = focus + axis_a * (radius * np.cos(angle)) + axis_b * (radius * np.sin(angle))
            pos[1] = altitude
            keyframes.append({"time": float(t), "position": pos.tolist(), "target": focus.tolist()})
        return self._package_shot(
            shot_id,
            "pan_arc",
            start_time,
            end_time,
            beat_range,
            self._finalize_keyframes(keyframes),
            description="Slow panoramic arc for outdoor vistas",
        )

    def _choose_focus_cluster(self, index: int = 0) -> List[float]:
        if not self.clusters:
            return self.motion_center.tolist()
        clamped = min(max(index, 0), len(self.clusters) - 1)
        return self.clusters[clamped]["center"]

    def _cluster_index_for_shot(self, shot_id: str, default: int = 0) -> int:
        if not self.clusters:
            return default
        return self._shot_number(shot_id) % len(self.clusters)

    def _horizontal_axis(self, axis: np.ndarray) -> np.ndarray:
        horizontal = axis.copy()
        horizontal[1] = 0.0
        if np.linalg.norm(horizontal) < 1e-3:
            horizontal = np.array([1.0, 0.0, 0.0])
        return horizontal / np.linalg.norm(horizontal)

    def _clamp_point(self, point: np.ndarray) -> np.ndarray:
        return np.clip(point, self.flight_min, self.flight_max)

    def _line_exits_bounds(self, origin: np.ndarray, target: np.ndarray) -> bool:
        samples = np.linspace(0.0, 1.0, num=10)[1:-1]
        for t in samples:
            point = origin * (1 - t) + target * t
            if np.any(point < self.flight_min) or np.any(point > self.flight_max):
                return True
        return False

    def _raise_for_clearance(self, position: np.ndarray, target: np.ndarray) -> np.ndarray:
        new_pos = position.copy()
        max_height = self.flight_max[1]
        added = 0.0
        while self._line_exits_bounds(new_pos, target) and added < self.max_clearance:
            new_pos[1] = min(new_pos[1] + self.clearance_step, max_height)
            added += self.clearance_step
            if new_pos[1] >= max_height:
                break
        return new_pos

    def _finalize_keyframes(self, keyframes: List[Dict], fix_orientation: bool = False) -> List[Dict]:
        # First pass: resolve collisions for individual points
        points = []
        targets = []
        for frame in keyframes:
            original_pos = np.array(frame["position"])
            pos = self._clamp_point(original_pos)
            # Do NOT clamp the target. The target is a look-at point and can be outside the bounds.
            tgt = np.array(frame["target"])
            
            pos = self._resolve_collision(pos)
            pos = self._clamp_point(pos)
            
            if self._line_exits_bounds(pos, tgt):
                pos = self._raise_for_clearance(pos, tgt)
                
            if self._is_in_collision(pos):
                 pos = self._resolve_collision(pos)
            
            # If we need to fix orientation, apply the position delta to the target
            if fix_orientation:
                delta = pos - original_pos
                tgt = tgt + delta
            
            # Ensure we don't get too close to the target (prevents flip)
            # If distance to target is very small, push back
            to_target = tgt - pos
            dist = np.linalg.norm(to_target)
            if dist < 2.0:
                # Push back along the view vector
                push_dir = -to_target / dist
                pos = pos + push_dir * (2.0 - dist)

            points.append(pos)
            targets.append(tgt)

        # Second pass: ensure paths between keyframes are clear
        # ... (rest of the function)
        
        for i in range(len(points) - 1):
            p1 = points[i]
            p2 = points[i+1]
            
            if self._path_intersects_obstacle(p1, p2):
                # Determine a safe pull target for this segment
                midpoint = (p1 + p2) / 2
                pull_target = self.motion_center
                if self.clusters:
                    dists = [np.linalg.norm(midpoint - np.array(c["center"])) for c in self.clusters]
                    nearest_idx = np.argmin(dists)
                    pull_target = np.array(self.clusters[nearest_idx]["center"])

                # ... (collision resolution logic)
                original_p1 = points[i].copy()
                original_p2 = points[i+1].copy()
                
                # Iteratively pull p1 and p2 towards pull_target
                for _ in range(10): # Max iterations
                    if not self._path_intersects_obstacle(p1, p2):
                        break
                    p1 = p1 * 0.8 + pull_target * 0.2
                    p2 = p2 * 0.8 + pull_target * 0.2
                
                points[i] = p1
                points[i+1] = p2
                
                if fix_orientation:
                    targets[i] += (p1 - original_p1)
                    targets[i+1] += (p2 - original_p2)

        adjusted: List[Dict] = []
        for i, frame in enumerate(keyframes):
            adjusted.append({
                "time": frame["time"], 
                "position": points[i].tolist(), 
                "target": targets[i].tolist()
            })
        return adjusted

    def _package_shot(
        self,
        shot_id: str,
        style: str,
        start_time: float,
        end_time: float,
        beat_range,
        keyframes: List[Dict],
        description: str,
    ) -> Dict:
        return {
            "id": shot_id,
            "style": style,
            "startTime": start_time,
            "endTime": end_time,
            "beatRange": beat_range,
            "keyframes": keyframes,
            "focus": keyframes[-1]["target"],
            "ease": self.config.ease,
            "description": description,
        }

    def _shot_number(self, shot_id: str) -> int:
        digits = "".join(ch for ch in shot_id if ch.isdigit())
        return int(digits) if digits else 0

    def _height_at_ratio(self, ratio: float, minimum: float | None = None) -> float:
        value = self.floor_height + float(np.clip(ratio, 0.0, 1.0)) * self.vertical_span
        if minimum is not None:
            value = max(value, minimum)
        return float(np.clip(value, self.flight_min[1], self.flight_max[1]))

    def _is_in_collision(self, point: np.ndarray, radius_multiplier: float = 1.0) -> bool:
        if self.obstacle_tree is None:
            return False
        # Query nearest neighbor
        dist, _ = self.obstacle_tree.query(point)
        return dist < (self.collision_radius * radius_multiplier)

    def _resolve_collision(self, point: np.ndarray) -> np.ndarray:
        if not self._is_in_collision(point):
            return point
        
        # Determine safe target: nearest cluster center or motion center
        target = self.motion_center
        if self.clusters:
            # Find nearest cluster center
            dists = [np.linalg.norm(point - np.array(c["center"])) for c in self.clusters]
            nearest_idx = np.argmin(dists)
            target = np.array(self.clusters[nearest_idx]["center"])

        # Try to move towards the safe target
        direction = target - point
        dist_to_target = np.linalg.norm(direction)
        if dist_to_target < 1e-3:
            return point
            
        direction /= dist_to_target
        
        # Iteratively step towards target
        current_pos = point.copy()
        step_size = 0.5
        max_steps = 20
        
        for _ in range(max_steps):
            current_pos += direction * step_size
            if not self._is_in_collision(current_pos):
                return current_pos
                
        # If simple push fails, try moving vertically towards target height
        current_pos = point.copy()
        vertical_dir = np.array([0.0, 1.0, 0.0])
        if target[1] < point[1]:
            vertical_dir *= -1
            
        for _ in range(max_steps):
            current_pos += vertical_dir * step_size
            if not self._is_in_collision(current_pos):
                return current_pos

        return point

    def _check_enclosure(self, point: np.ndarray) -> bool:
        """
        Checks if the point is 'enclosed' by obstacles, indicating it's inside a room.
        Casts rays in 6 cardinal directions.
        Returns True if at least 3 rays hit obstacles within a reasonable distance.
        """
        if not self.is_indoor or self.obstacle_tree is None:
            return True # Assume valid if not indoor or no obstacles

        hits = 0
        # Use scene diagonal as max ray distance
        max_dist = np.linalg.norm(self.motion_extent)
        
        directions = [
            np.array([1.0, 0.0, 0.0]),
            np.array([-1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, -1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
            np.array([0.0, 0.0, -1.0]),
        ]
        
        for direction in directions:
            end_point = point + direction * max_dist
            # We use a slightly larger radius multiplier to catch sparse walls
            if self._path_intersects_obstacle(point, end_point, radius_multiplier=1.5):
                hits += 1
        
        # If we are inside a room, we should hit walls/floor/ceiling in most directions.
        # 3 hits is a safe minimum (e.g. floor + 2 walls).
        return hits >= 3

    def _path_intersects_obstacle(self, start: np.ndarray, end: np.ndarray, radius_multiplier: float = 1.0) -> bool:
        if self.obstacle_tree is None:
            return False
        
        dist = np.linalg.norm(end - start)
        if dist < 1e-3:
            return self._is_in_collision(start, radius_multiplier)
            
        # Use the effective radius for step calculation to ensure we don't skip over obstacles
        effective_radius = self.collision_radius * radius_multiplier
        steps = max(5, int(dist / (effective_radius * 0.5)))
        for i in range(steps + 1):
            t = i / steps
            point = start * (1 - t) + end * t
            if self._is_in_collision(point, radius_multiplier):
                return True
        return False
