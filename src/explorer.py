from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from plyfile import PlyData


@dataclass
class SceneBounds:
    minimum: np.ndarray
    maximum: np.ndarray

    @property
    def center(self) -> np.ndarray:
        return (self.minimum + self.maximum) / 2.0

    @property
    def extent(self) -> np.ndarray:
        return self.maximum - self.minimum

    @property
    def diagonal(self) -> float:
        return float(np.linalg.norm(self.extent))


class SceneExplorer:
    """Extracts coarse geometric cues from a Gaussian splat PLY file."""

    def __init__(
        self,
        scene_path: Path,
        sample_size: int = 150_000,
        cluster_count: int = 4,
        safety_margin_ratio: float = 0.08,
    ) -> None:
        self.scene_path = scene_path
        self.sample_size = sample_size
        self.cluster_count = cluster_count
        self.safety_margin_ratio = safety_margin_ratio

    def analyze(self) -> Dict:
        vertices, colors = self._load_vertices()
        sampled, sampled_colors = self._sample(vertices, colors)
        bounds = SceneBounds(sampled.min(axis=0), sampled.max(axis=0))
        axes = self._principal_axes(sampled)
        oriented_coords = self._to_principal_frame(sampled, axes, bounds.center)
        oriented_bounds = SceneBounds(oriented_coords.min(axis=0), oriented_coords.max(axis=0))
        tight_min, tight_max = self._percentile_bounds(sampled)
        tight_bounds = SceneBounds(tight_min, tight_max)
        clusters = self._cluster(sampled, sampled_colors)
        coverage_nodes = self._coverage_nodes(bounds, axes)
        occupancy = self._generate_occupancy_grid(sampled)

        margin = np.minimum(bounds.extent * self.safety_margin_ratio, 0.45 * bounds.extent)
        safe_min = bounds.minimum + margin
        safe_max = bounds.maximum - margin
        safe_min = np.minimum(safe_min, bounds.maximum - 1e-4)
        safe_max = np.maximum(safe_max, bounds.minimum + 1e-4)
        coverage_metrics = self._coverage_metrics(oriented_coords, oriented_bounds)
        environment = self._classify_environment(sampled, axes, oriented_bounds, coverage_metrics, tight_bounds)

        analysis = {
            "scene_path": str(self.scene_path),
            "num_points": int(vertices.shape[0]),
            "sampled_points": int(sampled.shape[0]),
            "bounds": {
                "min": bounds.minimum.tolist(),
                "max": bounds.maximum.tolist(),
                "center": bounds.center.tolist(),
                "extent": bounds.extent.tolist(),
                "diagonal": bounds.diagonal,
            },
            "safe_bounds": {
                "min": safe_min.tolist(),
                "max": safe_max.tolist(),
                "margin_ratio": self.safety_margin_ratio,
            },
            "principal_axes": [axis.tolist() for axis in axes],
            "vertical_axis_index": int(environment["metrics"].get("vertical_axis_index", 1)),
            "clusters": clusters,
            "coverage_nodes": coverage_nodes,
            "occupancy": occupancy,
            "recommended_altitude": float(bounds.center[1] + 0.15 * bounds.extent[1]),
            "tight_bounds": {
                "min": tight_bounds.minimum.tolist(),
                "max": tight_bounds.maximum.tolist(),
            },
            "environment": environment,
        }
        return analysis

    def _load_vertices(self) -> Tuple[np.ndarray, np.ndarray]:
        ply = PlyData.read(str(self.scene_path))
        vertex_data = ply["vertex"].data
        coords = np.vstack([vertex_data[dim] for dim in ("x", "y", "z")]).T.astype(np.float32)
        color_fields = [
            ("red", "green", "blue"),
            ("r", "g", "b"),
            ("f_dc_0", "f_dc_1", "f_dc_2"),
        ]
        colors = np.ones_like(coords)
        for field_set in color_fields:
            if all(name in vertex_data.dtype.names for name in field_set):
                colors = np.vstack([vertex_data[name] for name in field_set]).T.astype(np.float32)
                break
        colors = colors / np.maximum(colors.max(axis=0, keepdims=True), 1e-5)
        return coords, colors

    def _sample(self, vertices: np.ndarray, colors: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if len(vertices) <= self.sample_size:
            return vertices, colors
        idx = np.random.default_rng(7).choice(len(vertices), self.sample_size, replace=False)
        return vertices[idx], colors[idx]

    def _principal_axes(self, points: np.ndarray) -> np.ndarray:
        centered = points - points.mean(axis=0)
        cov = np.cov(centered, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]
        return eigvecs[:, order].T

    def _cluster(self, points: np.ndarray, colors: np.ndarray) -> List[Dict]:
        if points.shape[0] < self.cluster_count:
            return [
                {
                    "center": points.mean(axis=0).tolist(),
                    "radius": float(np.mean(np.linalg.norm(points - points.mean(axis=0), axis=1))),
                    "score": 1.0,
                    "color": colors.mean(axis=0).tolist(),
                }
            ]
        centers = self._kmeans(points)
        clusters: List[Dict] = []
        for center in centers:
            distances = np.linalg.norm(points - center, axis=1)
            radius = float(np.percentile(distances, 70))
            mask = distances <= radius
            if not np.any(mask):
                mask = distances.argsort()[:100]
            cluster_points = points[mask]
            cluster_colors = colors[mask]
            color_variance = float(np.var(cluster_colors))
            density = float(len(cluster_points) / points.shape[0])
            score = (density + 1e-4) * (0.5 + color_variance)
            clusters.append(
                {
                    "center": center.tolist(),
                    "radius": radius,
                    "score": score,
                    "color": cluster_colors.mean(axis=0).tolist(),
                }
            )
        clusters.sort(key=lambda c: c["score"], reverse=True)
        return clusters

    def _kmeans(self, points: np.ndarray) -> np.ndarray:
        rng = np.random.default_rng(42)
        idx = rng.choice(points.shape[0], self.cluster_count, replace=False)
        centers = points[idx]
        for _ in range(25):
            distances = np.linalg.norm(points[:, None, :] - centers[None, :, :], axis=2)
            labels = np.argmin(distances, axis=1)
            new_centers = np.zeros_like(centers)
            for k in range(self.cluster_count):
                selection = points[labels == k]
                if len(selection) == 0:
                    new_centers[k] = centers[k]
                else:
                    new_centers[k] = selection.mean(axis=0)
            if np.allclose(new_centers, centers, atol=1e-3):
                break
            centers = new_centers
        return centers

    def _coverage_nodes(self, bounds: SceneBounds, axes: np.ndarray) -> List[List[float]]:
        center = bounds.center
        extent = bounds.extent
        nodes: List[List[float]] = []
        for axis in axes:
            offset = axis / np.linalg.norm(axis)
            span = extent.max() * 0.6
            nodes.append((center + offset * span).tolist())
            nodes.append((center - offset * span).tolist())
        # Elevated nodes for top-down cues
        elevated = center.copy()
        elevated[1] += extent[1] * 0.8
        nodes.append(elevated.tolist())
        return nodes

    def _percentile_bounds(self, points: np.ndarray, low: float = 5.0, high: float = 95.0) -> Tuple[np.ndarray, np.ndarray]:
        low_vals = np.percentile(points, low, axis=0)
        high_vals = np.percentile(points, high, axis=0)
        return low_vals, high_vals

    def _to_principal_frame(self, points: np.ndarray, axes: np.ndarray, center: np.ndarray) -> np.ndarray:
        centered = points - center
        return centered @ axes.T

    def _coverage_metrics(self, coords: np.ndarray, bounds: SceneBounds, band_ratio: float = 0.08) -> List[Dict]:
        metrics: List[Dict] = []
        for axis in range(3):
            span = bounds.extent[axis]
            if span <= 1e-6:
                metrics.append({"min_cover": 0.0, "max_cover": 0.0, "span": float(span)})
                continue
            normalized = (coords[:, axis] - bounds.minimum[axis]) / span
            min_cover = float(np.mean(normalized <= band_ratio))
            max_cover = float(np.mean(normalized >= 1.0 - band_ratio))
            metrics.append({"min_cover": min_cover, "max_cover": max_cover, "span": float(span)})
        return metrics

    def _classify_environment(
        self,
        points: np.ndarray,
        axes: np.ndarray,
        oriented_bounds: SceneBounds,
        coverage: List[Dict],
        tight_bounds: SceneBounds,
    ) -> Dict:
        if points.size == 0:
            return {"type": "unknown", "confidence": 0.0, "metrics": {}}

        floor_height = float(np.percentile(points[:, 1], 5))
        ceiling_height = float(np.percentile(points[:, 1], 95))
        vertical_idx = int(np.argmax(np.abs(axes[:, 1])))
        horizontal_idx = [i for i in range(3) if i != vertical_idx]

        def _face_score(idx: int) -> float:
            data = coverage[idx]
            return min(data["min_cover"], data["max_cover"])

        horizontal_scores = [_face_score(idx) for idx in horizontal_idx]
        horizontal_score = float(np.mean(horizontal_scores)) if horizontal_scores else 0.0
        vertical_data = coverage[vertical_idx]
        ceiling_score = float(vertical_data["max_cover"])
        floor_score = float(vertical_data["min_cover"])
        enclosure = float(np.mean(horizontal_scores + [min(ceiling_score, floor_score)]))

        horiz_span = float(np.mean([coverage[idx]["span"] for idx in horizontal_idx])) if horizontal_idx else 0.0
        vertical_span = float(vertical_data["span"])
        aspect_ratio = float(vertical_span / max(horiz_span, 1e-4)) if horiz_span else 0.0
        world_span = tight_bounds.extent
        world_vertical_span = float(world_span[1])
        world_horizontal_span = float((world_span[0] + world_span[2]) * 0.5)

        indoor_votes = 0.0
        if horizontal_score > 0.12:  # Lowered from 0.16
            indoor_votes += 0.4
        if ceiling_score > 0.1:      # Lowered from 0.12
            indoor_votes += 0.25
        if floor_score > 0.08:       # Lowered from 0.1
            indoor_votes += 0.15
        if aspect_ratio < 0.4:       # Raised from 0.35
            indoor_votes += 0.2
        indoor_confidence = min(1.0, indoor_votes)
        compact_volume = world_vertical_span < 7.5 and (world_vertical_span / max(world_horizontal_span, 1e-3)) < 0.25
        if compact_volume:
            indoor_confidence = max(indoor_confidence, 0.75)

        outdoor_threshold = 0.04
        if indoor_confidence >= 0.65:
            env_type = "indoor"
            confidence = indoor_confidence
        elif enclosure < outdoor_threshold and ceiling_score < 0.05:
            env_type = "outdoor"
            confidence = min(1.0, 0.5 + (outdoor_threshold - enclosure) / max(outdoor_threshold, 1e-4))
        else:
            env_type = "ambiguous"
            confidence = max(0.4, indoor_confidence)

        metrics = {
            "horizontal_score": horizontal_score,
            "ceiling_score": ceiling_score,
            "floor_score": floor_score,
            "enclosure_score": enclosure,
            "aspect_ratio": aspect_ratio,
            "vertical_span": vertical_span,
            "horizontal_span": horiz_span,
            "world_vertical_span": world_vertical_span,
            "world_horizontal_span": world_horizontal_span,
            "floor_height": floor_height,
            "ceiling_height": ceiling_height,
            "tight_extent": tight_bounds.extent.tolist(),
            "vertical_axis_index": vertical_idx,
            "compact_volume": compact_volume,
        }

        return {
            "type": env_type,
            "confidence": max(0.0, min(1.0, confidence)),
            "metrics": metrics,
        }

    def _generate_occupancy_grid(self, points: np.ndarray, voxel_size: float = 0.8) -> List[List[float]]:
        if points.size == 0:
            return []
        quantized = np.floor(points / voxel_size).astype(int)
        unique_voxels = np.unique(quantized, axis=0)
        voxel_centers = (unique_voxels * voxel_size) + (voxel_size / 2.0)
        return voxel_centers.tolist()
 