"""
YOLO-based object detection with persistent tracking for video frames.

Uses ultralytics YOLO for detection and maintains consistent bounding box
colors across consecutive frames by tracking objects based on position
and class similarity.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

try:
    from ultralytics import YOLO
    import torch
    HAS_ULTRALYTICS = True
    # Check CUDA availability
    CUDA_AVAILABLE = torch.cuda.is_available()
    if CUDA_AVAILABLE:
        # Get device capability to ensure compatibility
        capability = torch.cuda.get_device_capability(0)
        DEVICE = 0  # Use device index for ultralytics
        print(f"[Detector] CUDA available: {torch.cuda.get_device_name(0)} (compute {capability[0]}.{capability[1]})")
    else:
        DEVICE = "cpu"
        print("[Detector] CUDA not available, using CPU")
except ImportError:
    HAS_ULTRALYTICS = False
    CUDA_AVAILABLE = False
    DEVICE = "cpu"
    YOLO = None


@dataclass
class TrackedObject:
    """An object being tracked across frames."""
    track_id: int
    class_id: int
    class_name: str
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    color: Tuple[int, int, int]  # BGR color
    last_seen_frame: int
    center: Tuple[float, float]


class ObjectTracker:
    """Simple IoU-based object tracker for maintaining consistent IDs across frames.
    
    This tracker assigns persistent IDs to detected objects across frames,
    allowing bounding box colors to remain consistent as objects move.
    """
    
    def __init__(self, iou_threshold: float = 0.3, max_age: int = 10):
        """
        Args:
            iou_threshold: Minimum IoU for matching detections to tracks
            max_age: Maximum frames an object can be missing before track is deleted
        """
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.tracks: Dict[int, TrackedObject] = {}
        self.next_track_id = 0
        self.current_frame = 0
        
        # Pre-computed distinct colors for tracking (vivid, easily distinguishable)
        self.color_palette = self._generate_color_palette(100)
    
    def _generate_color_palette(self, n_colors: int) -> List[Tuple[int, int, int]]:
        """Generate a palette of distinct, vivid colors for tracking."""
        colors = []
        for i in range(n_colors):
            # Use golden ratio to spread hues evenly
            hue = (i * 0.618033988749895) % 1.0
            # High saturation and value for vivid colors
            saturation = 0.8 + (i % 3) * 0.1
            value = 0.9 + (i % 2) * 0.1
            
            # Convert HSV to BGR
            h = int(hue * 180)
            s = int(min(saturation, 1.0) * 255)
            v = int(min(value, 1.0) * 255)
            
            # Create a 1x1 HSV image and convert to BGR
            hsv = np.uint8([[[h, s, v]]])
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            colors.append(tuple(int(c) for c in bgr[0, 0]))
        
        return colors
    
    def _get_color_for_track(self, track_id: int, class_id: int) -> Tuple[int, int, int]:
        """Get a consistent color for a track based on its ID and class."""
        # Combine track_id and class_id for more variety
        color_idx = (track_id * 7 + class_id * 3) % len(self.color_palette)
        return self.color_palette[color_idx]
    
    def _compute_iou(self, box1: Tuple[int, int, int, int], 
                     box2: Tuple[int, int, int, int]) -> float:
        """Compute Intersection over Union between two bounding boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    def _compute_center(self, bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
        """Compute center of bounding box."""
        return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
    
    def _compute_distance(self, center1: Tuple[float, float], 
                          center2: Tuple[float, float]) -> float:
        """Compute Euclidean distance between two centers."""
        return np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
    
    def update(self, detections: List[Dict], frame_idx: int) -> List[TrackedObject]:
        """Update tracks with new detections.
        
        Args:
            detections: List of dicts with keys: bbox, class_id, class_name, confidence
            frame_idx: Current frame index
            
        Returns:
            List of TrackedObject with assigned track IDs and colors
        """
        self.current_frame = frame_idx
        
        # Remove old tracks
        expired_ids = [
            tid for tid, track in self.tracks.items()
            if frame_idx - track.last_seen_frame > self.max_age
        ]
        for tid in expired_ids:
            del self.tracks[tid]
        
        if not detections:
            return []
        
        # Convert detections to standard format
        det_bboxes = []
        det_info = []
        for det in detections:
            bbox = tuple(int(x) for x in det["bbox"])
            det_bboxes.append(bbox)
            det_info.append({
                "bbox": bbox,
                "class_id": det["class_id"],
                "class_name": det["class_name"],
                "confidence": det["confidence"],
                "center": self._compute_center(bbox),
            })
        
        # Match detections to existing tracks
        matched_tracks = {}  # detection_idx -> track_id
        matched_track_ids = set()
        
        if self.tracks:
            # Compute IoU matrix
            track_ids = list(self.tracks.keys())
            iou_matrix = np.zeros((len(det_bboxes), len(track_ids)))
            
            for i, bbox in enumerate(det_bboxes):
                for j, tid in enumerate(track_ids):
                    track = self.tracks[tid]
                    # Only match same class
                    if det_info[i]["class_id"] == track.class_id:
                        iou = self._compute_iou(bbox, track.bbox)
                        # Also consider center distance for small/fast objects
                        dist = self._compute_distance(det_info[i]["center"], track.center)
                        # Normalize distance by image diagonal (assume ~1000 px)
                        dist_score = max(0, 1 - dist / 500)
                        # Combined score
                        iou_matrix[i, j] = 0.7 * iou + 0.3 * dist_score
            
            # Greedy matching (highest IoU first)
            while True:
                if iou_matrix.size == 0:
                    break
                max_iou = iou_matrix.max()
                if max_iou < self.iou_threshold:
                    break
                
                det_idx, track_idx = np.unravel_index(iou_matrix.argmax(), iou_matrix.shape)
                track_id = track_ids[track_idx]
                
                matched_tracks[det_idx] = track_id
                matched_track_ids.add(track_id)
                
                # Zero out matched row and column
                iou_matrix[det_idx, :] = 0
                iou_matrix[:, track_idx] = 0
        
        # Update matched tracks and create new ones for unmatched detections
        results = []
        
        for det_idx, info in enumerate(det_info):
            if det_idx in matched_tracks:
                # Update existing track
                track_id = matched_tracks[det_idx]
                track = self.tracks[track_id]
                track.bbox = info["bbox"]
                track.confidence = info["confidence"]
                track.last_seen_frame = frame_idx
                track.center = info["center"]
            else:
                # Create new track
                track_id = self.next_track_id
                self.next_track_id += 1
                
                color = self._get_color_for_track(track_id, info["class_id"])
                
                track = TrackedObject(
                    track_id=track_id,
                    class_id=info["class_id"],
                    class_name=info["class_name"],
                    bbox=info["bbox"],
                    confidence=info["confidence"],
                    color=color,
                    last_seen_frame=frame_idx,
                    center=info["center"],
                )
                self.tracks[track_id] = track
            
            results.append(self.tracks[track_id] if det_idx in matched_tracks else track)
        
        return results


class YOLODetector:
    """YOLO-based object detector with tracking."""
    
    def __init__(
        self,
        model_name: str = "yolo11s.pt",  # YOLOv11 small model
        confidence_threshold: float = 0.4,
        iou_threshold: float = 0.3,
        track_max_age: int = 15,
        device = None,
    ):
        """
        Args:
            model_name: YOLO model to use (yolo11s.pt recommended for quality/speed balance)
            confidence_threshold: Minimum confidence for detections
            iou_threshold: IoU threshold for tracking
            track_max_age: Max frames an object can be missing
            device: Device to use ('cpu' recommended due to CUDA compatibility issues)
        """
        if not HAS_ULTRALYTICS:
            raise ImportError("ultralytics package is required. Install with: pip install ultralytics")
        
        # Force CPU due to CUDA illegal instruction errors with some GPU/driver combinations
        self.device = "cpu"
        self.model = YOLO(model_name)
        print(f"[Detector] Using device: {self.device} (CPU mode for stability)")
        
        self.confidence_threshold = confidence_threshold
        self.tracker = ObjectTracker(iou_threshold=iou_threshold, max_age=track_max_age)
    
    def detect(self, image: np.ndarray, frame_idx: int = 0) -> List[TrackedObject]:
        """Detect and track objects in an image.
        
        Args:
            image: BGR image as numpy array
            frame_idx: Current frame index for tracking
            
        Returns:
            List of TrackedObject with bounding boxes and consistent colors
        """
        # Run YOLO inference - device is passed directly to predict
        results = self.model.predict(image, verbose=False, conf=self.confidence_threshold, device=self.device)
        
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            
            for box in boxes:
                bbox = box.xyxy[0].cpu().numpy()
                class_id = int(box.cls[0].cpu().numpy())
                confidence = float(box.conf[0].cpu().numpy())
                class_name = self.model.names[class_id]
                
                detections.append({
                    "bbox": bbox,
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": confidence,
                })
        
        # Update tracker
        tracked_objects = self.tracker.update(detections, frame_idx)
        
        return tracked_objects
    
    def draw_detections(
        self,
        image: np.ndarray,
        tracked_objects: List[TrackedObject],
        line_thickness: int = 2,
        font_scale: float = 0.6,
        show_confidence: bool = True,
    ) -> np.ndarray:
        """Draw bounding boxes and labels on image.
        
        Args:
            image: BGR image as numpy array
            tracked_objects: List of tracked objects to draw
            line_thickness: Thickness of bounding box lines
            font_scale: Scale of label text
            show_confidence: Whether to show confidence scores
            
        Returns:
            Image with drawn detections
        """
        result = image.copy()
        
        for obj in tracked_objects:
            x1, y1, x2, y2 = obj.bbox
            color = obj.color
            
            # Draw bounding box
            cv2.rectangle(result, (x1, y1), (x2, y2), color, line_thickness)
            
            # Prepare label
            if show_confidence:
                label = f"{obj.class_name} {obj.confidence:.2f}"
            else:
                label = obj.class_name
            
            # Get text size for background
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
            )
            
            # Draw label background
            cv2.rectangle(
                result,
                (x1, y1 - text_height - baseline - 4),
                (x1 + text_width + 4, y1),
                color,
                -1,  # Filled
            )
            
            # Draw label text (white for contrast)
            cv2.putText(
                result,
                label,
                (x1 + 2, y1 - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        
        return result


def process_frames_directory(
    frames_dir: Path,
    model_name: str = "yolov8n.pt",
    confidence_threshold: float = 0.4,
    in_place: bool = True,
    output_dir: Optional[Path] = None,
) -> None:
    """Process all frames in a directory with YOLO detection and tracking.
    
    Args:
        frames_dir: Directory containing frame images (frame_00001.png, etc.)
        model_name: YOLO model to use
        confidence_threshold: Minimum confidence for detections
        in_place: If True, overwrite original frames. If False, use output_dir
        output_dir: Directory for output frames (required if in_place=False)
    """
    if not HAS_ULTRALYTICS:
        print("[Detector] Warning: ultralytics not installed, skipping detection")
        return
    
    frames_dir = Path(frames_dir)
    if not frames_dir.exists():
        print(f"[Detector] Warning: frames directory not found: {frames_dir}")
        return
    
    # Find all frame files
    frame_files = sorted(frames_dir.glob("frame_*.png"))
    if not frame_files:
        frame_files = sorted(frames_dir.glob("*.png"))
    if not frame_files:
        frame_files = sorted(frames_dir.glob("*.jpg"))
    
    if not frame_files:
        print("[Detector] No frame files found")
        return
    
    print(f"[Detector] Processing {len(frame_files)} frames with {model_name} on {DEVICE}")
    
    # Initialize detector with tracking
    detector = YOLODetector(
        model_name=model_name,
        confidence_threshold=confidence_threshold,
        iou_threshold=0.3,
        track_max_age=15,
        device=DEVICE,
    )
    
    # Determine output directory
    if in_place:
        out_dir = frames_dir
    else:
        out_dir = output_dir or frames_dir.parent / "frames_detected"
        out_dir.mkdir(parents=True, exist_ok=True)
    
    # Process frames with progress
    total = len(frame_files)
    detection_counts = []
    
    for idx, frame_path in enumerate(frame_files):
        # Read frame
        image = cv2.imread(str(frame_path))
        if image is None:
            continue
        
        # Detect and track
        tracked_objects = detector.detect(image, frame_idx=idx)
        detection_counts.append(len(tracked_objects))
        
        # Draw detections
        result = detector.draw_detections(image, tracked_objects)
        
        # Save result
        output_path = out_dir / frame_path.name
        cv2.imwrite(str(output_path), result)
        
        # Progress
        if (idx + 1) % 50 == 0 or idx == total - 1:
            avg_detections = sum(detection_counts[-50:]) / min(50, len(detection_counts[-50:]))
            print(f"[Detector] Processed {idx + 1}/{total} frames, avg {avg_detections:.1f} detections/frame")
    
    total_detections = sum(detection_counts)
    print(f"[Detector] Complete: {total_detections} total detections across {total} frames")
