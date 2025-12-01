from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Dict

from .beat_tracker import BeatAnalyzer
from .camera_planner import CameraPlanner, PlannerConfig
from .explorer import SceneExplorer
from .spark_config_builder import SparkConfigBuilder
from .utils.files import ensure_dir, read_yaml, write_json

# Optional detector module
try:
    from .detector import process_frames_directory
    HAS_DETECTOR = True
except ImportError:
    HAS_DETECTOR = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Autonomous cinematic navigation pipeline")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to YAML config file")
    parser.add_argument("--scene", default=None, help="Override scene PLY path")
    parser.add_argument("--music", default=None, help="Override music track path")
    parser.add_argument("--skip-render", action="store_true", help="Stop after planning")
    parser.add_argument("--dry-run", action="store_true", help="Alias for --skip-render")
    parser.add_argument("--node-bin", default="node", help="Node.js executable")
    parser.add_argument("--ffmpeg-bin", default="ffmpeg", help="ffmpeg executable")
    parser.add_argument("--no-detection", action="store_true", help="Skip object detection")
    parser.add_argument("--detection-model", default="yolo11s.pt", help="YOLO model to use (yolo11s.pt recommended)")
    parser.add_argument("--detection-conf", type=float, default=0.4, help="Detection confidence threshold")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = read_yaml(Path(args.config))
    workspace = Path(__file__).resolve().parents[1]
    scene_path = _resolve_scene_path(args.scene, cfg["scene"], workspace)
    music_path = _resolve_music_path(args.music, cfg["music"], workspace)
    paths = _normalize_paths(cfg["paths"], workspace)
    render_settings = cfg["render"]

    ensure_dir(paths["working_dir"])

    print("[1/6] Scene analysis ->", scene_path)
    safety_margin_ratio = cfg["shots"].get("safety_margin_ratio", 0.08)
    explorer = SceneExplorer(scene_path, safety_margin_ratio=safety_margin_ratio)
    scene_info = explorer.analyze()
    write_json(paths["scene_report"], scene_info)

    print("[2/6] Beat tracking ->", music_path)
    beat_analysis = BeatAnalyzer(music_path).analyze()
    write_json(paths["beats_report"], beat_analysis.to_dict())

    print("[3/6] Planning camera choreography")
    planner = CameraPlanner(
        scene_info,
        beat_analysis.to_dict(),
        PlannerConfig(
            beats_per_shot=cfg["shots"]["beats_per_shot"],
            ease=cfg["shots"].get("ease", "smoothstep"),
            lateral_margin_ratio=render_settings.get("lateral_margin_ratio", 0.2),
            zoom_margin_ratio=render_settings.get("zoom_margin_ratio", 0.08),
            orbit_altitude_ratio=cfg["shots"].get("orbit_altitude_ratio", 0.35),
            dolly_altitude_ratio=cfg["shots"].get("dolly_altitude_ratio", 0.15),
            min_duration=render_settings.get("min_duration_seconds", 60.0),
            max_duration=render_settings.get("max_duration_seconds", 110.0),
            safety_margin_ratio=safety_margin_ratio,
            clearance_ratio=cfg["shots"].get("clearance_ratio", 0.12),
            duration_scale=cfg["shots"].get("duration_scale", 3.0),
            shot_mix=cfg["shots"].get("shot_mix"),
        ),
    )
    schedule = planner.plan()
    write_json(paths["shots_report"], schedule)

    print("[4/6] Building SparkJS runtime config")
    scene_web = _to_web_path(scene_path, workspace)
    music_web = _to_web_path(music_path, workspace)

    spark_config = SparkConfigBuilder(
        scene_web,
        music_web,
        schedule,
        render_settings,
        paths,
    ).build()
    write_json(paths["renderer_config"], spark_config)

    if args.skip_render or args.dry_run:
        print("Skip render enabled, pipeline stops after planning phase")
        return

    frames_dir = Path(paths["frames_dir"])
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)

    print("[5/6] Rendering frames with Playwright + SparkJS")
    render_cmd = [
        args.node_bin,
        "renderer/render.js",
        "--config",
        str(paths["renderer_config"]),
        "--asset-root",
        str(workspace),
    ]
    subprocess.run(render_cmd, cwd=workspace, check=True)

    # Run object detection on frames
    if not args.no_detection and HAS_DETECTOR:
        print("[6/6] Running YOLO object detection on frames")
        process_frames_directory(
            frames_dir=frames_dir,
            model_name=args.detection_model,
            confidence_threshold=args.detection_conf,
            in_place=True,
        )
    else:
        if not HAS_DETECTOR:
            print("[6/6] Skipping object detection (detector module not available)")
        else:
            print("[6/6] Skipping object detection (--no-detection flag)")

    print("Encoding final video with ffmpeg")
    _encode_video(
        ffmpeg_bin=args.ffmpeg_bin,
        frames_dir=frames_dir,
        fps=render_settings.get("fps", 30),
        audio_path=music_path,
        output_path=Path(paths["video_output"]),
    )


def _resolve_scene_path(override: str | None, scene_cfg: Dict, workspace: Path) -> Path:
    candidates = [override, scene_cfg.get("primary_ply"), scene_cfg.get("fallback_ply"), scene_cfg.get("ply_path"), "input/scene.ply"]
    for candidate in candidates:
        if not candidate:
            continue
        path = (workspace / candidate).resolve() if not candidate.startswith("/") else Path(candidate)
        if path.exists():
            return path
    raise FileNotFoundError("No valid scene PLY file found")


def _resolve_music_path(override: str | None, music_cfg: Dict, workspace: Path) -> Path:
    candidates = [override, music_cfg.get("track_path"), "input/music.mp3"]
    for candidate in candidates:
        if not candidate:
            continue
        path = (workspace / candidate).resolve() if not candidate.startswith("/") else Path(candidate)
        if path.exists():
            return path
    raise FileNotFoundError("Music track not found")


def _normalize_paths(raw_paths: Dict, workspace: Path) -> Dict:
    return {key: (workspace / Path(value)).resolve() for key, value in raw_paths.items()}


def _to_web_path(file_path: Path, workspace: Path) -> str:
    try:
        rel = file_path.relative_to(workspace)
        return rel.as_posix()
    except ValueError:
        return file_path.as_posix()


def _encode_video(ffmpeg_bin: str, frames_dir: Path, fps: int, audio_path: Path, output_path: Path) -> None:
    pattern = str(frames_dir / "frame_%05d.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg_bin,
        "-y",
        "-framerate",
        str(fps),
        "-i",
        pattern,
        "-i",
        str(audio_path),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-shortest",
        str(output_path),
    ]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
