# Cinematic Navigation Pipeline — README

This repository implements an autonomous cinematic navigation pipeline that:

- analyzes a Gaussian-splat `.ply` scene,
- detects beats from a soundtrack,
- plans beat-aligned camera shots (wide, zoom, orbit, dolly), and
- renders frames via SparkJS in headless Chrome before muxing with `ffmpeg`.

## Installation

- Python 3.11+ (create a `venv` recommended)
- Node.js 18+ (for `renderer` and Puppeteer/Chromium)
- `ffmpeg` available on `PATH`

Quick install from the project root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cd renderer
npm install
cd ..
```

## Usage

Run the full pipeline (analysis → planning → render):

```bash
python -m src.pipeline --config configs/config.yaml
```

Examples:

- Run analysis and planning only (no render):
  `python -m src.pipeline --config configs/config.yaml --skip-render`
- Override scene or music:
  `python -m src.pipeline --scene input/custom.ply --music input/song.mp3`

Outputs (example):

- `outputs/scene_1/scene_analysis.json` — scene PCA, clusters, bounds
- `outputs/scene_1/beat_times.json` — tempo, beat timestamps, confidence
- `outputs/scene_1/shots.json` — ordered, beat-snapped shots
- `outputs/scene_1/runtime_config.json` — payload for SparkJS renderer
- `outputs/scene_1/panorama_tour.mp4` — final muxed video

## Algorithm overview

- Scene Explorer (`src/explorer.py`): samples the splat cloud, computes PCA for dominant axes, clusters salient Gaussian regions for zoom targets, and estimates safe camera altitudes.
- Beat Tracker (`src/beat_tracker.py`): uses `librosa` (on audio waves) to estimate tempo, beat locations, and a confidence score.
- Camera Planner (`src/camera_planner.py`): composes a sequence of shot templates (ensures at least one wide lateral establishing shot and one zoom-in) and snaps shot boundaries to beats; applies safety clamps against scene bounding volumes.
- Spark Config Builder (`src/spark_config_builder.py`): translates shots into a JSON keyframe contract consumed by the renderer.
- Renderer (`renderer/render.js`): launches headless Chromium, loads SparkJS scene, renders frames as PNGs, and `ffmpeg` muxes frames with the soundtrack.

## Dependencies

- Python packages: listed in `requirements.txt` (e.g., `librosa`, `numpy`, `scipy`, `opencv-python`)
- Node packages: in `renderer/package.json` (Puppeteer / SparkJS runtime)
- System: `ffmpeg`, headless Chromium (installed by Puppeteer or available on PATH)

## Known limitations

- Beat detection is less reliable on very ambient or textureless music; consider providing a clearer track or overriding tempo manually.
- Path planning sometimes doesn't work well: on rare scenes the planner can produce a path where the camera momentarily ends up outside of the scene geometry. However, almost all collisions are avoided by the safety margins, and the camera does not move outside of the scene on its own in most runs.
- Collision avoidance uses bounding-box and clearance heuristics rather than dense point-cloud collision checks — a more robust SDF or ray-marching approach would reduce edge cases.

## Support & Extending

- To bias shots toward semantic objects, add an object detector that produces priority targets for the planner.
- For stricter collision safety, add point-cloud SDF approximations or spatial hashing for per-frame occlusion tests.

## Contact

See `Report.md` for the technical write-up and improvement roadmap.
