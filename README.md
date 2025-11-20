# Cinematic Navigation Pipeline

Autonomous film-maker that analyzes a Gaussian splatting scene, locks shots to the beat of a soundtrack, and renders the final video through SparkJS (THREE.js Gaussian splatting renderer) using a headless Chrome harness. The pipeline guarantees at least one wide lateral establishing move and one zoomed focus shot while covering the environment with beat-aligned edits.

## Prerequisites

- Python 3.11+ with `venv`
- Node.js 18+ (Puppeteer requires Chromium download)
- `ffmpeg` available on `PATH`

Install dependencies from the project root:

```bash
python -m pip install -r requirements.txt
cd renderer
npm install
```

## Configuration

Defaults live in `configs/config.yaml`. Key knobs:

- `scene.primary_ply` / `scene.fallback_ply`: Gaussian splat `.ply` assets under `input/`
- `music.track_path`: soundtrack for beat detection and final mix
- `render`: resolution, fps, min/max clip duration, spatial margins
- `shots`: beat allocation per shot type, altitude ratios, and the new `safety_margin_ratio` / `clearance_ratio` that keep indoor cameras inside rooms and away from walls
- `paths`: output locations (JSON reports, runtime config, frames, MP4)

## Running the pipeline

From the repo root:

```bash
python -m src.pipeline --config configs/config.yaml
```

Optional flags:

- `--scene path/to/custom_scene.ply`
- `--music path/to/song.mp3`
- `--skip-render` (run analysis/planning only)
- `--node-bin /custom/node` or `--ffmpeg-bin /opt/homebrew/bin/ffmpeg`

The final beat-synced panorama appears at `outputs/scene_1/panorama_tour.mp4`. Intermediate artifacts include:

- `outputs/scene_1/scene_analysis.json`: bounding boxes, clusters, coverage nodes
- `outputs/scene_1/beat_times.json`: detected tempo and beat timestamps
- `outputs/scene_1/shots.json`: ordered shot definitions with beat-aligned boundaries
- `outputs/scene_1/runtime_config.json`: SparkJS runtime payload consumed by `renderer/render.js`

## Architecture

1. **Scene Explorer (`src/explorer.py`)**: parses PLY, computes principal axes, clusters salient regions for zoom shots, and recommends navigation altitudes.
2. **Beat Tracker (`src/beat_tracker.py`)**: runs `librosa` beat analysis to obtain tempo, beat grid, and confidence.
3. **Camera Planner (`src/camera_planner.py`)**: cycles through wide lateral, zoom, orbital, and elevated dolly moves. Durations snap to beat multiples to guarantee cut points land exactly on music beats.
4. **Spark Config Builder (`src/spark_config_builder.py`)**: converts schedule into a JSON contract that the Puppeteer harness understands (camera keyframes, shot metadata, fps, frame output path).
5. **Renderer Harness (`renderer/render.js`)**: launches headless Chrome with Puppeteer, loads `renderer/spark_scene.html`, and asks SparkJS to render each frame. Frames are written as PNGs before `ffmpeg` muxes them with the soundtrack.

## Creative Guarantees

- Establishing wide lateral sweep covering the dominant principal axis.
- Zoom-in shot targeting the most salient Gaussian cluster.
- Additional orbit and dolly shots for variety and coverage.
- All shot boundaries aligned to music beats for rhythmic editing.

## Known Limitations

- Beat detection assumes reasonably percussive audio; very ambient tracks may need manual tempo overrides.
- Obstacle avoidance is approximated via bounding-box constraints—finer collision checks would require point-cloud occlusion tests.
- Rendering depends on headless Chrome WebGL support; some CI hosts may require `--no-sandbox` tweaks.
- Only a single required video is automated (object-tour bonus path can be added by extending the planner with object selection heuristics).

## Extending

- Plug in an object detector to bias zoom/orbit shots towards semantic targets.
- Introduce easing curves per shot or blend tree for more advanced cinematography.
- Export secondary edits (object-focused cut) by instantiating another `CameraPlanner` with different templates.
