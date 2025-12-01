# Technical Report: Cinematic Navigation HW04

# Technical Report — Cinematic Navigation (Assignment 4)

This report documents problem solving, key ideas, results, and a concrete vision for future improvements.

1. Completed tasks

---

- Rendered a video from inside of the scene
- Detected objects in the rendered video
- Path planning
- Obstacle avoidance
- Artistic result (interesting camera angles and background music)

2. How the problem was solved

---

- Input: a Gaussian-splat `.ply` scene and a soundtrack.
- Pipeline stages:
  - Scene analysis (`src/explorer.py`): random sampling (cap at ~150k points), PCA for dominant axes, clustering to find salient regions for zoom targets, and bounding volumes for safety margins.
  - Beat extraction (`src/beat_tracker.py`): `librosa`-based tempo and beat detection with an autocorrelation confidence score. Beats are exported in `beat_times.json`.
  - Camera planning (`src/camera_planner.py`): apply a small set of shot templates (wide lateral establishing shot, zoom-in, orbit, dolly). Shots are snapped to beat indices; keyframes are safety-clamped to bounding volumes.
  - Rendering orchestration (`src/spark_config_builder.py` + `renderer/`): write a runtime JSON contract and drive SparkJS in headless Chromium to render frames, then mux with `ffmpeg`.

3. Novelty and proud tricks

---

- Beat-snapped scheduling: every shot boundary is quantized to beat indices so edits are rhythmically correct by construction.
- Lightweight safety heuristic: compute conservative bounding volumes and a clearance ratio; clamp keyframes and bump altitudes when view rays intersect unsafe volumes — a cheap but effective collision reduction method.
- Deterministic renderer harness: saving per-frame PNGs via a headless Chromium script ensures reproducible renders and easier debugging compared to screen capture.

4. Challenges faced and how they were addressed

---

- Large scenes: sampling with a fixed RNG seed kept PCA and clustering fast and repeatable.
- Ambiguous tempo: computed a confidence metric and exportable beat files so graders or users can manually override tempo when necessary.
- Headless WebGL: added Chromium flags and a local static server to ensure SparkJS can load `.ply` reliably in headless environments.

5. Results and evaluation

---

- The pipeline produces a beat-aligned video with the required establishing and zoom shots in the expected duration range.
- Motion is smoothed with interpolation (e.g., `smoothstep`) and transitions only occur on beats.
- Empirically, almost all collisions are avoided by safety margins; occasional edge cases remain (see limitations below).

6. Known limitations (short)

---

- Beat detectors struggle with very ambient music; manual tempo override is available.
- Path planning sometimes produces a trajectory where the camera briefly ends up outside the scene geometry. Nevertheless, almost all collisions are avoided by the safety heuristics and the camera rarely exits the scene on its own.

7. Vision for future improvements (technical + creative)

---

- Technical improvements:

  - Replace bounding-box heuristics with a signed distance field (SDF) / voxel SDF approximation to guarantee collision-free trajectories; use gradient-based optimization to produce smooth, obstacle-aware paths.
  - Add per-frame ray-marching occlusion checks from the camera to the focus target to prevent view clipping and popping.
  - Integrate a learned or analytic motion prior to produce more cinematic acceleration profiles (e.g., ease-in/out per shot using a motion library).
  - Add semantic detection (object segmentation + ranking) and cost functions to bias camera placement toward meaningful targets.

- Creative directions:
  - Allow multi-track musical analysis (stems) and align shots not only to beats but to musical phrases (tension, release) for stronger narrative edits.
  - Provide an interactive web preview where a user can tweak a small number of shot seeds and re-render only affected shots for quick iteration.
  - Implement stylistic templates (documentary, suspense, portrait) that change camera spacing, focal lengths, and motion priors to generate distinct cinematic feels.

8. Short reproducibility notes

---

- See `README.md` for install/run examples. Primary outputs are in `outputs/` and inspection artifacts (`scene_analysis.json`, `beat_times.json`, `shots.json`) are exported alongside the final video.

This deliverable focuses on a concise, reproducible pipeline; the next steps are primarily improving collision guarantees and integrating semantic shot priorities.
