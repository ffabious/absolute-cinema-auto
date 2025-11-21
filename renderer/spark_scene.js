import * as THREE from "three";
import { SparkRenderer, SplatMesh } from "@sparkjsdev/spark";

const canvas = document.getElementById("spark-canvas");
const contextAttributes = {
  antialias: false,
  preserveDrawingBuffer: true,
  alpha: false,
  depth: true,
  stencil: false,
  powerPreference: "high-performance",
};
const glContext =
  canvas.getContext("webgl2", contextAttributes) ||
  canvas.getContext("webgl", contextAttributes) ||
  canvas.getContext("experimental-webgl", contextAttributes);

if (!glContext) {
  throw new Error("Unable to acquire WebGL context");
}

const renderer = new THREE.WebGLRenderer({
  canvas,
  context: glContext,
  antialias: false,
  preserveDrawingBuffer: true,
  alpha: false,
});
renderer.setPixelRatio(1.0);
renderer.outputColorSpace = THREE.SRGBColorSpace;

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(55, 16 / 9, 0.1, 4000);
camera.up.set(0, -1, 0); // Scene is Y-down, so -Y is Up
const spark = new SparkRenderer({ renderer });
camera.add(spark);
scene.add(camera);

let runtimeConfig = null;
let splatMesh = null;
let lastShotIndex = 0;

const smoothstep = (edge0, edge1, x) => {
  const t = Math.min(Math.max((x - edge0) / (edge1 - edge0), 0.0), 1.0);
  return t * t * (3.0 - 2.0 * t);
};

function resizeRenderer(width, height) {
  renderer.setSize(width, height, false);
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
}

async function loadSplat(url) {
  if (splatMesh) {
    scene.remove(splatMesh);
    splatMesh.dispose?.();
  }
  return new Promise((resolve) => {
    splatMesh = new SplatMesh({
      url,
      onLoad: () => resolve(true),
    });
    scene.add(splatMesh);
  });
}

function findShot(time) {
  if (!runtimeConfig?.camera?.shots?.length) {
    return null;
  }
  const shots = runtimeConfig.camera.shots;
  const lastShot = shots[shots.length - 1];
  const clamped = Math.max(0, Math.min(time, lastShot.endTime));
  let shot = shots[lastShotIndex];
  const isLast = shot === lastShot;

  if (!(shot && clamped >= shot.startTime && (isLast ? clamped <= shot.endTime : clamped < shot.endTime))) {
    shot = shots.find((s, idx) => {
      const isLastS = idx === shots.length - 1;
      return clamped >= s.startTime && (isLastS ? clamped <= s.endTime : clamped < s.endTime);
    }) ?? lastShot;
    lastShotIndex = shots.indexOf(shot);
  }
  return { shot, time: clamped };
}

function interpolateKeyframes(shot, time) {
  const frames = shot.keyframes;
  if (time <= frames[0].time) {
    return frames[0];
  }
  if (time >= frames[frames.length - 1].time) {
    return frames[frames.length - 1];
  }
  for (let i = 0; i < frames.length - 1; i += 1) {
    const a = frames[i];
    const b = frames[i + 1];
    if (time >= a.time && time <= b.time) {
      const span = b.time - a.time || 1e-3;
      let t = (time - a.time) / span;
      if (shot.ease === "smoothstep") {
        t = smoothstep(0.0, 1.0, t);
      }
      const position = a.position.map((val, idx) => val * (1 - t) + b.position[idx] * t);
      const target = a.target.map((val, idx) => val * (1 - t) + b.target[idx] * t);
      return { position, target };
    }
  }
  return frames[frames.length - 1];
}

window.applyConfig = async (config) => {
  runtimeConfig = config;
  lastShotIndex = 0;
  const { width, height } = config.render.resolution;
  resizeRenderer(width, height);
  camera.fov = config.camera.fov ?? 55;
  camera.updateProjectionMatrix();
  await loadSplat(config.scene.plyUrl);
  return true;
};

window.renderFrame = (timeSeconds) => {
  if (!runtimeConfig || !splatMesh?.isInitialized) {
    return false;
  }
  const entry = findShot(timeSeconds);
  if (!entry) {
    return false;
  }
  const pose = interpolateKeyframes(entry.shot, entry.time);
  camera.position.fromArray(pose.position);
  const lookTarget = new THREE.Vector3().fromArray(pose.target);
  camera.lookAt(lookTarget);
  renderer.render(scene, camera);
  return true;
};

window.addEventListener("resize", () => {
  if (!runtimeConfig) {
    return;
  }
  const { width, height } = runtimeConfig.render.resolution;
  resizeRenderer(width, height);
});
