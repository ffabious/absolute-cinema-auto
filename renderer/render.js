import fs from "fs";
import http from "http";
import path from "path";
import process from "process";

import finalhandler from "finalhandler";
import serveStatic from "serve-static";
import { webkit } from "playwright";

const args = parseArgs(process.argv.slice(2));
const configPath = path.resolve(args.config || "../outputs/scene_1/runtime_config.json");
const assetRoot = path.resolve(args.assetRoot || path.join(path.dirname(configPath), "..", ".."));

if (!fs.existsSync(configPath)) {
  throw new Error(`Renderer config not found at ${configPath}`);
}
const config = JSON.parse(fs.readFileSync(configPath, "utf-8"));

const framesDir = path.resolve(config.output.framesDir);
fs.mkdirSync(framesDir, { recursive: true });

async function main() {
  const staticServer = await startStaticServer(assetRoot);
  const serverUrl = `http://127.0.0.1:${staticServer.port}`;
  const browser = await webkit.launch({ headless: true });
  const context = await browser.newContext({
    viewport: {
      width: config.render.resolution.width,
      height: config.render.resolution.height,
    },
    deviceScaleFactor: 1,
  });
  try {
    const page = await context.newPage();
    page.on("console", (msg) => console.log("[browser]", msg.text()));
    page.on("pageerror", (err) => console.error("[pageerror]", err));
    const target = `${serverUrl}/renderer/spark_scene.html`;
    await page.goto(target, { waitUntil: "domcontentloaded", timeout: 120000 });
    await page.waitForFunction(() => typeof window.applyConfig === "function", null, { timeout: 60000 });
    await page.evaluate((cfg) => window.applyConfig(cfg), config);
    await page.waitForFunction(() => typeof window.renderFrame === "function", null, { timeout: 60000 });

    const fps = config.render.fps;
    const totalFrames = config.timeline.totalFrames ?? Math.ceil(config.timeline.duration * fps) + 1;
    const prefix = config.output.framePrefix;
    const padding = config.output.framePadding;

    const startTime = Date.now();
    let lastUpdateTime = startTime;

    for (let frame = 0; frame < totalFrames; frame += 1) {
      const t = frame / fps;
      await page.evaluate((time) => window.renderFrame(time), t);
      const filename = `${prefix}${String(frame).padStart(padding, "0")}.png`;
      const outputPath = path.join(framesDir, filename);
      await page.screenshot({ path: outputPath });

      // Update progress display (throttled to avoid excessive console updates)
      const now = Date.now();
      if (now - lastUpdateTime >= 500 || frame === totalFrames - 1) {
        lastUpdateTime = now;
        const elapsed = (now - startTime) / 1000;
        const progress = (frame + 1) / totalFrames;
        const eta = progress > 0 ? (elapsed / progress) * (1 - progress) : 0;
        const rate = elapsed > 0 ? (frame + 1) / elapsed : 0;

        // Build progress bar
        const barWidth = 30;
        const filled = Math.round(barWidth * progress);
        const bar = "█".repeat(filled) + "░".repeat(barWidth - filled);

        // Format times
        const elapsedStr = formatTime(elapsed);
        const etaStr = formatTime(eta);
        const percent = (progress * 100).toFixed(1);

        // Clear line and write progress
        process.stdout.write(`\r${bar} ${percent}% | ${frame + 1}/${totalFrames} | ${rate.toFixed(1)} fps | ${elapsedStr}<${etaStr}`);
      }
    }
    // Final newline after progress bar
    console.log();
  } finally {
    await browser.close();
    await staticServer.stop();
  }
}

async function startStaticServer(rootDir) {
  const serve = serveStatic(rootDir);
  const server = http.createServer((req, res) => {
    serve(req, res, finalhandler(req, res));
  });
  await new Promise((resolve) => server.listen(0, resolve));
  const { port } = server.address();
  return {
    port,
    stop: () => new Promise((resolve) => server.close(resolve)),
  };
}

function parseArgs(argv) {
  const out = {};
  for (let i = 0; i < argv.length; i += 1) {
    const token = argv[i];
    if (token === "--config") {
      out.config = argv[++i];
    } else if (token === "--asset-root") {
      out.assetRoot = argv[++i];
    }
  }
  return out;
}

function formatTime(seconds) {
  if (!isFinite(seconds) || seconds < 0) return "??:??";
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${String(mins).padStart(2, "0")}:${String(secs).padStart(2, "0")}`;
}

main().catch((err) => {
  console.error("Renderer failed", err);
  process.exit(1);
});
