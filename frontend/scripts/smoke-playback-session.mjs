import { spawn, execFileSync } from "node:child_process";
import fs from "node:fs";
import net from "node:net";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { chromium, expect } from "@playwright/test";

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const frontendDir = path.resolve(scriptDir, "..");
const repoRoot = path.resolve(frontendDir, "..");
const keepArtifacts = process.env.SA3_KEEP_PLAYWRIGHT_FIXTURE === "1";
const tmpRoot = fs.mkdtempSync(path.join(os.tmpdir(), "sa3-playback-session-"));
const fixtureRoot = path.join(tmpRoot, "lab");
const fixturesDir = path.join(tmpRoot, "fixtures");
const screenshotDir = process.env.SA3_SMOKE_SCREENSHOT_DIR || path.join(os.tmpdir(), "sa3-native-lab-smoke-screenshots");
const apiPort = Number(process.env.SA3_SMOKE_API_PORT || await freePort());
const frontendPort = Number(process.env.SA3_SMOKE_FRONTEND_PORT || await freePort());
const apiBase = `http://127.0.0.1:${apiPort}`;
const frontendUrl = `http://127.0.0.1:${frontendPort}/`;
const children = [];

let browser;

try {
  const fixture = JSON.parse(execFileSync(
    "uv",
    [
      "run",
      "python",
      "scripts/create_playback_session_fixture.py",
      "--artifact-root",
      fixtureRoot,
      "--fixtures-dir",
      fixturesDir,
      "--json",
    ],
    { cwd: repoRoot, encoding: "utf8" },
  ));

  children.push(spawnProcess("api", "uv", [
    "run",
    "sa3-lab-api",
    "--host",
    "127.0.0.1",
    "--port",
    String(apiPort),
    "--artifact-root",
    fixture.artifact_root,
    "--repo-root",
    repoRoot,
  ], { cwd: repoRoot }));

  await waitForHttp(`${apiBase}/health`);

  children.push(spawnProcess("ui", "npm", [
    "run",
    "dev",
    "--prefix",
    "frontend",
    "--",
    "--port",
    String(frontendPort),
  ], {
    cwd: repoRoot,
    env: { ...process.env, VITE_SA3_API_BASE: apiBase },
  }));

  await waitForHttp(frontendUrl);

  browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({ viewport: { width: 1440, height: 960 } });
  const consoleErrors = [];
  page.on("console", (message) => {
    if (message.type() === "error") consoleErrors.push(message.text());
  });
  page.on("pageerror", (error) => consoleErrors.push(error.message));

  await page.goto(frontendUrl, { waitUntil: "networkidle", timeout: 30000 });
  await page.getByRole("button", { name: /Warm Smoke Take/ }).first().click();
  await expect(page.getByRole("heading", { name: "Warm Smoke Take" })).toBeVisible();
  await expect(page.locator(".operator-surface.has-selection")).toBeVisible();
  await expect(page.locator(".wave-bus.has-sources.has-job.has-family").first()).toBeVisible();

  await page.locator(".specimen").getByRole("button", { name: "Anchor", exact: true }).click();
  await page.getByRole("button", { name: /Source Pulse/ }).first().click();
  await expect(page.getByRole("heading", { name: "Source Pulse" })).toBeVisible();
  await page.locator(".specimen").getByRole("button", { name: "Source", exact: true }).click();
  await expect(page.locator(".compare-panel .compare-slot").nth(0)).toContainText("Warm Smoke Take");
  await expect(page.locator(".compare-panel .compare-slot").nth(1)).toContainText("Source Pulse");

  await page.getByRole("button", { name: /Warm Smoke Take/ }).first().click();
  await expect(page.getByRole("heading", { name: "Warm Smoke Take" })).toBeVisible();
  await page.locator(".annotation-panel").getByLabel("Label").fill("Warm Smoke Take");
  await page.locator(".annotation-panel").getByLabel("Tags").fill("smoke-test, loop, favorite");
  await page.locator(".annotation-panel").getByLabel("Notes").fill("browser annotation note");
  await page.locator(".annotation-panel").getByRole("button", { name: "Save annotation" }).click();

  await expect.poll(async () => {
    const item = await artifact(apiBase, fixture.take_artifact_id);
    return `${item.notes ?? "none"}:${(item.tags ?? []).join("|")}`;
  }, { timeout: 6000 }).toBe("browser annotation note:smoke-test|loop|favorite");

  const deck = page.locator(".specimen .audio-deck").first();
  await expect(deck.locator(".wave-surfer-stage")).toBeVisible({ timeout: 10000 });
  const audioPosition = deck.getByRole("slider", { name: "Audio position" });
  await audioPosition.press("ArrowRight");
  await deck.getByRole("button", { name: "In", exact: true }).click();
  await audioPosition.press("End");
  await deck.getByRole("button", { name: "Out", exact: true }).click();
  await deck.getByRole("button", { name: "Mark", exact: true }).click();
  await deck.getByLabel("Note for M1").fill("brittle seam");
  await expect(deck.getByRole("button", { name: "Save cues" })).toBeEnabled();
  await deck.locator(".deck-volume.zoom input").fill("72");
  await deck.getByRole("button", { name: "Save cues" }).click();

  await expect.poll(async () => {
    const state = await playbackState(apiBase, fixture.take_artifact_id);
    return `${state.markers?.[0]?.note ?? "none"}:${state.loop_region ? "loop" : "none"}`;
  }, { timeout: 6000 }).toBe("brittle seam:loop");

  await page.getByLabel("Sequence").selectOption("lineage");
  await expect(page.locator(".audition-stack article.selected small").first()).toContainText("anchor");
  await page.getByLabel("Sequence").selectOption("open");
  await expect(page.locator(".audition-stack article").first()).toContainText("open");

  const sessionTakeRow = page.locator(".session-tray .session-artifact", { hasText: "Warm Smoke Take" }).first();
  await expect(sessionTakeRow).toBeVisible();
  await sessionTakeRow.getByRole("button", { name: "Archive" }).click();
  await expect.poll(async () => {
    const item = await artifact(apiBase, fixture.take_artifact_id);
    return `${item.session_id ?? "null"}:${item.metadata?.archived_from_session_id ?? "none"}`;
  }, { timeout: 6000 }).toBe(`null:${fixture.session_id}`);

  await page.locator("summary", { hasText: "Archive" }).last().click();
  const archivedRow = page.locator(".session-artifact", { hasText: "Warm Smoke Take" }).first();
  await expect(archivedRow).toBeVisible();
  await archivedRow.getByRole("button", { name: "Recover" }).click();
  await expect.poll(async () => {
    const item = await artifact(apiBase, fixture.take_artifact_id);
    return item.session_id ?? "null";
  }, { timeout: 6000 }).toBe(fixture.session_id);

  fs.mkdirSync(screenshotDir, { recursive: true });
  const desktopScreenshot = path.join(screenshotDir, "playback-session-desktop.png");
  const mobileScreenshot = path.join(screenshotDir, "playback-session-mobile.png");
  await page.screenshot({ path: desktopScreenshot, fullPage: true });
  await page.setViewportSize({ width: 390, height: 844 });
  await page.waitForTimeout(250);
  await page.screenshot({ path: mobileScreenshot, fullPage: true });
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
  if (overflow > 4) throw new Error(`mobile horizontal overflow ${overflow}px`);

  const severeConsoleErrors = consoleErrors.filter((line) => !line.includes("Failed to load resource") && !line.includes("net::ERR_ABORTED"));
  if (severeConsoleErrors.length) throw new Error(`console errors: ${severeConsoleErrors.join(" | ")}`);

  console.log(JSON.stringify({
    ok: true,
    apiBase,
    frontendUrl,
    artifactRoot: fixture.artifact_root,
    screenshots: [desktopScreenshot, mobileScreenshot],
  }, null, 2));
} finally {
  if (browser) await browser.close();
  await Promise.all(children.map(stopProcess));
  if (!keepArtifacts) fs.rmSync(tmpRoot, { recursive: true, force: true });
}

async function playbackState(baseUrl, artifactId) {
  const item = await artifact(baseUrl, artifactId);
  return item.metadata?.playback_state ?? {};
}

async function artifact(baseUrl, artifactId) {
  const response = await fetch(`${baseUrl}/artifacts/${artifactId}/inspect`);
  if (!response.ok) throw new Error(`inspect failed ${response.status}`);
  return (await response.json()).artifact;
}

function spawnProcess(label, command, args, options) {
  const child = spawn(command, args, {
    cwd: options.cwd,
    env: options.env ?? process.env,
    stdio: ["ignore", "pipe", "pipe"],
  });
  child.stdout.on("data", (chunk) => process.stdout.write(`[${label}] ${chunk}`));
  child.stderr.on("data", (chunk) => process.stderr.write(`[${label}] ${chunk}`));
  child.on("exit", (code, signal) => {
    if (code && code !== 0) process.stderr.write(`[${label}] exited with ${code}${signal ? ` (${signal})` : ""}\n`);
  });
  return child;
}

async function stopProcess(child) {
  if (child.exitCode !== null || child.signalCode !== null) return;
  child.kill("SIGINT");
  await new Promise((resolve) => {
    const timer = setTimeout(() => {
      if (child.exitCode === null && child.signalCode === null) child.kill("SIGKILL");
      resolve();
    }, 5000);
    child.once("exit", () => {
      clearTimeout(timer);
      resolve();
    });
  });
}

async function waitForHttp(url, timeoutMs = 30000) {
  const started = Date.now();
  let lastError;
  while (Date.now() - started < timeoutMs) {
    try {
      const response = await fetch(url);
      if (response.status < 500) return;
      lastError = new Error(`HTTP ${response.status}`);
    } catch (error) {
      lastError = error;
    }
    await new Promise((resolve) => setTimeout(resolve, 250));
  }
  throw new Error(`timed out waiting for ${url}: ${lastError?.message ?? lastError}`);
}

async function freePort() {
  return new Promise((resolve, reject) => {
    const server = net.createServer();
    server.unref();
    server.on("error", reject);
    server.listen(0, "127.0.0.1", () => {
      const address = server.address();
      server.close(() => {
        if (address && typeof address === "object") resolve(address.port);
        else reject(new Error("could not allocate port"));
      });
    });
  });
}
