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
const tmpRoot = fs.mkdtempSync(path.join(os.tmpdir(), "sa3-first-use-"));
const fixtureRoot = path.join(tmpRoot, "lab");
const fixturesDir = path.join(tmpRoot, "fixtures");
const screenshotDir = process.env.SA3_FIRST_USE_SCREENSHOT_DIR || path.join(os.tmpdir(), "sa3-native-lab-first-use-screenshots");
const apiPort = Number(process.env.SA3_FIRST_USE_API_PORT || await freePort());
const frontendPort = Number(process.env.SA3_FIRST_USE_FRONTEND_PORT || await freePort());
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
  await expect(page.locator(".surface-head .eyebrow", { hasText: "Current Sound" })).toBeVisible();
  await expect(page.getByLabel("Gestures")).toBeVisible();
  const makeButton = page.locator(".gesture-buttons button", { hasText: "Make" }).first();
  await expect(makeButton).toBeVisible();
  await expect(makeButton).toBeEnabled();
  await expect(page.getByLabel("Make tune")).toBeVisible();
  await expect(page.locator(".tune-details > summary", { hasText: "Tune" })).toBeVisible();

  await expect(page.locator(".settings-panel #api-base")).not.toBeVisible();
  await expect(page.locator(".specimen .inspect-drawer > summary", { hasText: "Inspect sound" })).toBeVisible();

  const pendingPanel = page.locator(".pending-takes-panel");
  await expect(pendingPanel).toContainText("Pending Takes");
  await expect(pendingPanel).toContainText("No take landed");
  await expect(pendingPanel).toContainText("Recover or reselect the source material, then retry.");
  await expect(pendingPanel).toContainText("Adjust Tune");
  await expect(pendingPanel.locator(".inspect-mini summary").first()).toHaveText("Inspect");

  await page.getByRole("button", { name: /Warm Smoke Take/ }).first().click();
  await expect(page.getByRole("heading", { name: "Warm Smoke Take" })).toBeVisible();
  const nextPanel = page.locator(".next-actions-panel");
  await expect(nextPanel).toBeVisible();
  await expect(nextPanel).toContainText("Continue");
  await expect(nextPanel).toContainText("Vary");
  await expect(nextPanel).toContainText("Encode");
  await expect(nextPanel).toContainText("Remember");

  await nextPanel.getByRole("button", { name: /Continue/ }).click();
  await expect(page.getByRole("heading", { name: "Continue" })).toBeVisible();
  await expect(page.getByLabel("Continue tune")).toBeVisible();

  await nextPanel.getByRole("button", { name: /Remember/ }).click();
  await expect(page.getByRole("heading", { name: "Remember" })).toBeVisible();
  await expect(page.getByRole("button", { name: "Remember current sound" })).toBeVisible();

  await page.locator(".specimen").getByRole("button", { name: "Branch" }).click();
  await expect(page.locator(".fork-panel")).toContainText("Branch Gesture");
  await expect(page.locator(".fork-panel")).toContainText("Branch with changes");

  await page.locator(".session-tray summary", { hasText: "Memory" }).click();
  const memoryRow = page.locator(".session-tray .session-artifact", { hasText: "Archived Smoke Take" }).first();
  await expect(memoryRow).toBeVisible();
  await memoryRow.getByRole("button", { name: "Use as Source" }).click();
  await expect(page.getByRole("heading", { name: "Archived Smoke Take" })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Continue" })).toBeVisible();
  await expect(page.locator(".compare-panel .compare-slot").nth(1)).toContainText("Archived Smoke Take");

  await memoryRow.getByRole("button", { name: "Anchor" }).click();
  await expect(page.locator(".compare-panel .compare-slot").nth(0)).toContainText("Archived Smoke Take");

  await memoryRow.getByRole("button", { name: "Recover" }).click();
  await expect.poll(async () => {
    const item = await artifact(apiBase, fixture.archived_artifact_id);
    return item.session_id ?? "null";
  }, { timeout: 6000 }).toBe(fixture.session_id);

  await page.getByRole("button", { name: /Archived Smoke Take/ }).first().click();
  await expect(page.locator(".session-tray .session-artifact", { hasText: "Archived Smoke Take" }).first()).toContainText("Remember");

  fs.mkdirSync(screenshotDir, { recursive: true });
  const desktopScreenshot = path.join(screenshotDir, "first-use-desktop.png");
  const mobileScreenshot = path.join(screenshotDir, "first-use-mobile.png");
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

async function artifact(baseUrl, artifactId) {
  let lastStatus = "none";
  for (let attempt = 0; attempt < 8; attempt += 1) {
    const response = await fetch(`${baseUrl}/artifacts/${artifactId}/inspect`);
    lastStatus = String(response.status);
    if (response.ok) return (await response.json()).artifact;
    await new Promise((resolve) => setTimeout(resolve, 120));
  }
  throw new Error(`inspect failed ${lastStatus}`);
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
