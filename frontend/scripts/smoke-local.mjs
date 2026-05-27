import { chromium, expect } from "@playwright/test";

const baseUrl = process.env.SA3_FRONTEND_URL ?? "http://127.0.0.1:5173/";

const browser = await chromium.launch({ headless: true });
const page = await browser.newPage({ viewport: { width: 1440, height: 960 } });

try {
  await page.goto(baseUrl, { waitUntil: "networkidle", timeout: 30000 });
  await expect(page.getByText("SA3 Native Lab", { exact: true })).toBeVisible();
  await expect(page.locator(".surface-head .eyebrow", { hasText: "Listening Bench" })).toBeVisible();
  await expect(page.getByText("Spec covered").first()).toBeVisible();
  await expect(page.getByRole("button", { name: "Save annotation" })).toBeVisible();
  await page.getByRole("button", { name: "A2A" }).click();
  await expect(page.getByText("Init noise")).toBeVisible();
  await expect(page.getByText("Spec covered").first()).toBeVisible();
  await page.getByRole("button", { name: "Inpaint" }).click();
  await expect(page.getByText("Inpaint start")).toBeVisible();
  await expect(page.getByText("Spec covered").first()).toBeVisible();
  await expect(page.getByRole("button", { name: /Play|Pause/ }).first()).toBeVisible();
  await expect(page.getByRole("slider", { name: "Audio position" }).first()).toBeVisible();
  await expect(page.getByRole("button", { name: "New" })).toBeVisible();
  await expect(page.locator("summary", { hasText: "Archive" })).toBeVisible();
  console.log(JSON.stringify({ ok: true, title: await page.title(), url: page.url() }));
} finally {
  await browser.close();
}
