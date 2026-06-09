const https = require("https");
const sharp = require("sharp");
const path = require("path");
const fs = require("fs");
require("dotenv").config({ path: path.join(__dirname, "..", ".env.local") });

const API_KEY = process.env.RUNPOD_API_KEY;
const ENDPOINT_ID = process.env.RUNPOD_ENDPOINT_ID;
const MAX_WAIT_SEC = 120;

const SIZES = [
  { w: 100, h: 75, label: "small" },
  { w: 512, h: 384, label: "medium" },
  { w: 1000, h: 750, label: "HD" },
  { w: 2000, h: 1500, label: "FHD" },
  { w: 3000, h: 2250, label: "3K" },
  { w: 4000, h: 3000, label: "4K" },
  { w: 5000, h: 3750, label: "5K" },
  { w: 6000, h: 4000, label: "large" },
  { w: 8000, h: 4500, label: "8K wide" },
];

function apiRequest(method, path, body) {
  return new Promise((resolve, reject) => {
    const data = body ? JSON.stringify(body) : null;
    const opts = {
      hostname: "api.runpod.ai",
      port: 443,
      path,
      method,
      headers: {
        Authorization: `Bearer ${API_KEY}`,
        "Content-Type": "application/json",
      },
    };
    if (data) opts.headers["Content-Length"] = Buffer.byteLength(data);
    const req = https.request(opts, (res) => {
      let resp = "";
      res.on("data", (c) => (resp += c));
      res.on("end", () => {
        try {
          resolve({ status: res.statusCode, data: JSON.parse(resp) });
        } catch {
          resolve({ status: res.statusCode, data: resp });
        }
      });
    });
    req.on("error", reject);
    if (data) req.write(data);
    req.end();
  });
}

function delay(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

async function waitForJob(jobId, timeoutSec) {
  const start = Date.now();
  for (let i = 0; i < timeoutSec * 2; i++) {
    const { data } = await apiRequest("GET", `/v2/${ENDPOINT_ID}/status/${jobId}`);
    if (data.status === "COMPLETED") return { ok: true, data };
    if (data.status === "FAILED") return { ok: false, error: data.error || "FAILED", data };
    if (data.status === "CANCELLED") return { ok: false, error: "CANCELLED", data };
    await delay(500);
  }
  return { ok: false, error: "timeout" };
}

async function testSize(size) {
  const { w, h, label } = size;
  console.log(`\n--- Testing ${label} (${w}x${h}) ---`);

  const buf = await sharp({
    create: { width: w, height: h, channels: 3, background: { r: 100, g: 150, b: 200 } },
  })
    .jpeg({ quality: 85 })
    .toBuffer();

  const base64 = buf.toString("base64");
  const kb = (buf.length / 1024).toFixed(1);

  console.log(`  base64 size: ${kb}KB (raw: ${((buf.length * 4 / 3 + buf.length / 96) / 1024).toFixed(1)}KB)`);

  const body = {
    input: {
      source_image: base64,
      model: "RealESRGAN_x4plus",
      scale: 4,
      face_enhance: false,
      half: false,
    },
  };

  const start = Date.now();

  const { status, data } = await apiRequest("POST", `/v2/${ENDPOINT_ID}/runsync`, body);

  const elapsed = ((Date.now() - start) / 1000).toFixed(1);

  if (status === 200 && data.status === "COMPLETED") {
    const hasOutput = data.output && (data.output.image || data.output.image_url);
    const outputSize = hasOutput ? `${(data.output.image?.length / 1024 || 0).toFixed(1)}KB` : "none";
    console.log(`  ✅ PASS (${elapsed}s) output=${outputSize}`);
    return { pass: true, elapsed, error: null };
  }

  const errMsg = data?.error || data?.detail || data?.message || JSON.stringify(data);
  console.log(`  ❌ FAIL (${elapsed}s) status=${status} error=${errMsg.slice(0, 200)}`);
  return { pass: false, elapsed, error: errMsg, data };
}

async function main() {
  console.log("=".repeat(60));
  console.log("RunPod Upscaler Scale Test");
  console.log("=".repeat(60));
  console.log(`Endpoint: ${ENDPOINT_ID}`);
  console.log(`Sizes: ${SIZES.length}`);
  console.log("=".repeat(60));

  const results = [];
  for (const size of SIZES) {
    const r = await testSize(size);
    results.push({ ...size, ...r });
  }

  console.log("\n" + "=".repeat(60));
  console.log("RESULTS");
  console.log("=".repeat(60));
  for (const r of results) {
    const icon = r.pass ? "✅" : "❌";
    const time = r.elapsed ? `(${r.elapsed}s)` : "";
    const err = r.error ? ` ${r.error.slice(0, 100)}` : "";
    console.log(`  ${icon} ${r.label} (${r.w}x${r.h}) ${time}${err}`);
  }

  const passed = results.filter((r) => r.pass).length;
  console.log(`\n${passed}/${results.length} passed`);

  if (passed < results.length) {
    const firstFail = results.find((r) => !r.pass);
    console.log(`\nFirst failure at: ${firstFail.label} (${firstFail.w}x${firstFail.h})`);
    console.log(`Error: ${firstFail.error}`);
  }
}

main().catch(console.error);
