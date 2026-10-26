import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

const appDir = path.dirname(fileURLToPath(import.meta.url));
const datasetRoot = path.resolve(appDir, "../../data/removal_dataset");

function contentType(filePath) {
  if (filePath.endsWith(".png")) return "image/png";
  if (filePath.endsWith(".txt")) return "text/plain; charset=utf-8";
  if (filePath.endsWith(".csv")) return "text/csv; charset=utf-8";
  return "application/octet-stream";
}

function resolveDatasetFile(urlPath) {
  const relativePath = decodeURIComponent(urlPath).replace(/^\/+/, "");
  const resolved = path.resolve(datasetRoot, relativePath);
  const relative = path.relative(datasetRoot, resolved);
  if (relative.startsWith("..") || path.isAbsolute(relative)) return null;
  return resolved;
}

function readSamples() {
  const mappingPath = path.join(datasetRoot, "mapping.csv");
  const lines = fs.readFileSync(mappingPath, "utf8").trim().split(/\r?\n/).slice(1);
  return lines.filter(Boolean).map((line) => {
    const [index, sampleId, reference, targetImage, targetText] = line.split(",");
    const prompt = fs.readFileSync(path.join(datasetRoot, targetText), "utf8").trim();
    return {
      index,
      sampleId,
      reference: `/dataset/${reference}`,
      target: `/dataset/${targetImage}`,
      prompt,
    };
  });
}

function attachDataset(server) {
  server.middlewares.use((req, res, next) => {
    const url = req.url?.split("?")[0] ?? "";
    if (url === "/api/samples") {
      res.setHeader("Content-Type", "application/json; charset=utf-8");
      res.end(JSON.stringify(readSamples()));
      return;
    }
    if (!url.startsWith("/dataset/")) {
      next();
      return;
    }
    const filePath = resolveDatasetFile(url.slice("/dataset/".length));
    if (!filePath || !fs.existsSync(filePath) || !fs.statSync(filePath).isFile()) {
      res.statusCode = 404;
      res.end("Not found");
      return;
    }
    res.setHeader("Content-Type", contentType(filePath));
    fs.createReadStream(filePath).pipe(res);
  });
}

function datasetPlugin() {
  return {
    name: "removal-dataset",
    configureServer: attachDataset,
    configurePreviewServer: attachDataset,
  };
}

export default defineConfig({
  plugins: [react(), datasetPlugin()],
  server: {
    port: 5173,
    strictPort: true,
  },
});
