import { readFileSync, writeFileSync, existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';
import { UMAP } from 'umap-js';
import seedrandom from 'seedrandom';
import clustering from 'density-clustering';

import { tokenize, termCounts } from '../shared/tokenize.js';
import { fitTfidf, denseFromSparse, cosineSparse } from '../shared/tfidf.js';

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(__dirname, '..');
const RAW_PATH = resolve(ROOT, 'cache', 'papers_raw.json');
const OUT_PATH = resolve(ROOT, 'data', 'corpus.json');

const UMAP_PARAMS = { nNeighbors: 10, minDist: 0.05, nComponents: 2 };
const VOCAB_PARAMS = { maxVocab: 3000, minDf: 3, maxDfRatio: 0.5 };
const DBSCAN_MIN_PTS = 12;
const LABEL_TERM_COUNT = 3;
const LABEL_GENERIC_FRACTION = 0.5;
const SEED = 'whitespace-v1';

function loadRaw() {
  if (!existsSync(RAW_PATH)) {
    console.error(`Missing ${RAW_PATH}. Run \`npm run fetch\` first.`);
    process.exit(1);
  }
  return JSON.parse(readFileSync(RAW_PATH, 'utf-8'));
}

function fitUmap(vectors, vocabSize) {
  const dense = vectors.map(v => Array.from(denseFromSparse(v, vocabSize)));
  const rng = seedrandom(SEED);
  const umap = new UMAP({
    ...UMAP_PARAMS,
    distanceFn: cosineDense,
    random: rng
  });
  const coords = umap.fit(dense);
  return coords;
}

function cosineDense(a, b) {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  if (na === 0 || nb === 0) return 1;
  return 1 - dot / Math.sqrt(na * nb);
}

function clusterPapers(coords) {
  const ranges = coords.reduce((r, [x, y]) => {
    if (x < r.minX) r.minX = x; if (x > r.maxX) r.maxX = x;
    if (y < r.minY) r.minY = y; if (y > r.maxY) r.maxY = y;
    return r;
  }, { minX: Infinity, maxX: -Infinity, minY: Infinity, maxY: -Infinity });
  const span = Math.max(ranges.maxX - ranges.minX, ranges.maxY - ranges.minY);
  const eps = span * 0.025;

  const dbscan = new clustering.DBSCAN();
  const labelArrays = dbscan.run(coords, eps, DBSCAN_MIN_PTS);

  const labels = new Array(coords.length).fill(-1);
  labelArrays.forEach((indices, clusterId) => {
    for (const idx of indices) labels[idx] = clusterId;
  });
  return { labels, eps };
}

function ctfIdfLabels(papers, labels, abstracts) {
  const clusterIds = [...new Set(labels)].filter(id => id !== -1).sort((a, b) => a - b);
  if (clusterIds.length === 0) return new Map();

  const tokensPerCluster = new Map();
  for (const id of clusterIds) tokensPerCluster.set(id, new Map());

  for (let i = 0; i < papers.length; i++) {
    const id = labels[i];
    if (id === -1) continue;
    const counts = termCounts(tokenize(abstracts[i]));
    const bucket = tokensPerCluster.get(id);
    for (const [t, c] of counts) bucket.set(t, (bucket.get(t) || 0) + c);
  }

  const dfPerTerm = new Map();
  for (const bucket of tokensPerCluster.values()) {
    for (const t of bucket.keys()) dfPerTerm.set(t, (dfPerTerm.get(t) || 0) + 1);
  }

  const N = clusterIds.length;
  const topTermsByCluster = new Map();
  const topNCandidates = LABEL_TERM_COUNT * 4;

  for (const [id, bucket] of tokensPerCluster) {
    const total = [...bucket.values()].reduce((a, b) => a + b, 0) || 1;
    const scored = [];
    for (const [term, count] of bucket) {
      const tf = count / total;
      const idf = Math.log(N / dfPerTerm.get(term));
      scored.push([term, tf * idf]);
    }
    scored.sort((a, b) => b[1] - a[1]);
    topTermsByCluster.set(id, scored.slice(0, topNCandidates).map(([t]) => t));
  }

  const genericLimit = Math.ceil(LABEL_GENERIC_FRACTION * N);
  const termClusterCount = new Map();
  for (const terms of topTermsByCluster.values()) {
    for (const t of terms.slice(0, 10)) {
      termClusterCount.set(t, (termClusterCount.get(t) || 0) + 1);
    }
  }
  const generic = new Set([...termClusterCount].filter(([, c]) => c > genericLimit).map(([t]) => t));

  const labelsByCluster = new Map();
  for (const [id, candidates] of topTermsByCluster) {
    const filtered = candidates.filter(t => !generic.has(t));
    labelsByCluster.set(id, (filtered.length >= LABEL_TERM_COUNT ? filtered : candidates).slice(0, LABEL_TERM_COUNT));
  }
  return labelsByCluster;
}

function clusterCentroids(coords, labels) {
  const sums = new Map();
  for (let i = 0; i < coords.length; i++) {
    const id = labels[i];
    if (id === -1) continue;
    if (!sums.has(id)) sums.set(id, { x: 0, y: 0, n: 0 });
    const s = sums.get(id);
    s.x += coords[i][0];
    s.y += coords[i][1];
    s.n++;
  }
  const out = new Map();
  for (const [id, s] of sums) out.set(id, [s.x / s.n, s.y / s.n]);
  return out;
}

function densityGrid(coords, gridSize = 64) {
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  for (const [x, y] of coords) {
    if (x < minX) minX = x;
    if (x > maxX) maxX = x;
    if (y < minY) minY = y;
    if (y > maxY) maxY = y;
  }
  const padX = (maxX - minX) * 0.05;
  const padY = (maxY - minY) * 0.05;
  minX -= padX; maxX += padX; minY -= padY; maxY += padY;
  const grid = new Float32Array(gridSize * gridSize);
  const cellW = (maxX - minX) / gridSize;
  const cellH = (maxY - minY) / gridSize;
  const sigma = Math.max(cellW, cellH) * 1.5;
  const sigma2 = 2 * sigma * sigma;
  const reach = 3;

  for (const [x, y] of coords) {
    const gx = (x - minX) / cellW;
    const gy = (y - minY) / cellH;
    const ix = Math.floor(gx);
    const iy = Math.floor(gy);
    for (let dy = -reach; dy <= reach; dy++) {
      const yy = iy + dy;
      if (yy < 0 || yy >= gridSize) continue;
      for (let dx = -reach; dx <= reach; dx++) {
        const xx = ix + dx;
        if (xx < 0 || xx >= gridSize) continue;
        const cx = minX + (xx + 0.5) * cellW;
        const cy = minY + (yy + 0.5) * cellH;
        const dd = (cx - x) * (cx - x) + (cy - y) * (cy - y);
        grid[yy * gridSize + xx] += Math.exp(-dd / sigma2);
      }
    }
  }
  let maxV = 0;
  for (const v of grid) if (v > maxV) maxV = v;
  if (maxV > 0) for (let i = 0; i < grid.length; i++) grid[i] /= maxV;
  return {
    extent: [[minX, minY], [maxX, maxY]],
    gridSize,
    values: Array.from(grid)
  };
}

function summarizePapers(papers) {
  return papers.map(p => ({
    title: p.title,
    year: p.year,
    authors: p.authors,
    citationCount: p.citationCount,
    abstract: p.abstract.length > 1600 ? p.abstract.slice(0, 1600) + '…' : p.abstract
  }));
}

async function main() {
  console.log('Loading papers_raw.json…');
  const raw = loadRaw();
  const papers = raw.papers;
  console.log(`  ${papers.length} papers loaded`);

  console.log('Tokenizing & fitting TF-IDF…');
  const docs = papers.map(p => `${p.title}\n${p.abstract}`);
  const { vocabulary, idf, vectors } = fitTfidf(docs, VOCAB_PARAMS);
  console.log(`  vocabulary=${vocabulary.length}, avg nnz/doc=${(vectors.reduce((s, v) => s + v.indices.length, 0) / vectors.length).toFixed(1)}`);

  console.log('Fitting UMAP (cosine, this may take a minute)…');
  const t0 = Date.now();
  const coords = fitUmap(vectors, vocabulary.length);
  console.log(`  done in ${((Date.now() - t0) / 1000).toFixed(1)}s`);

  console.log('Clustering (DBSCAN on 2D UMAP)…');
  const { labels, eps } = clusterPapers(coords);
  const numClusters = new Set(labels.filter(l => l !== -1)).size;
  const noiseCount = labels.filter(l => l === -1).length;
  console.log(`  eps=${eps.toFixed(3)}, ${numClusters} clusters, ${noiseCount} noise points`);

  console.log('Computing c-TF-IDF labels…');
  const labelTerms = ctfIdfLabels(papers, labels, papers.map(p => p.abstract));
  for (const [id, terms] of labelTerms) {
    console.log(`  cluster ${id}: ${terms.join(' / ')}`);
  }

  const centroids = clusterCentroids(coords, labels);
  const clusterEntries = [];
  for (const [id, terms] of labelTerms) {
    clusterEntries.push({
      id,
      terms,
      centroid: centroids.get(id),
      paperIndices: labels.map((l, i) => l === id ? i : -1).filter(i => i >= 0)
    });
  }

  console.log('Computing density grid…');
  const density = densityGrid(coords);

  const corpus = {
    version: 1,
    builtAt: new Date().toISOString(),
    paperMeta: summarizePapers(papers),
    paperCoords: coords.map(([x, y]) => [Number(x.toFixed(4)), Number(y.toFixed(4))]),
    paperVectors: vectors.map(v => ({
      indices: v.indices,
      values: v.values.map(x => Number(x.toFixed(5)))
    })),
    paperClusterIds: labels,
    vocabulary,
    idf: idf.map(x => Number(x.toFixed(5))),
    clusters: clusterEntries,
    density
  };

  writeFileSync(OUT_PATH, JSON.stringify(corpus));
  const sizeMB = (Buffer.byteLength(JSON.stringify(corpus)) / 1024 / 1024).toFixed(2);
  console.log(`\nWrote ${OUT_PATH} (${sizeMB} MB)`);
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
