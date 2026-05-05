# Whitespace Map

Interactive 2-D map of ~2,000 highly-cited neuroscience papers. Paste an abstract; your flag lands in the literature. Hover any dot to see the paper. Click for the abstract.

The pedagogical point: research questions live in a high-dimensional space, much of which is unoccupied — and 2-D projections like UMAP / t-SNE distort that occupancy in ways worth feeling.

## Try it

```
npm install
npm run dev
```

Open the local URL Vite prints. The committed `data/corpus.json` is used by default — no API calls at runtime.

## Rebuild the corpus from OpenAlex

```
npm run data:fetch     # 1 req/sec, polite pool, ~30s
npm run data:corpus    # tokenize, TF-IDF, UMAP, cluster, c-TF-IDF labels
```

`npm run data` does both. Output is written to `data/corpus.json`.

## Stack

- **Build**: Node + `umap-js` + `density-clustering` + custom TF-IDF (~50 lines)
- **Runtime**: React + plain SVG (no d3 dependency in the deliverable). Single component file `app/src/WhitespaceMap.jsx`
- **Similarity**: cosine on sublinear-TF / smooth-IDF / L2-normalized sparse vectors
- **Layout**: UMAP fit at build time (cosine, nNeighbors=8, minDist=0.0), shipped as 2-D coords
- **Runtime projection**: cosine-weighted KNN of paper TF-IDF vectors → flag position. Avoids shipping a fitted UMAP model.
- **Clusters**: DBSCAN on the 2-D UMAP, c-TF-IDF labels with generic-term filtering

## Files

```
build/         Node-only data pipeline (fetch_openalex.js, pipeline.js)
shared/        tokenize.js + tfidf.js (mirror of the runtime versions inside WhitespaceMap.jsx)
app/           index.html + src/main.jsx (test harness) + src/WhitespaceMap.jsx (deliverable)
data/          corpus.json (committed)
cache/         papers_raw.json (gitignored, regenerable)
```

## The deliverable

`app/src/WhitespaceMap.jsx` is a single React component intended to drop into a multi-screen activity in the C4R framework. It uses bare imports (`react` only) and inline-styled component replicas (`InstructionCard`, etc.) marked with `// TODO: Replace with shared MUI component`.

It accepts:

- `corpus` (object) **or** `corpusUrl` (string, default `/corpus.json`)
- `onInteract` (callback, fires on first ≥1 in-vocab token; for host-side `canProceed` gating)

The live-typing-to-flag motion is a deliberate exception to the C4R layout-stability rule (marked in code), because watching your question move through the literature is the lesson.
