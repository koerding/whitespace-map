# Whitespace Map for Neuroscience — Build Spec

## Goal

Build an interactive web app that teaches **whitespace analysis** to neuroscientists. The user sees a 2D UMAP of ~1000 neuroscience papers as a scatter plot. They type a research question into a text field; their "flag" moves in real time onto the map, showing where their question lives relative to existing literature. Hovering individual papers reveals their titles and abstracts, letting the user reverse-engineer what each region of the map is about. Cluster labels float as ghosted text over each region. The five nearest papers to the user's flag pulse and connect to it with thin lines. If the user is far from any cluster, the app tells them they may be in genuine whitespace.

The pedagogical point: **research questions live in a high-dimensional space, much of which is unoccupied. The map makes occupancy visible.**

## Hard constraints

- **No external API calls at runtime.** All data shipped as a static JSON blob.
- All ML (TF-IDF, UMAP, projection of new text) runs in the browser.
- The data-fetching pipeline runs once, in a Node build script.
- Pure static site at runtime — works on any static host.

## Architecture

```
/build          # Node scripts: pull OpenAlex, fit UMAP, write corpus.json
/app            # Static frontend (vanilla JS or React, your call)
/data           # corpus.json lives here, served as static asset
```

## Tech stack

- **`umap-js`** (PAIR-code) for both fitting and runtime `.transform()` of new points
- Plain JS TF-IDF (write it; it's ~30 lines)
- **D3** for the scatter plot and animations, or plain Canvas if perf matters
- **`density-clustering`** npm package for HDBSCAN to find clusters
- c-TF-IDF (BERTopic-style) for cluster labeling — compute in build script
- **OpenAlex API** for paper data — free, no auth, just include a contact email in User-Agent

## Build in two phases

### Phase 1: synthetic corpus (do this first)

Before pulling real data, build the entire pipeline against a synthetic corpus so UX can be iterated quickly.

- Generate ~200 fake "papers" with 4–5 hand-tagged clusters (e.g., visual cortex, motor learning, hippocampus/memory, neurogenesis, ion channels).
- For each cluster, draw abstracts from a templated vocabulary specific to that cluster, with some shared filler words.
- Run the full pipeline on this synthetic corpus end-to-end.
- Verify the app works, the clusters are visible, the typing-to-flag interaction feels good, the hover reveals the right info.
- Only then move on to phase 2.

### Phase 2: real corpus

- Query OpenAlex for ~1000 neuroscience papers.
- Mix: **700 most-cited** in the neuroscience concept tree + **300 recent** (last 3 years) sampled across concepts. The recent ones are crucial — citation count is biased *against* whitespace by construction, so the recent sample lets the map show emerging/sparse regions honestly.
- Required fields: title, abstract, year, authors, citation count, concept tags. Filter out papers without abstracts.
- Be polite to OpenAlex: include a `mailto:` in the User-Agent header.

## Build pipeline (Node script)

1. Load corpus (synthetic or real).
2. Tokenize titles + abstracts: lowercase, strip punctuation, remove English stopwords plus academic filler (`study`, `results`, `here`, `show`, `paper`, `propose`, `using`, etc.). Light stemming optional.
3. Build vocabulary, capped at top ~3000 terms by document frequency. Drop terms appearing in fewer than 3 documents or more than 50% of documents.
4. Compute IDF weights for each vocab term.
5. Compute sparse TF-IDF vector per paper.
6. Fit `umap-js` on the TF-IDF matrix: `nNeighbors=15`, `minDist=0.1`, `nComponents=2`.
7. Run HDBSCAN on the resulting 2D coordinates to identify clusters.
8. For each cluster: compute c-TF-IDF (treat the cluster as one concatenated document, run TF-IDF *across* clusters). Take top 3 terms as the cluster label.
9. Serialize `corpus.json`:
   ```
   {
     paperCoords: [[x, y], ...],            // UMAP positions
     paperMeta:   [{title, year, authors, citationCount, abstract, clusterId}, ...],
     vocabulary:  [term, ...],
     idfWeights:  [w, ...],
     clusters:    [{id, centroid: [x,y], terms: [...], paperIndices: [...]}, ...]
   }
   ```

## Runtime UMAP transform — design decision

`umap-js` doesn't have a built-in serialize/deserialize for fitted models. Two options:

**Option A (recommended): fit in browser at startup.** Ship the TF-IDF matrix, fit UMAP on first load (~10–30s for 1000 papers), cache the fitted model + result coords in IndexedDB. Subsequent loads are instant. Once fitted, `umap.transform()` for new typed text is a simple call.

**Option B: weighted-KNN approximation.** Save only the final coords. At runtime, project the user's text into TF-IDF space, find the k=10 nearest papers, place the flag at the cosine-weighted centroid of their UMAP coords. Not true UMAP transform, but geometrically reasonable and avoids the fitting wait.

Start with A. Fall back to B if startup latency is unacceptable.

## Runtime app

On load:
- Fetch `corpus.json`.
- Either fit UMAP in browser (Option A) or just plot the coords (Option B).
- Render scatter plot of all papers, colored by cluster.
- Float cluster labels (top 3 terms) as ghosted text over each cluster centroid.
- Compute and store local paper density (kernel density on UMAP coords) — used later to highlight whitespace.

User interaction:
- Big input field at top: *"Type your research question..."*
- On every keystroke (debounced ~100ms):
  - Tokenize input (same pipeline as build).
  - Build TF-IDF vector using shipped vocab + IDF.
  - Call `umap.transform([vector])` → get `[x, y]`.
  - Animate flag to new position (CSS transition or D3 transition, ~200ms).
  - Compute cosine distance in TF-IDF space to all papers; identify 5 nearest.
  - Highlight 5 nearest as pulsing gold dots with thin lines to the flag.
  - Display below the input: `nearest paper: 0.71 cosine, in the [place cells / hippocampus] region`.
- Hover any paper → tooltip with title, year, authors, citation count, abstract snippet.
- If nearest paper distance exceeds threshold (e.g., cosine > 0.85), the flag changes to a flag-with-question-mark and a banner appears: *"You may be in genuine whitespace."*

Optional polish:
- "Show me whitespace" toggle: shade low-density regions of the map.
- Save/share: encode the user's question in the URL hash so they can share their flag.
- An honest disclaimer near the map: *"UMAP preserves local but not global structure. Visual distance is approximate; cosine distance to specific papers is the ground truth."*

## Visual design

- Match a clean, slightly playful aesthetic — this is for scientists but should feel exciting, not like a dashboard.
- Plenty of whitespace (literally) on the canvas. Subtle paper dots. Bold flag.
- Cluster labels in a light gray, italic, large-ish.
- Color palette: muted base for papers, single saturated color (e.g., red or magenta) for the flag, gold for the 5 nearest neighbors.
- The "you are in whitespace" state should feel like a small reward, not a warning.

## Suggested execution order

1. Project scaffold + install deps (`umap-js`, `density-clustering`, `d3`, dev server).
2. Write synthetic corpus generator (~200 fake papers, 4–5 clusters).
3. Write build pipeline: tokenize → TF-IDF → fit UMAP → HDBSCAN → c-TF-IDF labels → serialize. Run on synthetic data.
4. Build runtime app against the synthetic `corpus.json`. Get the typing-to-flag UX feeling great.
5. Add the 5-nearest highlighting + cosine readout + whitespace banner.
6. Add hover tooltips and cluster labels.
7. Replace synthetic with real OpenAlex pull. Tune.
8. Deploy as static site.

## Notes for Claude Code

- Use TypeScript if you want, but plain JS is fine — the codebase is small.
- Keep the runtime app in a single small bundle. No SSR. Vite is a clean dev server choice.
- Test with synthetic data first. Don't pull from OpenAlex until the app feels right.
- The cluster labels are 80% of what makes the map feel alive — get c-TF-IDF working well before polishing anything else.
