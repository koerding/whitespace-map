/**
 * WhitespaceMap — interactive 2D map of neuroscience literature.
 *
 * A single-file React component matching C4R artifact conventions.
 * The user types a research question; their flag moves in real time
 * onto the map, showing where their question lives relative to the
 * literature. Hovering a paper reveals its title and abstract.
 *
 * Inputs:
 *   - corpus    (object) — pre-built corpus.json shape, OR
 *   - corpusUrl (string) — URL to fetch corpus.json from (default: '/corpus.json')
 *   - onInteract (fn)    — optional callback fired on first meaningful query
 *                          (≥1 in-vocab token); host can use to gate canProceed
 *
 * Build system note:
 *   This component uses bare React imports and inline-styled inline component
 *   replicas (PageContainer, InstructionCard) to match the C4R activity
 *   pattern. The dev team will swap inline components for shared MUI
 *   components in production.
 *
 *   The map screen deliberately violates the C4R "layout stability" rule
 *   (no UI changes from passive input). Live typing -> flag motion is the
 *   pedagogical point of this activity. See // TODO: DEV note in the input
 *   handler below.
 */

import React, { useEffect, useMemo, useRef, useState, useCallback } from 'react';

// ---------------------------------------------------------------------------
// Design tokens (mirrors C4R reference)
// ---------------------------------------------------------------------------
const PRIMARY = {
  purple:       '#6F00FF',
  blue:         '#0082FF',
  yellow:       '#FFC800',
  brightOrange: '#FF5A00',
};
const NEUTRAL = {
  black:         '#202020',
  darkGray:      '#333132',
  midGray:       '#999999',
  intDarkGray:   '#A2A2A2',
  intLightGray:  '#E0E0E0',
  lightGray:     '#F3F3F3',
  white:         '#FFFFFF',
};
const TEXT = {
  title:   { fontSize: '1.2rem', fontWeight: 700, color: NEUTRAL.black },
  heading: { fontSize: '1.2rem', fontWeight: 700, color: NEUTRAL.black },
  body:    { fontSize: '1.0rem', fontWeight: 300, color: NEUTRAL.black, lineHeight: 1.5 },
  caption: { fontSize: '0.8rem', fontWeight: 300, color: NEUTRAL.darkGray, lineHeight: 1.5 },
};
const BORDER_RADIUS = 6;

// ---------------------------------------------------------------------------
// Inline component replicas (TODO: DEV — swap for shared MUI components)
// ---------------------------------------------------------------------------

// TODO: Replace with shared PageContainer component
function PageContainer({ children, maxWidth = '900px', style = {} }) {
  return (
    <div style={{ maxWidth, margin: '0 auto', padding: '24px 16px', ...style }}>
      {children}
    </div>
  );
}

// TODO: Replace with shared InstructionCard component
function InstructionCard({ title, children }) {
  return (
    <div style={{
      marginBottom: 16,
      backgroundColor: '#F0F4FF',
      borderLeft: `4px solid ${PRIMARY.blue}`,
      borderRadius: BORDER_RADIUS,
      padding: 16,
      boxShadow: '0 1px 3px rgba(0,0,0,0.06)'
    }}>
      {title && <div style={{ ...TEXT.heading, marginBottom: 8 }}>{title}</div>}
      <div style={TEXT.body}>{children}</div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Tokenizer + TF-IDF vectorizer (must match shared/tokenize.js + shared/tfidf.js
// used by build/pipeline.js exactly — same vocab/IDF in -> same vectors out).
// ---------------------------------------------------------------------------
const STOPWORDS = new Set([
  'the','and','for','are','but','not','you','all','any','can','had','her','was','one','our','out','his','has','how','its','may','new','now','old','see','two','way','who','boy','did','use','many','then','them','these','some','what','were','when','your','said','each','which','their','about','would','there','could','other','than','more','very','also','from','this','that','have','with','will','been','they','through','during','before','after','above','below','between','under','over','into','onto','upon','only','such','same','those','where','while','here','because','being','both','once','itself','themselves','ourselves','yourself','should','might','must','does','done','doing','having','using','used','show','shown','showed','study','studies','studied','results','result','propose','proposed','paper','papers','data','analysis','method','methods','findings','finding','however','therefore','thus','further','among','within','across','toward','towards','via','per','due','given','overall','various','several','much','most','least','few','either','neither','rather','quite','still','already','though','although','since','until','unless','versus','despite','besides','indeed','perhaps','likely','possibly','generally','typically','specifically','particularly','approximately','roughly','nearly','almost','found','find','finds','present','presented','presents','reported','reports','report','observed','observe','observes','demonstrated','demonstrate','demonstrates','suggest','suggests','suggested','shows','indicate','indicates','indicated'
]);

function tokenize(text) {
  if (!text) return [];
  const out = [];
  for (const m of text.toLowerCase().matchAll(/[a-z]+/g)) {
    const t = m[0];
    if (t.length < 3 || STOPWORDS.has(t)) continue;
    out.push(t);
  }
  return out;
}

function vectorizeQuery(text, termIndex, idf) {
  const counts = new Map();
  for (const t of tokenize(text)) counts.set(t, (counts.get(t) || 0) + 1);
  const indices = [];
  const values = [];
  for (const [term, c] of counts) {
    const idx = termIndex.get(term);
    if (idx === undefined) continue;
    const tfidf = (1 + Math.log(c)) * idf[idx];
    indices.push(idx);
    values.push(tfidf);
  }
  let norm = 0;
  for (const v of values) norm += v * v;
  norm = Math.sqrt(norm);
  if (norm > 0) for (let i = 0; i < values.length; i++) values[i] /= norm;
  const order = indices.map((_, i) => i).sort((a, b) => indices[a] - indices[b]);
  return {
    indices: order.map(i => indices[i]),
    values: order.map(i => values[i])
  };
}

function cosineSparse(a, b) {
  let i = 0, j = 0, dot = 0;
  const ai = a.indices, av = a.values, bi = b.indices, bv = b.values;
  while (i < ai.length && j < bi.length) {
    if (ai[i] === bi[j]) { dot += av[i] * bv[j]; i++; j++; }
    else if (ai[i] < bi[j]) i++;
    else j++;
  }
  return dot;
}

// ---------------------------------------------------------------------------
// KNN-based projection (replaces UMAP transform at runtime).
// Find k nearest paper vectors in TF-IDF space, place flag at the
// cosine-weighted centroid of their pre-computed 2D coords.
// ---------------------------------------------------------------------------
const KNN_K = 10;
const NEAREST_DISPLAY = 5;
const FLAG_WEIGHT_POWER = 4;

function nearestPapers(queryVec, paperVectors, k) {
  if (queryVec.indices.length === 0) return [];
  const sims = new Array(paperVectors.length);
  for (let i = 0; i < paperVectors.length; i++) {
    sims[i] = cosineSparse(queryVec, paperVectors[i]);
  }
  const idx = sims.map((_, i) => i);
  idx.sort((a, b) => sims[b] - sims[a]);
  return idx.slice(0, k).map(i => ({ index: i, similarity: sims[i] }));
}

function projectToCoords(neighbors, paperCoords) {
  if (neighbors.length === 0) return null;
  let wx = 0, wy = 0, wsum = 0;
  for (const { index, similarity } of neighbors) {
    const w = Math.max(0, similarity) ** FLAG_WEIGHT_POWER;
    if (w <= 0) continue;
    const [x, y] = paperCoords[index];
    wx += w * x; wy += w * y; wsum += w;
  }
  if (wsum === 0) {
    const [x, y] = paperCoords[neighbors[0].index];
    return [x, y];
  }
  return [wx / wsum, wy / wsum];
}

// ---------------------------------------------------------------------------
// Scatter plot scales
// ---------------------------------------------------------------------------
function makeScales(coords, width, height, padding = 36) {
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  for (const [x, y] of coords) {
    if (x < minX) minX = x;
    if (x > maxX) maxX = x;
    if (y < minY) minY = y;
    if (y > maxY) maxY = y;
  }
  const dx = (maxX - minX) * 0.05 || 1;
  const dy = (maxY - minY) * 0.05 || 1;
  minX -= dx; maxX += dx; minY -= dy; maxY += dy;
  const sx = v => padding + ((v - minX) / (maxX - minX)) * (width - 2 * padding);
  const sy = v => padding + ((maxY - v) / (maxY - minY)) * (height - 2 * padding);
  return { sx, sy, extent: [[minX, minY], [maxX, maxY]] };
}

// Distinct hues for clusters; -1 (noise) renders muted gray.
const CLUSTER_PALETTE = [
  '#1F77B4', '#FF7F0E', '#2CA02C', '#D62728', '#9467BD',
  '#8C564B', '#E377C2', '#17BECF', '#BCBD22', '#7F7F7F',
  '#3B9BB7', '#E08214', '#5AAE61', '#C2A5CF', '#F46D43'
];
function clusterColor(id) {
  if (id < 0) return NEUTRAL.intDarkGray;
  return CLUSTER_PALETTE[id % CLUSTER_PALETTE.length];
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------
export default function WhitespaceMap({
  corpus: corpusProp,
  corpusUrl = '/corpus.json',
  onInteract,
}) {
  const [corpus, setCorpus] = useState(corpusProp || null);
  const [loadError, setLoadError] = useState(null);
  const [query, setQuery] = useState('');
  const [hovered, setHovered] = useState(null);
  const [selected, setSelected] = useState(null);
  const [debouncedQuery, setDebouncedQuery] = useState('');
  const [width, setWidth] = useState(760);
  const [height, setHeight] = useState(520);
  const wrapperRef = useRef(null);
  const interactedRef = useRef(false);

  useEffect(() => {
    if (!wrapperRef.current) return;
    const update = () => {
      const w = Math.floor(wrapperRef.current.getBoundingClientRect().width);
      const clamped = Math.max(320, w);
      setWidth(clamped);
      setHeight(Math.round(clamped * 0.62));
    };
    update();
    const ro = new ResizeObserver(update);
    ro.observe(wrapperRef.current);
    return () => ro.disconnect();
  }, [corpus]);

  useEffect(() => {
    if (corpusProp) { setCorpus(corpusProp); return; }
    let cancelled = false;
    fetch(corpusUrl)
      .then(r => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json(); })
      .then(c => { if (!cancelled) setCorpus(c); })
      .catch(err => { if (!cancelled) setLoadError(err.message); });
    return () => { cancelled = true; };
  }, [corpusProp, corpusUrl]);

  useEffect(() => {
    const id = setTimeout(() => setDebouncedQuery(query), 200);
    return () => clearTimeout(id);
  }, [query]);

  useEffect(() => {
    if (selected === null) return;
    const onKey = (e) => { if (e.key === 'Escape') setSelected(null); };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [selected]);

  const termIndex = useMemo(() => {
    if (!corpus) return null;
    const m = new Map();
    for (let i = 0; i < corpus.vocabulary.length; i++) m.set(corpus.vocabulary[i], i);
    return m;
  }, [corpus]);

  // Per-keystroke: vectorize query, find K-nearest, project flag.
  // TODO: DEV — this live-update on typing is a deliberate exception to the
  // C4R layout-stability rule. The pedagogical interaction is "watch your
  // question move through the literature as you type".
  const liveProjection = useMemo(() => {
    if (!corpus || !termIndex || !query.trim()) return null;
    const vec = vectorizeQuery(query, termIndex, corpus.idf);
    if (vec.indices.length === 0) return null;
    const nbrs = nearestPapers(vec, corpus.paperVectors, KNN_K);
    const coords = projectToCoords(nbrs, corpus.paperCoords);
    return { vec, neighbors: nbrs, coords };
  }, [query, corpus, termIndex]);

  // Debounced: 5-nearest highlights and region label.
  const debouncedReadout = useMemo(() => {
    if (!corpus || !termIndex || !debouncedQuery.trim()) return null;
    const vec = vectorizeQuery(debouncedQuery, termIndex, corpus.idf);
    if (vec.indices.length === 0) return null;
    const top5 = nearestPapers(vec, corpus.paperVectors, NEAREST_DISPLAY);
    const flagPos = projectToCoords(nearestPapers(vec, corpus.paperVectors, KNN_K), corpus.paperCoords);
    let regionLabel = null;
    if (flagPos && corpus.clusters.length) {
      let best = null, bestD = Infinity;
      for (const c of corpus.clusters) {
        const d = (c.centroid[0] - flagPos[0]) ** 2 + (c.centroid[1] - flagPos[1]) ** 2;
        if (d < bestD) { bestD = d; best = c; }
      }
      if (best) regionLabel = best.terms.join(' / ');
    }
    return { top5, regionLabel, flagPos };
  }, [debouncedQuery, corpus, termIndex]);

  const handleQueryChange = useCallback((e) => {
    setQuery(e.target.value);
    if (!interactedRef.current && e.target.value.trim().length > 0) {
      interactedRef.current = true;
      if (onInteract) onInteract();
    }
  }, [onInteract]);

  if (loadError) {
    return (
      <PageContainer>
        <InstructionCard title="Couldn't load the corpus">
          {loadError}. Make sure <code>corpus.json</code> is reachable at <code>{corpusUrl}</code>.
        </InstructionCard>
      </PageContainer>
    );
  }
  if (!corpus) {
    return (
      <PageContainer>
        <div style={{ ...TEXT.body, color: NEUTRAL.midGray }}>Loading the literature map…</div>
      </PageContainer>
    );
  }

  const scales = makeScales(corpus.paperCoords, width, height);

  return (
    <PageContainer>
      <div ref={wrapperRef}>
      <InstructionCard title="Where does your work live?">
        Paste an abstract into the box below — ideally one of your own (a paper draft, a grant aim, a project idea). Your flag will move onto a 2-D map of {corpus.paperMeta.length.toLocaleString()} highly-cited neuroscience papers as you type or paste. Hover any dot for the paper's title; click for the full abstract and authors.
        <div style={{ ...TEXT.caption, marginTop: 10, color: NEUTRAL.darkGray, fontStyle: 'italic' }}>
          Heads up: to keep this responsive in the browser we use a deliberately simple algorithm (TF-IDF on words, no semantic embeddings). It catches the broad strokes but won't always understand synonyms or paraphrase. Sorry.
        </div>
      </InstructionCard>

      <div style={{ marginBottom: 12 }}>
        <textarea
          value={query}
          onChange={handleQueryChange}
          placeholder="Paste an abstract here — ideally your own…"
          rows={5}
          style={{
            width: '100%',
            padding: '14px 16px',
            fontSize: '1rem',
            fontFamily: 'inherit',
            fontWeight: 400,
            color: NEUTRAL.black,
            border: `1px solid ${NEUTRAL.intLightGray}`,
            borderRadius: BORDER_RADIUS,
            outline: 'none',
            boxSizing: 'border-box',
            backgroundColor: NEUTRAL.white,
            boxShadow: '0 1px 2px rgba(0,0,0,0.04)',
            resize: 'vertical',
            lineHeight: 1.45
          }}
        />
      </div>

      <div style={{ ...TEXT.title, fontSize: '1.15rem', marginBottom: 8 }}>
        Map of popular neuroscience papers
      </div>

      <ScatterPlot
        corpus={corpus}
        scales={scales}
        width={width}
        height={height}
        flagPos={liveProjection?.coords || null}
        nearestIndices={debouncedReadout ? debouncedReadout.top5.map(t => t.index) : []}
        flagPosForLines={debouncedReadout?.flagPos || liveProjection?.coords || null}
        hovered={hovered}
        setHovered={setHovered}
        selected={selected}
        setSelected={setSelected}
      />

      <div style={{ ...TEXT.caption, marginTop: 12, color: NEUTRAL.midGray, fontStyle: 'italic' }}>
        Gold dots are the papers most similar to your text in the original (high-dimensional) space — the true nearest neighbors. Their lines to the flag are often long: UMAP preserves local neighborhoods but not global distances, and you cannot project high-D data into 2-D without losing information. Worth remembering whenever you read a UMAP or t-SNE map.
      </div>
      </div>
    </PageContainer>
  );
}

// ---------------------------------------------------------------------------
// Scatter rendering
// ---------------------------------------------------------------------------
function ScatterPlot({ corpus, scales, width, height, flagPos, nearestIndices, flagPosForLines, hovered, setHovered, selected, setSelected }) {
  const { sx, sy } = scales;
  const nearestSet = useMemo(() => new Set(nearestIndices), [nearestIndices]);

  return (
    <div style={{ position: 'relative', width, height }}>
      <svg
        width={width}
        height={height}
        onClick={(e) => { if (e.target.tagName === 'svg') setSelected(null); }}
        style={{
          backgroundColor: NEUTRAL.white,
          borderRadius: BORDER_RADIUS,
          border: `1px solid ${NEUTRAL.intLightGray}`,
          display: 'block',
          fontFamily: '"JetBrains Mono", monospace'
        }}
      >
        {corpus.paperCoords.map(([x, y], i) => {
          const cx = sx(x), cy = sy(y);
          const cid = corpus.paperClusterIds[i];
          const isNearest = nearestSet.has(i);
          const isHovered = hovered === i;
          const isSelected = selected === i;
          return (
            <circle
              key={i}
              cx={cx}
              cy={cy}
              r={isSelected ? 6 : isNearest ? 5 : isHovered ? 5 : 2.5}
              fill={isNearest ? PRIMARY.yellow : clusterColor(cid)}
              stroke={isSelected ? NEUTRAL.black : isNearest ? '#B88800' : 'none'}
              strokeWidth={isSelected ? 2 : isNearest ? 1 : 0}
              opacity={isNearest || isSelected ? 1 : 0.55}
              onMouseEnter={() => setHovered(i)}
              onMouseLeave={() => setHovered(h => (h === i ? null : h))}
              onClick={() => setSelected(s => (s === i ? null : i))}
              style={{ cursor: 'pointer', transition: 'r 0.15s ease' }}
            />
          );
        })}

        {flagPosForLines && nearestIndices.map(idx => {
          const [px, py] = corpus.paperCoords[idx];
          return (
            <line
              key={`line-${idx}`}
              x1={sx(flagPosForLines[0])}
              y1={sy(flagPosForLines[1])}
              x2={sx(px)}
              y2={sy(py)}
              stroke={PRIMARY.purple}
              strokeWidth={1}
              opacity={0.35}
              pointerEvents="none"
            />
          );
        })}

        {flagPos && (
          <g style={{ pointerEvents: 'none' }}
             transform={`translate(${sx(flagPos[0])}, ${sy(flagPos[1])})`}>
            <circle r={6} fill={PRIMARY.purple} opacity={0.16}>
              <animate attributeName="r" values="5;9;5" dur="1.8s" repeatCount="indefinite" />
              <animate attributeName="opacity" values="0.28;0.04;0.28" dur="1.8s" repeatCount="indefinite" />
            </circle>
            <circle r={2.5} fill={PRIMARY.purple} stroke={NEUTRAL.white} strokeWidth={1} />
            <line x1={0} y1={-1} x2={0} y2={-22} stroke={PRIMARY.purple} strokeWidth={1.6} strokeLinecap="round" />
            <path d="M 0 -22 L 14 -19 L 0 -15 Z" fill={PRIMARY.purple} stroke={NEUTRAL.white} strokeWidth={0.6} strokeLinejoin="round" />
            <text
              x={0}
              y={16}
              textAnchor="middle"
              style={{
                fontSize: '0.78rem',
                fontWeight: 600,
                fill: PRIMARY.purple,
                stroke: NEUTRAL.white,
                strokeWidth: 3,
                paintOrder: 'stroke',
                userSelect: 'none'
              }}
            >
              your paper
            </text>
          </g>
        )}

        <g transform={`translate(${width - 184}, 14)`} style={{ pointerEvents: 'none' }}>
          <rect
            x={0}
            y={0}
            width={170}
            height={30}
            rx={6}
            fill={NEUTRAL.white}
            opacity={0.92}
            stroke={NEUTRAL.intLightGray}
            strokeWidth={1}
          />
          <circle cx={16} cy={15} r={5} fill={PRIMARY.yellow} stroke="#B88800" strokeWidth={1} />
          <text
            x={30}
            y={19}
            style={{
              fontSize: '0.85rem',
              fill: NEUTRAL.darkGray,
              fontFamily: 'inherit'
            }}
          >
            5 most similar papers
          </text>
        </g>
      </svg>

      {hovered !== null && hovered !== selected && (
        <HoverTitle
          title={corpus.paperMeta[hovered].title}
          x={sx(corpus.paperCoords[hovered][0])}
          y={sy(corpus.paperCoords[hovered][1])}
          containerWidth={width}
        />
      )}

      {selected !== null && (
        <PaperPopup
          paper={corpus.paperMeta[selected]}
          x={sx(corpus.paperCoords[selected][0])}
          y={sy(corpus.paperCoords[selected][1])}
          containerWidth={width}
          containerHeight={height}
          onClose={() => setSelected(null)}
        />
      )}
    </div>
  );
}

function HoverTitle({ title, x, y, containerWidth }) {
  const maxW = 260;
  const gap = 28;
  const left = Math.min(Math.max(maxW / 2 + 8, x), containerWidth - maxW / 2 - 8);
  return (
    <div style={{
      position: 'absolute',
      left,
      top: y - gap,
      transform: 'translate(-50%, -100%)',
      pointerEvents: 'none',
      zIndex: 5
    }}>
      <div style={{
        maxWidth: maxW,
        backgroundColor: NEUTRAL.black,
        color: NEUTRAL.white,
        borderRadius: 4,
        padding: '5px 9px',
        fontSize: '0.76rem',
        lineHeight: 1.3,
        boxShadow: '0 2px 6px rgba(0,0,0,0.18)',
        textAlign: 'center'
      }}>
        {title}
      </div>
    </div>
  );
}

function PaperPopup({ paper, x, y, containerWidth, containerHeight, onClose }) {
  const w = 440;
  const estLines = Math.ceil((paper.abstract?.length || 0) / 64) + 2;
  const estHeight = Math.min(540, 110 + estLines * 20);
  const flipX = x + 18 + w > containerWidth;
  const flipY = y + 18 + estHeight > containerHeight;
  const left = flipX ? Math.max(8, x - w - 14) : x + 14;
  const top  = flipY ? Math.max(-30, y - estHeight) : y + 14;
  return (
    <div
      onClick={(e) => e.stopPropagation()}
      style={{
        position: 'absolute',
        left, top,
        width: w,
        maxHeight: 540,
        overflowY: 'auto',
        backgroundColor: NEUTRAL.white,
        border: `1px solid ${NEUTRAL.intLightGray}`,
        borderRadius: BORDER_RADIUS,
        padding: 16,
        boxShadow: '0 6px 20px rgba(0,0,0,0.16)',
        zIndex: 8
      }}
    >
      <button
        onClick={onClose}
        aria-label="Close"
        style={{
          position: 'absolute',
          top: 6, right: 6,
          width: 28, height: 28,
          borderRadius: '50%',
          border: 'none',
          background: 'transparent',
          cursor: 'pointer',
          color: NEUTRAL.darkGray,
          fontSize: '1.2rem',
          lineHeight: 1
        }}
      >
        ×
      </button>
      <div style={{ ...TEXT.heading, fontSize: '1rem', marginBottom: 6, paddingRight: 28 }}>
        {paper.url ? (
          <a
            href={paper.url}
            target="_blank"
            rel="noopener noreferrer"
            style={{ color: NEUTRAL.black, textDecoration: 'none', borderBottom: `1px solid ${NEUTRAL.intLightGray}` }}
            onMouseEnter={e => { e.currentTarget.style.color = PRIMARY.purple; e.currentTarget.style.borderBottomColor = PRIMARY.purple; }}
            onMouseLeave={e => { e.currentTarget.style.color = NEUTRAL.black; e.currentTarget.style.borderBottomColor = NEUTRAL.intLightGray; }}
          >
            {paper.title}
          </a>
        ) : paper.title}
      </div>
      <div style={{ ...TEXT.caption, marginBottom: 10 }}>
        {paper.year}{paper.authors?.length ? ` · ${paper.authors.slice(0, 4).join(', ')}${paper.authors.length > 4 ? ' et al.' : ''}` : ''}{typeof paper.citationCount === 'number' ? ` · ${paper.citationCount.toLocaleString()} cites` : ''}
      </div>
      <div style={{ ...TEXT.body, fontSize: '0.92rem', color: NEUTRAL.darkGray, lineHeight: 1.55 }}>
        {paper.abstract}
      </div>
      {paper.url && (
        <div style={{ marginTop: 10 }}>
          <a
            href={paper.url}
            target="_blank"
            rel="noopener noreferrer"
            style={{
              fontSize: '0.85rem',
              color: PRIMARY.purple,
              textDecoration: 'none',
              fontWeight: 600
            }}
          >
            Read paper →
          </a>
        </div>
      )}
    </div>
  );
}
