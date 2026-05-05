import { tokenize, termCounts } from './tokenize.js';

export function fitTfidf(docs, { maxVocab = 3000, minDf = 3, maxDfRatio = 0.5 } = {}) {
  const N = docs.length;
  const df = new Map();
  const docCounts = new Array(N);

  for (let i = 0; i < N; i++) {
    const counts = termCounts(tokenize(docs[i]));
    docCounts[i] = counts;
    for (const term of counts.keys()) {
      df.set(term, (df.get(term) || 0) + 1);
    }
  }

  const maxDf = Math.floor(maxDfRatio * N);
  const candidates = [];
  for (const [term, freq] of df) {
    if (freq < minDf) continue;
    if (freq > maxDf) continue;
    candidates.push([term, freq]);
  }
  candidates.sort((a, b) => b[1] - a[1]);
  const kept = candidates.slice(0, maxVocab);

  const vocabulary = kept.map(([t]) => t);
  vocabulary.sort();
  const termIndex = new Map(vocabulary.map((t, i) => [t, i]));

  const idf = new Float64Array(vocabulary.length);
  for (let i = 0; i < vocabulary.length; i++) {
    const f = df.get(vocabulary[i]);
    idf[i] = Math.log((1 + N) / (1 + f)) + 1;
  }

  const vectors = docCounts.map(counts => sparseFromCounts(counts, termIndex, idf));

  return { vocabulary, idf: Array.from(idf), vectors, termIndex };
}

export function vectorizeQuery(text, vocabulary, idf) {
  const termIndex = new Map(vocabulary.map((t, i) => [t, i]));
  const counts = termCounts(tokenize(text));
  return sparseFromCounts(counts, termIndex, idf);
}

function sparseFromCounts(counts, termIndex, idf) {
  const indices = [];
  const values = [];
  for (const [term, c] of counts) {
    const idx = termIndex.get(term);
    if (idx === undefined) continue;
    const tf = 1 + Math.log(c);
    indices.push(idx);
    values.push(tf * idf[idx]);
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

export function denseFromSparse(sparse, dim) {
  const out = new Float64Array(dim);
  const { indices, values } = sparse;
  for (let i = 0; i < indices.length; i++) out[indices[i]] = values[i];
  return out;
}

export function cosineSparse(a, b) {
  let i = 0, j = 0, dot = 0;
  const ai = a.indices, av = a.values, bi = b.indices, bv = b.values;
  while (i < ai.length && j < bi.length) {
    if (ai[i] === bi[j]) { dot += av[i] * bv[j]; i++; j++; }
    else if (ai[i] < bi[j]) i++;
    else j++;
  }
  return dot;
}

export function cosineDistanceSparse(a, b) {
  return 1 - cosineSparse(a, b);
}
