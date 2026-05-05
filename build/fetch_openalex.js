import { writeFileSync, existsSync, mkdirSync } from 'node:fs';
import { setTimeout as sleep } from 'node:timers/promises';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';

const MAILTO = 'koerding@gmail.com';
const FIELD_ID = 'fields/28';
const PER_PAGE = 200;
const TARGET = 2000;
const MAX_FETCH = 2500;
const RATE_LIMIT_MS = 1000;
const MIN_ABSTRACT_CHARS = 80;

const __dirname = dirname(fileURLToPath(import.meta.url));
const CACHE_DIR = resolve(__dirname, '..', 'cache');
const OUT_PATH = resolve(CACHE_DIR, 'papers_raw.json');

function reconstructAbstract(invertedIndex) {
  if (!invertedIndex || typeof invertedIndex !== 'object') return null;
  const positions = [];
  for (const [word, idxs] of Object.entries(invertedIndex)) {
    if (!Array.isArray(idxs)) continue;
    for (const i of idxs) positions.push([i, word]);
  }
  if (positions.length === 0) return null;
  positions.sort((a, b) => a[0] - b[0]);
  return positions.map(([, w]) => w).join(' ');
}

async function fetchPage(cursor) {
  const url = new URL('https://api.openalex.org/works');
  url.searchParams.set('filter', `primary_topic.field.id:${FIELD_ID},has_abstract:true,type:article,language:en`);
  url.searchParams.set('sort', 'cited_by_count:desc');
  url.searchParams.set('per_page', String(PER_PAGE));
  url.searchParams.set('cursor', cursor);
  url.searchParams.set('mailto', MAILTO);
  url.searchParams.set('select', 'id,title,publication_year,cited_by_count,authorships,abstract_inverted_index,primary_topic');

  const res = await fetch(url, {
    headers: {
      'User-Agent': `whitespace-map/0.1 (mailto:${MAILTO})`,
      'Accept': 'application/json'
    }
  });
  if (!res.ok) {
    const body = await res.text().catch(() => '');
    throw new Error(`OpenAlex ${res.status} ${res.statusText}: ${body.slice(0, 200)}`);
  }
  return res.json();
}

async function main() {
  if (existsSync(OUT_PATH) && !process.argv.includes('--force')) {
    console.log(`papers_raw.json already exists. Delete it or pass --force to refetch.`);
    return;
  }
  if (!existsSync(CACHE_DIR)) mkdirSync(CACHE_DIR, { recursive: true });

  const papers = [];
  const seen = new Set();
  let cursor = '*';
  let totalSeen = 0;
  let pageNum = 0;

  while (papers.length < TARGET && totalSeen < MAX_FETCH && cursor) {
    pageNum++;
    const t0 = Date.now();
    console.log(`[page ${pageNum}] kept=${papers.length}/${TARGET}, scanned=${totalSeen}, cursor=${cursor.slice(0, 12)}…`);

    const data = await fetchPage(cursor);
    const results = data.results || [];

    for (const w of results) {
      totalSeen++;
      if (papers.length >= TARGET) break;
      if (!w.id || seen.has(w.id)) continue;

      const abstract = reconstructAbstract(w.abstract_inverted_index);
      if (!abstract || abstract.length < MIN_ABSTRACT_CHARS) continue;

      const title = w.title;
      if (!title || title.length < 4) continue;

      seen.add(w.id);
      papers.push({
        id: w.id,
        title,
        abstract,
        year: w.publication_year ?? null,
        authors: (w.authorships || []).slice(0, 5).map(a => a.author?.display_name).filter(Boolean),
        citationCount: w.cited_by_count ?? 0,
        primaryTopic: w.primary_topic?.display_name ?? null,
        primarySubfield: w.primary_topic?.subfield?.display_name ?? null
      });
    }

    cursor = data.meta?.next_cursor || null;
    if (results.length === 0) break;

    if (papers.length < TARGET && cursor && totalSeen < MAX_FETCH) {
      const elapsed = Date.now() - t0;
      const wait = Math.max(RATE_LIMIT_MS - elapsed, 0);
      if (wait > 0) await sleep(wait);
    }
  }

  writeFileSync(
    OUT_PATH,
    JSON.stringify({
      fetchedAt: new Date().toISOString(),
      filterField: FIELD_ID,
      target: TARGET,
      kept: papers.length,
      scanned: totalSeen,
      papers
    }, null, 2)
  );
  console.log(`\nSaved ${papers.length} papers (scanned ${totalSeen}) → ${OUT_PATH}`);
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
