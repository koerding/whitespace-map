const ENGLISH_STOPWORDS = new Set([
  'the','and','for','are','but','not','you','all','any','can','had','her','was','one','our','out','his','has','had','how','its','may','new','now','old','see','two','way','who','boy','did','use','her','many','then','them','these','some','what','were','when','your','said','each','which','their','about','would','there','could','other','than','more','very','also','from','this','that','have','with','will','been','they','through','during','before','after','above','below','between','under','over','into','onto','upon','only','such','same','those','where','while','here','because','being','both','once','itself','themselves','ourselves','yourself','should','could','would','might','must','does','done','doing','having','using','used','show','shown','showed','study','studies','studied','results','result','here','propose','proposed','paper','papers','data','analysis','method','methods','findings','finding','however','therefore','thus','further','among','within','across','toward','towards','via','per','due','given','overall','various','several','many','much','most','least','few','either','neither','rather','quite','still','already','though','although','since','until','unless','versus','despite','besides','indeed','perhaps','likely','possibly','generally','typically','specifically','particularly','approximately','roughly','nearly','almost','found','find','finds','present','presented','presents','reported','reports','report','observed','observe','observes','demonstrated','demonstrate','demonstrates','suggest','suggests','suggested','show','shows','indicate','indicates','indicated'
]);

const TOKEN_RE = /[a-z]+/g;

export function tokenize(text) {
  if (!text) return [];
  const lower = text.toLowerCase();
  const out = [];
  for (const m of lower.matchAll(TOKEN_RE)) {
    const t = m[0];
    if (t.length < 3) continue;
    if (ENGLISH_STOPWORDS.has(t)) continue;
    out.push(t);
  }
  return out;
}

export function termCounts(tokens) {
  const counts = new Map();
  for (const t of tokens) counts.set(t, (counts.get(t) || 0) + 1);
  return counts;
}
