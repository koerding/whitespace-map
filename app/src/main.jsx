/**
 * Local test harness — NOT part of the deliverable.
 * Wraps WhitespaceMap.jsx in a minimal multi-screen activity scaffold so
 * we can verify the component works in context before handing it off.
 */
import React, { useState } from 'react';
import { createRoot } from 'react-dom/client';
import WhitespaceMap from './WhitespaceMap.jsx';

const PRIMARY = { purple: '#6F00FF', blue: '#0082FF' };
const NEUTRAL = {
  black: '#202020', darkGray: '#333132', midGray: '#999999',
  intLightGray: '#E0E0E0', lightGray: '#F3F3F3', white: '#FFFFFF'
};
const TEXT = {
  heading: { fontSize: '1.4rem', fontWeight: 700, color: NEUTRAL.black },
  body:    { fontSize: '1.0rem', fontWeight: 300, color: NEUTRAL.black, lineHeight: 1.5 }
};

const SCREENS = [
  { kind: 'intro',     heading: 'What is whitespace?' },
  { kind: 'map',       heading: 'Locate your work' },
  { kind: 'reflection',heading: 'Reflect' },
  { kind: 'wrap',      heading: 'Wrap up' }
];

function Progress({ current, total }) {
  return (
    <div style={{ marginBottom: 16 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 6, fontSize: '0.8rem', color: NEUTRAL.darkGray }}>
        <span>{SCREENS[current].heading}</span>
        <span>Screen {current + 1} of {total}</span>
      </div>
      <div style={{ display: 'flex', gap: 4 }}>
        {Array.from({ length: total }).map((_, i) => (
          <div key={i} style={{
            flex: 1, height: 8, borderRadius: 4,
            background: i <= current ? PRIMARY.purple : NEUTRAL.intLightGray,
            transition: 'background 400ms ease'
          }} />
        ))}
      </div>
    </div>
  );
}

function NavBar({ canPrev, canNext, onPrev, onNext, isLast }) {
  const baseBtn = {
    padding: '10px 20px',
    fontSize: '0.95rem',
    fontWeight: 600,
    borderRadius: 8,
    cursor: 'pointer',
    border: '1px solid transparent',
    transition: 'background 0.15s ease, color 0.15s ease',
    fontFamily: 'inherit',
    textTransform: 'uppercase',
    letterSpacing: '0.04em'
  };
  return (
    <div style={{
      position: 'fixed', bottom: 0, left: 0, right: 0,
      background: NEUTRAL.white,
      borderTop: `1px solid ${NEUTRAL.intLightGray}`,
      padding: '12px 16px',
      display: 'flex', justifyContent: 'space-between',
      boxShadow: '0 -2px 8px rgba(0,0,0,0.04)',
      zIndex: 10
    }}>
      <button
        onClick={onPrev}
        disabled={!canPrev}
        style={{
          ...baseBtn,
          background: NEUTRAL.lightGray,
          color: canPrev ? NEUTRAL.darkGray : NEUTRAL.midGray,
          opacity: canPrev ? 1 : 0.4
        }}
      >Previous</button>
      <button
        onClick={onNext}
        disabled={!canNext}
        style={{
          ...baseBtn,
          background: canNext ? NEUTRAL.black : NEUTRAL.lightGray,
          color: canNext ? NEUTRAL.white : NEUTRAL.midGray,
          opacity: canNext ? 1 : 0.4
        }}
      >{isLast ? 'Complete activity' : 'Next'}</button>
    </div>
  );
}

function Intro() {
  return (
    <div style={{ maxWidth: 900, margin: '0 auto', padding: 24 }}>
      <div style={{ ...TEXT.heading, marginBottom: 12 }}>What is whitespace?</div>
      <div style={TEXT.body}>
        Many research questions have already been studied — or at least very similar questions have been asked. But not all of them. The questions that haven't are <em>whitespace</em> — places in the high-dimensional space of possible questions where the literature is sparse.
      </div>
      <div style={{ ...TEXT.body, marginTop: 12 }}>
        Whitespace can mean: a real opportunity, an unanswered question worth pursuing — or a dead-end that no one bothered with for a reason. The map you're about to use makes occupancy visible. The judgment about whether a given empty region is opportunity or dead-end is yours.
      </div>
    </div>
  );
}

function Reflection({ value, onChange }) {
  return (
    <div style={{ maxWidth: 900, margin: '0 auto', padding: 24 }}>
      <div style={{ ...TEXT.heading, marginBottom: 12 }}>Reflect</div>
      <div style={TEXT.body}>
        You just saw that the gold dots — the papers actually closest to yours — often sit far from your flag on the map. UMAP and t-SNE don't really preserve the geometry of an intellectual landscape; the whitespace you see in a 2-D projection is partly an artifact of the projection itself.
      </div>
      <div style={{ ...TEXT.body, marginTop: 10 }}>
        Now describe the intellectual landscape <em>your own</em> work lives in. Where does it sit? Which directions does it border? Which gaps around it are real whitespace, and which are just the kind of distortion the map showed you?
      </div>
      <textarea
        value={value}
        onChange={e => onChange(e.target.value)}
        placeholder="Write a few sentences…"
        style={{
          width: '100%',
          minHeight: 140,
          marginTop: 12,
          padding: 12,
          fontSize: '1rem',
          fontFamily: 'inherit',
          color: NEUTRAL.black,
          border: `1px solid ${NEUTRAL.intLightGray}`,
          borderRadius: 6,
          boxSizing: 'border-box',
          resize: 'vertical'
        }}
      />
    </div>
  );
}

function Wrap() {
  return (
    <div style={{ maxWidth: 900, margin: '0 auto', padding: 24, textAlign: 'center' }}>
      <div style={{ ...TEXT.heading, marginBottom: 12 }}>Nice work.</div>
      <div style={TEXT.body}>
        You used the map to feel the shape of an existing literature and to locate one of your own questions inside it. The next time you're framing a project, do this exercise on real abstracts in your subfield.
      </div>
      <div style={{ ...TEXT.body, marginTop: 18, fontSize: '0.95rem', color: NEUTRAL.darkGray }}>
        To see the code that made this, please see our <a href="https://github.com/koerding/whitespace-map" target="_blank" rel="noopener noreferrer" style={{ color: PRIMARY.purple, fontWeight: 600 }}>repo</a>.
      </div>
    </div>
  );
}

function App() {
  const [screen, setScreen] = useState(0);
  const [mapInteracted, setMapInteracted] = useState(false);
  const [reflection, setReflection] = useState('');

  const canProceedByScreen = [
    true,
    mapInteracted,
    reflection.trim().length > 0,
    true
  ];
  const canNext = canProceedByScreen[screen];
  const isLast = screen === SCREENS.length - 1;

  return (
    <div style={{ paddingBottom: 80 }}>
      <div style={{ maxWidth: 900, margin: '0 auto', padding: '24px 16px 0' }}>
        <Progress current={screen} total={SCREENS.length} />
      </div>

      {SCREENS[screen].kind === 'intro' && <Intro />}
      {SCREENS[screen].kind === 'map' && (
        <WhitespaceMap
          corpusUrl="/corpus.json"
          onInteract={() => setMapInteracted(true)}
        />
      )}
      {SCREENS[screen].kind === 'reflection' && <Reflection value={reflection} onChange={setReflection} />}
      {SCREENS[screen].kind === 'wrap' && <Wrap />}

      <NavBar
        canPrev={screen > 0}
        canNext={canNext}
        onPrev={() => setScreen(s => Math.max(0, s - 1))}
        onNext={() => isLast ? alert('Activity complete') : setScreen(s => Math.min(SCREENS.length - 1, s + 1))}
        isLast={isLast}
      />
    </div>
  );
}

createRoot(document.getElementById('root')).render(<App />);
