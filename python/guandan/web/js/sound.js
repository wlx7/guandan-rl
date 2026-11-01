// Tiny synthesized sound effects (no audio files needed).
let ctx = null;
let enabled = (() => { try { return localStorage.getItem('gd_sound') !== '0'; } catch { return true; } })();

function ac() {
  if (!ctx) {
    const C = window.AudioContext || window.webkitAudioContext;
    if (!C) return null;
    ctx = new C();
  }
  if (ctx.state === 'suspended') ctx.resume();
  return ctx;
}

function noise(dur, freq, gain = 0.25, q = 1.2) {
  const a = ac();
  if (!a) return;
  const len = Math.floor(a.sampleRate * dur);
  const buf = a.createBuffer(1, len, a.sampleRate);
  const d = buf.getChannelData(0);
  for (let i = 0; i < len; i++) d[i] = (Math.random() * 2 - 1) * Math.pow(1 - i / len, 3);
  const src = a.createBufferSource();
  src.buffer = buf;
  const f = a.createBiquadFilter();
  f.type = 'bandpass';
  f.frequency.value = freq;
  f.Q.value = q;
  const g = a.createGain();
  g.gain.value = gain;
  src.connect(f).connect(g).connect(a.destination);
  src.start();
}

function tone(freq, dur, type = 'sine', gain = 0.12, when = 0) {
  const a = ac();
  if (!a) return;
  const o = a.createOscillator();
  const g = a.createGain();
  o.type = type;
  o.frequency.value = freq;
  const t0 = a.currentTime + when;
  g.gain.setValueAtTime(0.0001, t0);
  g.gain.exponentialRampToValueAtTime(gain, t0 + 0.01);
  g.gain.exponentialRampToValueAtTime(0.0001, t0 + dur);
  o.connect(g).connect(a.destination);
  o.start(t0);
  o.stop(t0 + dur + 0.05);
}

export const sound = {
  get enabled() { return enabled; },
  toggle() {
    enabled = !enabled;
    try { localStorage.setItem('gd_sound', enabled ? '1' : '0'); } catch {}
    if (enabled) ac();
    return enabled;
  },
  unlock() { if (enabled) ac(); },
  play() { if (enabled) noise(0.09, 2400, 0.35, 0.8); },
  select() { if (enabled) noise(0.03, 3800, 0.12, 1.5); },
  pass() { if (enabled) tone(330, 0.18, 'triangle', 0.06); },
  turn() { if (enabled) { tone(784, 0.16, 'sine', 0.09); tone(1046, 0.22, 'sine', 0.08, 0.12); } },
  tick() { if (enabled) tone(1500, 0.05, 'square', 0.03); },
  bomb() {
    if (!enabled) return;
    const a = ac();
    if (!a) return;
    noise(0.7, 120, 0.9, 0.6);
    const o = a.createOscillator();
    const g = a.createGain();
    o.type = 'sine';
    o.frequency.setValueAtTime(160, a.currentTime);
    o.frequency.exponentialRampToValueAtTime(38, a.currentTime + 0.6);
    g.gain.setValueAtTime(0.5, a.currentTime);
    g.gain.exponentialRampToValueAtTime(0.0001, a.currentTime + 0.7);
    o.connect(g).connect(a.destination);
    o.start();
    o.stop(a.currentTime + 0.75);
  },
  deal() { if (enabled) for (let i = 0; i < 6; i++) setTimeout(() => noise(0.05, 3000, 0.12), i * 70); },
  win() { if (enabled) [523, 659, 784, 1046].forEach((f, i) => tone(f, 0.3, 'triangle', 0.09, i * 0.12)); },
  lose() { if (enabled) [392, 330, 262].forEach((f, i) => tone(f, 0.35, 'triangle', 0.07, i * 0.15)); },
};
