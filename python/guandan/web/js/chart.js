// Minimal canvas line chart with crosshair tooltip. Series colors validated for the felt surface.
import { T as _T } from './i18n.js';

export const SERIES_COLORS = ['#B08A30', '#2A9FD0', '#D65A50'];

export function fmtFrames(x) {
  if (x >= 1e9) return (x / 1e9).toFixed(2) + 'B';
  if (x >= 1e6) return (x / 1e6).toFixed(x >= 1e8 ? 0 : 1) + 'M';
  if (x >= 1e3) return (x / 1e3).toFixed(0) + 'K';
  return String(Math.round(x));
}

function niceTicks(min, max, n = 4) {
  if (max - min < 1e-9) { max = min + 1; }
  const span = max - min;
  const step0 = span / n;
  const mag = Math.pow(10, Math.floor(Math.log10(step0)));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => span / s <= n + 0.5) || mag * 10;
  const lo = Math.floor(min / step) * step;
  const hi = Math.ceil(max / step) * step;
  const out = [];
  for (let v = lo; v <= hi + step * 0.5; v += step) out.push(+v.toFixed(10));
  return out;
}

export class LineChart {
  constructor(canvas, tip, opts = {}) {
    this.c = canvas;
    this.tip = tip;
    this.opts = opts;
    this.series = [];
    this.hoverX = null;
    canvas.addEventListener('pointermove', (e) => { const r = canvas.getBoundingClientRect(); this.hoverX = e.clientX - r.left; this.draw(); });
    canvas.addEventListener('pointerleave', () => { this.hoverX = null; this.tip.style.display = 'none'; this.draw(); });
  }
  set(series) { this.series = series; this.draw(); }
  draw() {
    const c = this.c;
    const dpr = window.devicePixelRatio || 1;
    const W = c.clientWidth;
    const H = c.clientHeight;
    if (!W || !H) return;
    if (c.width !== Math.round(W * dpr) || c.height !== Math.round(H * dpr)) { c.width = Math.round(W * dpr); c.height = Math.round(H * dpr); }
    const g = c.getContext('2d');
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, W, H);
    const pts = this.series.flatMap((s) => s.points);
    if (!pts.length) return;
    const o = this.opts;
    let xmin = Math.min(...pts.map((p) => p.x));
    let xmax = Math.max(...pts.map((p) => p.x));
    if (xmax === xmin) { const d = Math.max(1, Math.abs(xmin) * 0.1); xmin -= d; xmax += d; }
    let ymin = o.yMin ?? Math.min(...pts.map((p) => p.y));
    let ymax = o.yMax ?? Math.max(...pts.map((p) => p.y));
    if (o.yMin == null || o.yMax == null) { const pad = (ymax - ymin) * 0.08 || 0.1; if (o.yMin == null) ymin -= pad; if (o.yMax == null) ymax += pad; }
    const yt = niceTicks(ymin, ymax, 4);
    if (o.yMin == null) ymin = yt[0];
    if (o.yMax == null) ymax = yt[yt.length - 1];
    const L = 44, R = 12, T = 8, B = 22;
    const X = (x) => L + (x - xmin) / (xmax - xmin) * (W - L - R);
    const Y = (y) => T + (1 - (y - ymin) / (ymax - ymin)) * (H - T - B);
    g.font = '11px system-ui, sans-serif';
    g.fillStyle = 'rgba(179,199,188,0.85)';
    g.strokeStyle = 'rgba(233,207,146,0.10)';
    g.lineWidth = 1;
    g.textAlign = 'right';
    g.textBaseline = 'middle';
    for (const v of yt) {
      if (v < ymin - 1e-9 || v > ymax + 1e-9) continue;
      const y = Math.round(Y(v)) + 0.5;
      g.beginPath(); g.moveTo(L, y); g.lineTo(W - R, y); g.stroke();
      g.fillText(o.yFormat ? o.yFormat(v) : String(v), L - 6, y);
    }
    if (o.refLine != null && o.refLine >= ymin && o.refLine <= ymax) {
      g.save(); g.setLineDash([4, 4]); g.strokeStyle = 'rgba(234,240,234,0.35)';
      const y = Math.round(Y(o.refLine)) + 0.5; g.beginPath(); g.moveTo(L, y); g.lineTo(W - R, y); g.stroke(); g.restore();
    }
    g.textAlign = 'center';
    g.textBaseline = 'top';
    const xt = [...new Set(niceTicks(xmin, xmax, Math.max(2, Math.floor(W / 110))).map((v) => fmtFrames(v)))].length < 2 ? [] : niceTicks(xmin, xmax, Math.max(2, Math.floor(W / 110)));
    const seen = new Set();
    for (const v of xt) {
      const lab = o.xFormat ? o.xFormat(v) : fmtFrames(v);
      if (seen.has(lab)) continue;
      seen.add(lab);
      if (v < xmin || v > xmax) continue;
      g.fillText(lab, X(v), H - B + 6);
    }
    this.series.forEach((s) => {
      if (!s.points.length) return;
      g.strokeStyle = s.color;
      g.lineWidth = 2;
      g.lineJoin = 'round';
      g.beginPath();
      s.points.forEach((p, i) => (i ? g.lineTo(X(p.x), Y(p.y)) : g.moveTo(X(p.x), Y(p.y))));
      g.stroke();
      const last = s.points[s.points.length - 1];
      g.fillStyle = s.color;
      g.strokeStyle = '#102A22';
      g.lineWidth = 2;
      g.beginPath(); g.arc(X(last.x), Y(last.y), 4, 0, Math.PI * 2); g.fill(); g.stroke();
    });
    if (this.hoverX != null) {
      const xv = xmin + (this.hoverX - L) / (W - L - R) * (xmax - xmin);
      let best = null;
      for (const p of pts) if (!best || Math.abs(p.x - xv) < Math.abs(best.x - xv)) best = p;
      if (!best) return;
      const x = X(best.x);
      g.strokeStyle = 'rgba(234,240,234,0.35)'; g.lineWidth = 1;
      g.beginPath(); g.moveTo(x + 0.5, T); g.lineTo(x + 0.5, H - B); g.stroke();
      const rows = [];
      this.series.forEach((s) => {
        let q = null;
        for (const p of s.points) if (!q || Math.abs(p.x - best.x) < Math.abs(q.x - best.x)) q = p;
        if (q && Math.abs(q.x - best.x) <= (xmax - xmin) * 0.02 + 1e-9) {
          rows.push(`<div><i style="display:inline-block;width:10px;height:3px;background:${s.color};margin-right:6px;vertical-align:middle"></i>${s.name}：<b>${o.yFormat ? o.yFormat(q.y) : q.y.toFixed(3)}</b></div>`);
          g.fillStyle = s.color; g.strokeStyle = '#102A22'; g.lineWidth = 2;
          g.beginPath(); g.arc(X(q.x), Y(q.y), 4.5, 0, Math.PI * 2); g.fill(); g.stroke();
        }
      });
      this.tip.innerHTML = `<div style="color:#b3c7bc">${_T('{label}：{value}', { label: o.xLabel || _T('样本'), value: o.xFormat ? o.xFormat(best.x) : fmtFrames(best.x) })}</div>${rows.join('')}`;
      this.tip.style.display = 'block';
      const tw = this.tip.offsetWidth;
      this.tip.style.left = `${Math.min(Math.max(0, x + 14), W - tw)}px`;
      this.tip.style.top = '34px';
    }
  }
}
