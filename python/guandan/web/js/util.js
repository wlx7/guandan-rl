// Shared helpers
export const api = {
  async get(url) {
    const r = await fetch(url, { cache: 'no-store' });
    if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
    return r.json();
  },
  async post(url, body = {}) {
    const r = await fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
    if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
    return r.json();
  },
};

export function toast(msg, kind = '', ms = 2600) {
  const layer = document.getElementById('toasts');
  const t = document.createElement('div');
  t.className = 'toast' + (kind ? ' ' + kind : '');
  if (msg instanceof Node) t.appendChild(msg); else t.textContent = msg;
  layer.appendChild(t);
  while (layer.children.length > 4) layer.firstChild.remove();
  setTimeout(() => { t.classList.add('out'); setTimeout(() => t.remove(), 320); }, ms);
}

export function h(tag, attrs = {}, ...kids) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === null || v === undefined || v === false) continue;
    if (k === 'class') el.className = v;
    else if (k === 'style' && typeof v === 'object') Object.assign(el.style, v);
    else if (k.startsWith('on')) el.addEventListener(k.slice(2), v);
    else if (k === 'html') el.innerHTML = v;
    else el.setAttribute(k, v === true ? '' : v);
  }
  for (const kid of kids.flat()) {
    if (kid === null || kid === undefined || kid === false) continue;
    el.appendChild(kid instanceof Node ? kid : document.createTextNode(String(kid)));
  }
  return el;
}

// this device's seats, one per table: {room id: {room, seat, token}}; "current" is the table
// it used last (old versions kept a single seat under gd_seat)
const SEATS = 'gd_seats';
function loadSeats() {
  try {
    const v = JSON.parse(localStorage.getItem(SEATS) || 'null');
    if (v && typeof v === 'object') return v;
  } catch {}
  try {
    const o = JSON.parse(localStorage.getItem('gd_seat') || 'null');
    if (o && o.room) return { [o.room]: o };
  } catch {}
  return {};
}
export const seatStore = {
  get(room) {
    const id = room || this.current();
    return (id && loadSeats()[id]) || null;
  },
  set(v) {
    const all = loadSeats();
    all[v.room] = v;
    try { localStorage.setItem(SEATS, JSON.stringify(all)); localStorage.setItem('gd_room', v.room); } catch {}
  },
  current() {
    try { return localStorage.getItem('gd_room') || ((JSON.parse(localStorage.getItem('gd_seat') || 'null') || {}).room) || null; } catch { return null; }
  },
  keep(ids) { // forget seats of tables that no longer exist
    const all = loadSeats();
    let changed = false;
    for (const k of Object.keys(all)) if (!ids.includes(k)) { delete all[k]; changed = true; }
    if (changed) try { localStorage.setItem(SEATS, JSON.stringify(all)); } catch {}
  },
};
export const nameStore = {
  get() { try { return localStorage.getItem('gd_name') || ''; } catch { return ''; } },
  set(v) { try { localStorage.setItem('gd_name', v); } catch {} },
};

export function modal(content, { onClose } = {}) {
  const back = h('div', { class: 'modal-back' });
  const box = h('div', { class: 'modal', role: 'dialog' });
  box.appendChild(content);
  back.appendChild(box);
  document.body.appendChild(back);
  const close = () => { back.remove(); onClose && onClose(); };
  back.addEventListener('click', (e) => { if (e.target === back && onClose) close(); });
  return { close, el: back };
}


// Model <select> grouped as 内置 / 模型库 / 训练, showing display names.
export function modelLabel(m) {
  return m.label || `${m.source}：${m.id.split('/').slice(-1)[0]}`;
}
export function fillModelSelect(sel, models, value) {
  const groups = new Map();
  for (const m of models) {
    const g = m.group || m.source || '';
    if (!groups.has(g)) groups.set(g, []);
    groups.get(g).push(m);
  }
  for (const [g, list] of groups) {
    const og = h('optgroup', { label: g });
    for (const m of list) og.appendChild(h('option', { value: m.id, selected: m.id === value }, modelLabel(m)));
    sel.appendChild(og);
  }
  return sel;
}
