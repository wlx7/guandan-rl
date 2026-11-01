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

export const seatStore = {
  get() { try { return JSON.parse(localStorage.getItem('gd_seat') || 'null'); } catch { return null; } },
  set(v) { try { localStorage.setItem('gd_seat', JSON.stringify(v)); } catch {} },
  clear() { try { localStorage.removeItem('gd_seat'); } catch {} },
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

