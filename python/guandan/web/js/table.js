import { cardEl, columns, withIds, sortCards, miniRow, cardName, RANKS, rankOf } from './cards.js';
import { sound } from './sound.js';
import { api, toast, h, seatStore, modal } from './util.js';

const PLACE = ['头游', '二游', '三游', '末游'];
const BOMB_WORD = { 8: '炸弹', 9: '同花顺', 10: '天王炸' };
const DIR = ['bottom', 'right', 'top', 'left'];

export function mountTable(app) {
  const seat = seatStore.get() || { seat: -1, token: null };
  const S = {
    s: null, prev: null, ws: null, closed: false, retry: 0,
    sel: new Set(), desc: true, reveal: false, hints: null, hintsSeq: -1,
    held: null, heldTimer: null, offset: 0, lastTick: -1, raf: 0, busy: false,
    reviewKey: null, resultShown: null, modal: null, layout: null, lastTurnKey: null,
  };

  // ------------------------------------------------------------ skeleton
  const backBtn = h('button', { class: 'btn small icon', title: '离开牌桌', onclick: leave }, '离开');
  const soundBtn = h('button', { class: 'btn small icon', title: '音效', onclick: () => { sound.toggle(); soundBtn.textContent = sound.enabled ? '音效开' : '音效关'; } }, sound.enabled ? '音效开' : '音效关');
  const lvBox = h('div', { class: 'lvbox' });
  const meta = h('div', { class: 'meta' });
  const topbar = h('div', { class: 'topbar' }, backBtn, h('div', { class: 'grow' }, lvBox), meta, soundBtn);
  const board = h('div', { class: 'board' });
  const seatEls = DIR.map((d) => h('div', { class: `seat ${d}` }));
  const zoneEls = DIR.map((d) => h('div', { class: `zone ${d}` }));
  const centerMsg = h('div', { class: 'center-msg' });
  board.append(...zoneEls, ...seatEls, centerMsg);
  const actionbar = h('div', { class: 'actionbar' });
  const hand = h('div', { class: 'hand' });
  const handWrap = h('div', { class: 'hand-wrap' }, hand);
  const side = h('div', { class: 'side-panel' });
  const root = h('div', { class: 'gt' }, topbar, board, actionbar, handWrap, side);
  app.replaceChildren(root);

  // ------------------------------------------------------------ socket
  function connect() {
    if (S.closed) return;
    const proto = location.protocol === 'https:' ? 'wss' : 'ws';
    const q = seat.token ? `?token=${encodeURIComponent(seat.token)}` : '';
    const ws = new WebSocket(`${proto}://${location.host}/ws${q}`);
    S.ws = ws;
    ws.onopen = () => { S.retry = 0; };
    ws.onmessage = (e) => {
      let m;
      try { m = JSON.parse(e.data); } catch { return; }
      onMessage(m);
    };
    ws.onclose = () => {
      if (S.closed) return;
      S.retry++;
      if (S.retry === 2) toast('连接中断，正在重连…', 'err');
      setTimeout(connect, Math.min(4000, 500 * S.retry));
    };
  }
  function send(obj) {
    if (S.ws && S.ws.readyState === 1) S.ws.send(JSON.stringify(obj));
    else toast('未连接到主机', 'err');
  }

  function onMessage(m) {
    if (m.type === 'state') return onState(m);
    if (m.type === 'error') {
      S.busy = false;
      renderActions();
      toast(m.message, 'err');
      if (m.fatal) {
        S.closed = true;
        setTimeout(() => { location.hash = '#/'; }, 1500);
      }
      return;
    }
    if (m.type === 'choose') { S.busy = false; return chooseDialog(m.options); }
    if (m.type === 'hints') { S.hints = m.hints; S.hintsSeq = m.seq; return renderSide(); }
  }

  // ------------------------------------------------------------ helpers
  const you = () => (S.s ? S.s.you : -1);
  const base = () => (you() >= 0 ? you() : 0);
  const relOf = (p) => (p - base() + 4) % 4;
  const ourTeam = () => base() % 2;
  const nameOf = (p) => (S.s && S.s.seats[p] ? S.s.seats[p].name || `玩家${p + 1}` : '');
  const myTurn = () => S.s && you() >= 0 && S.s.current === you() && S.s.status === 'playing' && (S.s.phase === 0 || S.s.phase === 1);
  const myHand = () => (S.s ? (you() >= 0 ? S.s.hand || [] : S.s.hands['0'] || []) : []);

  // ------------------------------------------------------------ state
  function onState(s) {
    const prev = S.s;
    S.offset = s.server_time - Date.now();
    const isNew = !prev || s.seq !== prev.seq;
    const ev = s.event || {};
    let flyFrom = null;
    if (isNew && prev && ev.type === 'play' && ev.player === you()) flyFrom = captureHandRects(ev.cards);
    const newRound = !prev || prev.round_no !== s.round_no || prev.match_no !== s.match_no || (prev.phase >= 2 && s.phase < 2);
    S.prev = prev;
    S.s = s;
    if (newRound) {
      S.sel.clear();
      S.hints = null;
      S.held = null;
      if (S.modal) { S.modal.close(); S.modal = null; }
      S.resultShown = null;
    }
    if (isNew && prev && ev.type === 'play') S.sel.clear();
    if (S.hints && (s.current !== you() || s.seq !== S.hintsSeq)) S.hints = null;
    // keep the finished trick visible for a moment
    if (isNew && ev.trick_end && prev) {
      const keep = { ...(prev.trick || {}) };
      keep[String(ev.player)] = { pass_: true };
      S.held = keep;
      clearTimeout(S.heldTimer);
      S.heldTimer = setTimeout(() => { S.held = null; renderZones(); }, 1100);
      if (ev.jiefeng) toast(`接风：${nameOf(ev.leader)} 获得出牌权`);
    } else if (isNew && ev.type === 'play') {
      S.held = null;
    }
    S.busy = false;
    render();
    if (isNew && prev) animateEvent(ev, prev, flyFrom);
    if (isNew && !prev && s.status === 'playing' && ev.type === 'deal') animateDeal();
    const turnKey = `${s.seq}`;
    if (myTurn() && prev && !(prev.current === you() && prev.phase === s.phase && prev.status === s.status) && S.lastTurnKey !== turnKey) {
      S.lastTurnKey = turnKey;
      sound.turn();
    }
    if (s.review && s.review.seat === you()) {
      const k = JSON.stringify(s.review);
      if (k !== S.reviewKey) { S.reviewKey = k; showReview(s.review); }
    }
    if (s.phase >= 2 && s.result) {
      const key = `${s.match_no}-${s.round_no}-${s.phase}`;
      if (S.resultShown !== key) {
        S.resultShown = key;
        setTimeout(() => { if (S.s && S.s.phase >= 2) showResult(); }, isNew && prev ? 1300 : 100);
      } else if (S.modal && S.modal.update) S.modal.update();
    }
  }

  // ------------------------------------------------------------ render
  function render() {
    const s = S.s;
    if (!s) return;
    renderTop();
    if (s.status === 'waiting') return renderWaiting();
    renderSeats();
    renderZones();
    renderCenter();
    renderActions();
    renderHand();
    renderSide();
  }

  function renderTop() {
    const s = S.s;
    const ot = ourTeam();
    const lv = (t) => RANKS[s.team_level[t]];
    const prevLevel = S.prev ? S.prev.level : s.level;
    const plaque = h('div', { class: 'plaque' + (prevLevel !== s.level ? ' pulse' : '') },
      h('div', { class: 'big' }, '打 ', h('span', { class: 'n' }, RANKS[s.level])),
      h('div', { class: 'sub' }, `红桃${RANKS[s.level]} 逢人配`));
    if (s.length === 'quick') {
      lvBox.replaceChildren(plaque);
    } else {
      lvBox.replaceChildren(
        h('div', { class: 'teamlv ours' + (s.banker === ot ? ' banker' : '') }, h('span', {}, '我方'), h('b', {}, lv(ot))),
        plaque,
        h('div', { class: 'teamlv' + (s.banker === 1 - ot ? ' banker' : '') }, h('span', {}, '对方'), h('b', {}, lv(1 - ot))));
    }
    const modeName = s.mode === 'teach' ? '教学' : '实战';
    const line2 = s.length === 'quick'
      ? `积分 ${s.points[ot]} : ${s.points[1 - ot]}`
      : `第 ${s.match_no} 场  胜场 ${s.score[ot]} : ${s.score[1 - ot]}`;
    meta.replaceChildren(h('div', {}, h('b', {}, `${modeName}${s.length === 'quick' ? '快速' : '竞技'}`), `  第 ${s.round_no} 局`), h('div', {}, line2));
  }

  function renderWaiting() {
    const s = S.s;
    seatEls.forEach((e) => e.replaceChildren());
    zoneEls.forEach((e) => e.replaceChildren());
    actionbar.replaceChildren();
    hand.replaceChildren();
    hand.style.height = '0px';
    const list = h('div', { class: 'seat-list' });
    s.seats.forEach((st, i) => {
      list.appendChild(h('div', { class: 'seat-row' },
        h('span', { class: 'pos' }, ['南', '东', '北', '西'][i]),
        h('span', { class: 'nm' }, st.kind === 'ai' ? st.name : (st.connected ? st.name : '等待加入…')),
        h('span', { class: 'dot' + (st.connected ? ' on' : '') })));
    });
    const box = h('div', { class: 'panel', style: { width: 'min(640px, 94vw)', pointerEvents: 'auto', textAlign: 'left' } },
      h('h2', {}, '等待玩家加入'),
      h('div', { class: 'wait-grid' },
        h('div', {}, h('p', { class: 'muted', style: { marginTop: 0 } }, '让朋友连接同一个 WiFi，用手机或电脑浏览器打开：'),
          h('div', { class: 'url-big', id: 'lanurl' }, '…'), list),
        h('div', { class: 'qr', id: 'qrbox' })));
    centerMsg.replaceChildren(box);
    centerMsg.style.pointerEvents = 'auto';
    api.get('/api/info').then((info) => {
      const url = (info.lan_urls && info.lan_urls[0]) || location.origin + '/';
      const u = url + '#/join';
      const el = document.getElementById('lanurl');
      if (el) el.textContent = u;
      const q = document.getElementById('qrbox');
      if (q) q.replaceChildren(h('img', { src: '/api/qr?text=' + encodeURIComponent(u), alt: '扫码加入' }));
    }).catch(() => {});
  }

  function ringSvg() {
    const r = 30;
    const c = 2 * Math.PI * r;
    return h('svg', { class: 'ring', viewBox: '0 0 66 66', html: `<circle class="bg" cx="33" cy="33" r="${r}"/><circle class="fg" cx="33" cy="33" r="${r}" stroke-dasharray="${c}" stroke-dashoffset="0"/>` });
  }

  function renderSeats() {
    const s = S.s;
    board.classList.toggle('revealed', Object.keys(s.hands || {}).some((k) => +k !== base()) && (s.mode !== 'teach' || S.reveal || you() < 0));
    for (let r = 0; r < 4; r++) {
      const p = (base() + r) % 4;
      const el = seatEls[r];
      const st = s.seats[p];
      const place = s.finished.indexOf(p);
      const isTurn = s.current === p && (s.phase === 0 || s.phase === 1);
      el.className = `seat ${DIR[r]} ${p % 2 === ourTeam() ? 'team-ours' : 'team-theirs'}${isTurn ? ' turn' : ''}`;
      el.dataset.p = p;
      const avText = st.kind === 'ai' ? (st.name || 'AI').slice(-1) : (st.name || '?').slice(0, 1);
      const av = h('div', { class: 'avatar' + (st.kind === 'ai' ? ' ai' : '') }, avText);
      if (isTurn && s.deadline) av.appendChild(ringSvg());
      if (place >= 0) av.appendChild(h('span', { class: 'badge' + (place === 0 ? ' p1' : '') }, PLACE[place]));
      const cnt = s.counts[p];
      const info = h('div', { class: 'info' },
        h('div', { class: 'nm', title: st.name }, st.name + (r === 0 && you() >= 0 && st.name !== '我' ? '（你）' : '')),
        r === 0 ? null : h('div', { class: 'count' + (cnt > 0 && cnt <= 6 ? ' low' : '') }, h('i', { class: 'cb' }), String(cnt)),
        st.kind === 'human' && !st.connected ? h('div', { class: 'off' }, '离线') : null);
      const kids = [av, info];
      const vis = s.hands[String(p)];
      const showHand = vis && r !== 0 && (s.mode !== 'teach' || S.reveal || you() < 0);
      if (showHand && vis.length) kids.push(miniRow(vis, s.level, 'm'));
      el.replaceChildren(...kids);
    }
  }

  function renderZones() {
    const s = S.s;
    const trick = S.held || s.trick || {};
    const leadP = s.last_move ? s.last_move.player : -1;
    for (let r = 0; r < 4; r++) {
      const p = (base() + r) % 4;
      const z = zoneEls[r];
      const t = trick[String(p)];
      z.className = `zone ${DIR[r]}` + (S.held ? ' fade' : '') + (!S.held && p === leadP ? ' lead' : '');
      if (r === 0 && (myTurn() || (you() >= 0 && s.phase < 2 && s.current === you()))) {
        // keep my zone visible but small when it's my turn
      }
      if (!t) { z.replaceChildren(); continue; }
      if (t.pass_) { z.replaceChildren(h('div', { class: 'pass-bubble' }, '不出')); continue; }
      const row = h('div', { class: 'cards' });
      for (const c of sortCards(t.cards, s.level)) row.appendChild(cardEl(c, { level: s.level, size: 'p' }));
      z.replaceChildren(row, h('div', { class: 'desc' }, t.desc));
    }
  }

  function renderCenter() {
    const s = S.s;
    centerMsg.style.pointerEvents = 'none';
    const parts = [];
    if (s.phase === 0) {
      const pr = s.pending_returns[0];
      if (pr) {
        const [recv, payer] = pr;
        parts.push(recv === you() ? `请选一张 10 或以下的牌还给 ${nameOf(payer)}` : `等待 ${nameOf(recv)} 还贡…`);
      }
    }
    const trib = s.tribute || [];
    const showTrib = trib.length && (s.phase === 0 || (s.phase === 1 && (s.trick && Object.keys(s.trick).length === 0) && s.counts.every((c) => c === 27)));
    if (s.anti_tribute && s.phase === 1 && s.counts.every((c) => c === 27)) parts.push('抗贡！进贡方持有两张大王，由头游先出');
    const box = h('div', { class: 't' });
    for (const p of parts) box.appendChild(h('span', {}, p));
    if (showTrib) {
      for (const t of trib) {
        box.appendChild(h('span', { style: { display: 'inline-flex', alignItems: 'center', gap: '6px' } },
          `${nameOf(t.from)} ${t.is_return ? '还' : '贡'}`, cardEl(t.card, { level: s.level, size: 'm' }), `给 ${nameOf(t.to)}`));
      }
    }
    centerMsg.replaceChildren(...(box.childNodes.length ? [box] : []));
  }

  function renderActions() {
    const s = S.s;
    if (!s) return;
    const kids = [];
    if (s.status !== 'playing') { actionbar.replaceChildren(); return; }
    if (myTurn()) {
      if (s.deadline) kids.push(h('span', { class: 'timer-text', id: 'timer' }, ''));
      if (s.phase === 0) {
        kids.push(h('span', { class: 'yourturn' }, '还贡'),
          h('button', { class: 'btn primary', disabled: S.sel.size !== 1 || S.busy, onclick: doReturn }, '还给他'));
      } else {
        kids.push(
          h('button', { class: 'btn', disabled: s.is_leading || S.busy, onclick: doPass }, '不出'),
          s.hint_available ? h('button', { class: 'btn', disabled: S.busy, onclick: () => send({ action: 'hint' }) }, 'AI 提示') : null,
          h('button', { class: 'btn primary', disabled: S.sel.size === 0 || S.busy, onclick: doPlay }, '出牌'));
      }
    } else if (s.phase < 2) {
      const cur = s.current;
      kids.push(h('span', { class: 'dim' }, you() < 0 ? `观战中，轮到 ${nameOf(cur)}` : `等待 ${nameOf(cur)}${s.phase === 0 ? ' 还贡' : ' 出牌'}…`));
    }
    const tools = h('div', { class: 'tools' });
    if (you() >= 0 && myHand().length) {
      tools.appendChild(h('button', { class: 'btn small', onclick: () => { S.sel.clear(); renderHand(); renderActions(); }, disabled: !S.sel.size }, '重选'));
      tools.appendChild(h('button', { class: 'btn small', onclick: () => { S.desc = !S.desc; renderHand(); } }, S.desc ? '大→小' : '小→大'));
    }
    if (s.mode === 'teach' && you() >= 0) {
      tools.appendChild(h('button', { class: 'btn small' + (S.reveal ? ' primary' : ''), onclick: () => { S.reveal = !S.reveal; renderSeats(); renderActions(); } }, S.reveal ? '盖牌' : '明牌'));
    }
    if (tools.childNodes.length) kids.push(tools);
    actionbar.replaceChildren(...kids.filter(Boolean));
  }

  // ------------------------------------------------------------ hand
  function cardSize() {
    const vh = window.innerHeight;
    const vw = window.innerWidth;
    let ch = Math.min(118, Math.max(58, vh * 0.15));
    if (vw < 480) ch = Math.min(ch, 74);
    return { cw: Math.round(ch * 0.7), ch: Math.round(ch) };
  }

  function renderHand() {
    const s = S.s;
    const cards = myHand();
    const interactive = you() >= 0 && myTurn();
    hand.classList.toggle('disabled', !interactive);
    if (!cards.length) { hand.replaceChildren(); hand.style.height = '8px'; return; }
    const ids = withIds(cards);
    const byType = {};
    for (const c of ids) (byType[c.t] = byType[c.t] || []).push(c.id);
    for (const id of [...S.sel]) if (!ids.some((c) => c.id === id)) S.sel.delete(id);
    const cols = columns(cards, s.level, S.desc);
    const { cw, ch } = cardSize();
    const W = handWrap.clientWidth - 24;
    const n = cols.length;
    const sx = n > 1 ? Math.min(cw * 1.06, (W - cw) / (n - 1)) : 0;
    const maxStack = Math.max(...cols.map((c) => c.length));
    let sy = ch * 0.27;
    const maxH = window.innerHeight * 0.34;
    if (ch + (maxStack - 1) * sy > maxH) sy = Math.max(ch * 0.16, (maxH - ch) / Math.max(1, maxStack - 1));
    const lift = ch * 0.22;
    const totalW = (n - 1) * sx + cw;
    hand.style.width = `${totalW}px`;
    hand.style.height = `${lift + ch + (maxStack - 1) * sy}px`;
    hand.style.setProperty('--cw', `${cw}px`);
    hand.style.setProperty('--ch', `${ch}px`);
    const used = {};
    const els = [];
    cols.forEach((col, i) => {
      col.forEach((t, j) => {
        used[t] = (used[t] || 0);
        const id = byType[t][used[t]++];
        const el = cardEl(t, { level: s.level });
        el.dataset.id = id;
        el.dataset.col = i;
        el.style.left = `${i * sx}px`;
        el.style.top = `${lift + j * sy}px`;
        el.style.zIndex = String(i * 10 + j + 1);
        if (S.sel.has(id)) el.classList.add('sel');
        els.push(el);
      });
    });
    hand.replaceChildren(...els);
    S.layout = { cw, ch };
  }

  // drag-select
  let drag = null;
  function cardFromPoint(x, y) {
    const el = document.elementFromPoint(x, y);
    const c = el && el.closest ? el.closest('.hand .card') : null;
    return c && hand.contains(c) ? c : null;
  }
  function setSel(el, on) {
    const id = el.dataset.id;
    if (!id) return;
    if (on) S.sel.add(id); else S.sel.delete(id);
    el.classList.toggle('sel', on);
  }
  hand.addEventListener('pointerdown', (e) => {
    if (you() < 0) return;
    const c = cardFromPoint(e.clientX, e.clientY);
    if (!c) return;
    e.preventDefault();
    const on = !S.sel.has(c.dataset.id);
    drag = { on, touched: new Set([c.dataset.id]) };
    setSel(c, on);
    sound.select();
    hand.setPointerCapture && hand.setPointerCapture(e.pointerId);
  });
  hand.addEventListener('pointermove', (e) => {
    if (!drag) return;
    const c = cardFromPoint(e.clientX, e.clientY);
    if (c && !drag.touched.has(c.dataset.id)) {
      drag.touched.add(c.dataset.id);
      setSel(c, drag.on);
      sound.select();
    }
  });
  const endDrag = () => { if (drag) { drag = null; renderActions(); } };
  hand.addEventListener('pointerup', endDrag);
  hand.addEventListener('pointercancel', endDrag);
  hand.addEventListener('dblclick', (e) => {
    const c = cardFromPoint(e.clientX, e.clientY);
    if (!c) return;
    const col = c.dataset.col;
    const all = [...hand.querySelectorAll(`.card[data-col="${col}"]`)];
    const on = !all.every((x) => S.sel.has(x.dataset.id));
    all.forEach((x) => setSel(x, on));
    renderActions();
  });
  hand.addEventListener('contextmenu', (e) => { e.preventDefault(); S.sel.clear(); renderHand(); renderActions(); });

  function selectedTypes() {
    return [...S.sel].map((id) => +id.split(':')[0]);
  }
  function doPlay() {
    if (!S.sel.size) return;
    S.busy = true;
    renderActions();
    send({ action: 'play', cards: selectedTypes() });
  }
  function doPass() {
    S.busy = true;
    S.sel.clear();
    renderHand();
    renderActions();
    send({ action: 'pass' });
  }
  function doReturn() {
    const t = selectedTypes();
    if (t.length !== 1) return toast('请选择一张牌', 'err');
    S.busy = true;
    renderActions();
    send({ action: 'return', card: t[0] });
  }
  function selectCards(types) {
    S.sel.clear();
    const els = [...hand.querySelectorAll('.card')];
    for (const t of types) {
      const el = els.find((x) => +x.dataset.t === t && !S.sel.has(x.dataset.id));
      if (el) S.sel.add(el.dataset.id);
    }
    renderHand();
    renderActions();
  }

  // ------------------------------------------------------------ side panel (hints)
  function renderSide() {
    if (!S.hints || !myTurn()) { side.replaceChildren(); return; }
    const s = S.s;
    const box = h('div', { class: 'hints' }, h('h3', {}, h('span', {}, 'AI 建议（期望升级数）'), h('button', { class: 'btn small', onclick: () => { S.hints = null; renderSide(); } }, '收起')));
    S.hints.forEach((m, i) => {
      const v = Math.max(-3, Math.min(3, m.q));
      const w = Math.abs(v) / 3 * 50;
      const bar = h('i', { style: { width: `${w}%`, left: v >= 0 ? '50%' : `${50 - w}%`, background: v >= 0 ? 'var(--brass)' : 'var(--cinnabar-hi)' } });
      const mini = h('div', { class: 'mini' });
      for (const c of sortCards(m.cards, s.level)) mini.appendChild(cardEl(c, { level: s.level, size: 'm' }));
      box.appendChild(h('div', { class: 'hint-row' + (i === 0 ? ' best' : ''), onclick: () => (m.kind === 0 ? doPass() : selectCards(m.cards)) },
        h('span', { class: 'd' }, m.desc), h('span', { class: 'v' }, (m.q >= 0 ? '+' : '') + m.q.toFixed(2)),
        m.cards.length ? mini : null, h('div', { class: 'bar' }, bar)));
    });
    side.style.bottom = `${handWrap.offsetHeight + actionbar.offsetHeight + 6}px`;
    box.style.maxHeight = `${Math.max(180, board.offsetHeight - 20)}px`;
    box.style.overflow = 'auto';
    side.replaceChildren(box);
  }

  function showReview(rv) {
    if (!rv || rv.rank == null) return;
    const node = rv.rank === 1
      ? h('span', {}, `AI 点评：与 AI 首选一致（共 ${rv.total} 种出法）`)
      : h('span', {}, `AI 点评：你的出法在 ${rv.total} 种中排第 ${rv.rank}，期望 ${fmt(rv.chosen_q)}；AI 首选「${rv.best.desc}」期望 ${fmt(rv.best.q)}`);
    toast(node, '', 5200);
  }
  const fmt = (q) => (q == null ? '—' : (q >= 0 ? '+' : '') + q.toFixed(2));

  function chooseDialog(options) {
    const s = S.s;
    const box = h('div', {}, h('h2', {}, '选择牌型'), h('div', { class: 'sub' }, '这手牌可以有多种打法'));
    const list = h('div', { class: 'result-list' });
    let m;
    for (const o of options) {
      list.appendChild(h('button', { class: 'btn', style: { width: '100%' }, onclick: () => { m.close(); S.busy = true; renderActions(); send({ action: 'play', cards: o.cards, kind: o.kind, key: o.key }); } }, o.desc));
    }
    box.append(list, h('div', { class: 'actions' }, h('button', { class: 'btn small', onclick: () => m.close() }, '取消')));
    m = modal(box, { onClose: () => {} });
    void s;
  }

  // ------------------------------------------------------------ result modal
  function showResult() {
    const s = S.s;
    if (!s || !s.result) return;
    if (S.modal) S.modal.close();
    const r = s.result;
    const ot = ourTeam();
    const weWin = r.win_team === ot;
    const box = h('div');
    const fill = () => {
      const s2 = S.s;
      const rr = s2.result || r;
      const title = rr.match_over ? (weWin ? '我方过 A，赢下整场！' : '对方过 A，赢下整场') : (weWin ? (rr.gain === 3 ? '我方双上！' : '我方赢了这一局') : (rr.gain === 3 ? '对方双上' : '对方赢了这一局'));
      const rows = h('div', { class: 'result-list' });
      rr.order.forEach((p, i) => {
        const place = i < 3 || s2.finished.includes(p) ? PLACE[i] : '末游';
        rows.appendChild(h('div', { class: 'result-row' + (p % 2 === ot ? ' ours' : '') },
          h('span', { class: 'pl' }, rr.gain === 3 && i >= 2 ? '下游' : place),
          h('span', {}, nameOf(p), h('span', { class: 'rd' }, p % 2 === ot ? '  我方' : '  对方')),
          h('span', { class: 'rd' }, p === you() ? '你' : '')));
      });
      const wt = rr.win_team;
      let change;
      if (s2.length === 'quick') {
        change = h('div', { class: 'sub' }, `${weWin ? '我方' : '对方'} +${rr.gain} 分　总积分 ${s2.points[ot]} : ${s2.points[1 - ot]}`);
      } else {
        change = h('div', {},
          h('div', { class: 'lvchange' }, h('span', {}, RANKS[rr.before[wt]]), h('span', { class: 'arrow' }, `${wt === ot ? '我方' : '对方'}升 ${rr.gain} 级`), h('span', { class: 'to' }, RANKS[rr.after[wt]])),
          rr.a_reset ? h('div', { class: 'sub' }, '打 A 三次未过，退回打 2') : null);
      }
      const readyHumans = s2.seats.filter((x) => x.kind === 'human');
      const waiting = readyHumans.filter((x) => !x.ready).map((x) => x.name);
      const meReady = you() >= 0 && s2.seats[you()].ready;
      const btnText = rr.match_over && s2.length === 'full' ? '再来一场' : '下一局';
      box.replaceChildren(
        rr.match_over && s2.length === 'full' ? h('div', { class: 'win-banner' }, weWin ? '胜' : '负') : null,
        h('h2', {}, title),
        h('div', { class: 'sub' }, s2.length === 'quick' ? `本局打 ${RANKS[s2.level]}` : `本局打 ${RANKS[s2.level]}`),
        rows, change,
        h('div', { class: 'actions' },
          you() >= 0 ? h('button', { class: 'btn primary', disabled: meReady, onclick: () => send({ action: 'ready' }) }, meReady ? '已准备' : btnText) : null,
          h('button', { class: 'btn', onclick: leave }, '离开')),
        waiting.length && (meReady || you() < 0) ? h('div', { class: 'sub', style: { marginTop: '10px' } }, `等待：${waiting.join('、')}`) : null,
      );
    };
    fill();
    S.modal = modal(box);
    S.modal.update = fill;
    if (weWin) sound.win(); else sound.lose();
  }

  // ------------------------------------------------------------ animations
  function captureHandRects(cards) {
    const els = [...hand.querySelectorAll('.card')];
    const out = {};
    const used = new Set();
    for (const t of cards) {
      const el = els.find((x) => +x.dataset.t === t && x.classList.contains('sel') && !used.has(x)) || els.find((x) => +x.dataset.t === t && !used.has(x));
      if (el) { used.add(el); (out[t] = out[t] || []).push(el.getBoundingClientRect()); }
    }
    return out;
  }

  function seatCenter(p) {
    const el = seatEls[relOf(p)];
    const av = el.querySelector('.avatar') || el;
    const r = av.getBoundingClientRect();
    return { x: r.left + r.width / 2, y: r.top + r.height / 2 };
  }

  function animateEvent(ev, prev, flyFrom) {
    if (!ev || !ev.type) return;
    if (ev.type === 'play') {
      sound.play();
      const z = zoneEls[relOf(ev.player)];
      const cardsEls = [...z.querySelectorAll('.card')];
      if (flyFrom && Object.keys(flyFrom).length) {
        const pool = Object.fromEntries(Object.entries(flyFrom).map(([k, v]) => [k, [...v]]));
        for (const el of cardsEls) {
          const list = pool[el.dataset.t];
          const from = list && list.shift();
          if (!from) continue;
          const to = el.getBoundingClientRect();
          const dx = from.left + from.width / 2 - (to.left + to.width / 2);
          const dy = from.top + from.height / 2 - (to.top + to.height / 2);
          const sc = from.width / to.width;
          el.animate([{ transform: `translate(${dx}px, ${dy}px) scale(${sc})` }, { transform: 'none' }], { duration: 320, easing: 'cubic-bezier(.2,.8,.3,1)' });
        }
      } else {
        const c = seatCenter(ev.player);
        const zr = z.getBoundingClientRect();
        const dx = c.x - (zr.left + zr.width / 2);
        const dy = c.y - (zr.top + zr.height / 2);
        cardsEls.forEach((el, i) => el.animate([{ transform: `translate(${dx}px, ${dy}px) scale(.45)`, opacity: 0.2 }, { transform: 'none', opacity: 1 }], { duration: 360, delay: i * 18, easing: 'cubic-bezier(.2,.8,.3,1)', fill: 'backwards' }));
      }
      if (ev.bomb) boom(BOMB_WORD[ev.kind] || '炸弹');
      if (ev.how === 'timeout') toast(`${nameOf(ev.player)} 超时，系统随机出牌`, 'err');
      if (ev.finished) {
        const place = S.s.finished.indexOf(ev.player);
        if (place >= 0) toast(`${nameOf(ev.player)} 出完了：${PLACE[place]}`);
      }
    } else if (ev.type === 'pass') {
      sound.pass();
      if (ev.how === 'timeout') toast(`${nameOf(ev.player)} 超时`, 'err');
    } else if (ev.type === 'deal') {
      animateDeal();
    } else if (ev.type === 'tribute') {
      const trib = (S.s.tribute || []).filter((t) => !t.is_return);
      if (S.s.anti_tribute) toast('抗贡！进贡方持有两张大王，免进贡');
      trib.forEach((t, i) => setTimeout(() => flyCard(t.card, t.from, t.to), i * 350));
    } else if (ev.type === 'return') {
      flyCard(ev.card, ev.from, ev.to);
      if (ev.how === 'timeout') toast(`${nameOf(ev.from)} 超时，随机还贡`, 'err');
    }
  }

  function flyCard(t, from, to) {
    const a = seatCenter(from);
    const b = seatCenter(to);
    const el = cardEl(t, { level: S.s.level, size: 'p' });
    el.classList.add('flycard');
    el.style.left = `${a.x - 23}px`;
    el.style.top = `${a.y - 32}px`;
    document.body.appendChild(el);
    sound.play();
    requestAnimationFrame(() => requestAnimationFrame(() => { el.style.transform = `translate(${b.x - a.x}px, ${b.y - a.y}px) rotate(8deg)`; }));
    setTimeout(() => { el.style.opacity = '0'; }, 900);
    setTimeout(() => el.remove(), 1500);
  }

  function animateDeal() {
    sound.deal();
    const hr = hand.getBoundingClientRect();
    const cx = window.innerWidth / 2;
    const cy = window.innerHeight * 0.4;
    [...hand.querySelectorAll('.card')].forEach((el, i) => {
      const r = el.getBoundingClientRect();
      el.style.setProperty('--dx', `${cx - r.left - r.width / 2}px`);
      el.style.setProperty('--dy', `${cy - r.top - r.height / 2}px`);
      el.style.animationDelay = `${i * 22}ms`;
      el.classList.add('dealing');
      setTimeout(() => el.classList.remove('dealing'), 600 + i * 22);
    });
    void hr;
  }

  function boom(word) {
    sound.bomb();
    const layer = h('div', { class: 'boom' }, h('div', { class: 'flash' }), h('div', { class: 'word' }, word));
    document.body.appendChild(layer);
    root.classList.remove('shake');
    void root.offsetWidth;
    root.classList.add('shake');
    setTimeout(() => layer.remove(), 1200);
    setTimeout(() => root.classList.remove('shake'), 600);
  }

  // ------------------------------------------------------------ timer loop
  function tick() {
    S.raf = requestAnimationFrame(tick);
    const s = S.s;
    if (!s || !s.deadline || s.phase >= 2) return;
    const now = Date.now() + S.offset;
    const remain = Math.max(0, (s.deadline - now) / 1000);
    const total = s.timeout || 20;
    const ring = seatEls[relOf(s.current)].querySelector('.ring');
    if (ring) {
      const fg = ring.querySelector('.fg');
      const c = 2 * Math.PI * 30;
      fg.setAttribute('stroke-dashoffset', String(c * (1 - remain / total)));
      ring.classList.toggle('warn', remain <= 5);
    }
    const tt = document.getElementById('timer');
    if (tt) {
      const secs = Math.ceil(remain);
      tt.textContent = String(secs);
      tt.classList.toggle('warn', remain <= 5);
      if (myTurn() && remain <= 5 && secs !== S.lastTick && secs > 0) { S.lastTick = secs; sound.tick(); }
    }
  }
  S.raf = requestAnimationFrame(tick);

  // ------------------------------------------------------------ misc
  function leave() {
    const box = h('div', {}, h('h2', {}, '离开牌桌？'), h('div', { class: 'sub' }, '牌桌会继续进行。之后可以从首页回到牌桌。'));
    let m;
    box.appendChild(h('div', { class: 'actions' },
      h('button', { class: 'btn', onclick: () => m.close() }, '留下'),
      h('button', { class: 'btn danger', onclick: () => { m.close(); location.hash = '#/'; } }, '离开')));
    m = modal(box, { onClose: () => {} });
  }

  const onResize = () => { if (S.s && S.s.status === 'playing') renderHand(); };
  window.addEventListener('resize', onResize);
  const onKey = (e) => {
    if (!myTurn() || e.target.tagName === 'INPUT') return;
    if (e.key === 'Enter') doPlay();
    else if (e.key === ' ' && !S.s.is_leading) { e.preventDefault(); doPass(); }
    else if (e.key === 'Escape') { S.sel.clear(); renderHand(); renderActions(); }
  };
  window.addEventListener('keydown', onKey);
  connect();

  return () => {
    S.closed = true;
    cancelAnimationFrame(S.raf);
    clearTimeout(S.heldTimer);
    window.removeEventListener('resize', onResize);
    window.removeEventListener('keydown', onKey);
    if (S.modal) S.modal.close();
    try { S.ws && S.ws.close(); } catch {}
    document.querySelectorAll('.modal-back, .flycard, .boom').forEach((x) => x.remove());
  };
}

void cardName; void rankOf;
