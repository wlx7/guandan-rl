import { cardEl, columns, withIds, sortCards, miniRow, cardName, RANKS, SUITS, rankOf, arrangeHand, cardCounter, straightFlushes, suitSvg } from './cards.js';
import { sound } from './sound.js';
import { api, toast, h, seatStore, modal, wsAuth } from './util.js';

const PLACE = ['头游', '二游', '三游', '末游'];
const BOMB_WORD = { 8: '炸弹', 9: '同花顺', 10: '天王炸' };
const DIR = ['bottom', 'right', 'top', 'left'];

const pref = {
  get(k, d) { try { const v = localStorage.getItem('gd_' + k); return v === null ? d : JSON.parse(v); } catch { return d; } },
  set(k, v) { try { localStorage.setItem('gd_' + k, JSON.stringify(v)); } catch {} },
};

// landscape phones: see renderActions (same breakpoint as the short-screen CSS)
const compactMQ = window.matchMedia('(max-height: 520px)');

export function mountTable(app, roomId, watch) {
  // watch: open the table as a spectator even if this device has a seat there
  const mine = seatStore.get(roomId);
  const seat = (!watch && mine && mine.token) ? mine : { room: roomId, seat: -1, token: null };
  if (seat.token) seatStore.set(seat); // remember it as this device's current table
  const S = {
    s: null, prev: null, ws: null, closed: false, retry: 0,
    sel: new Set(), desc: true, reveal: false, hints: null, hintsSeq: -1,
    arrange: pref.get('arrange', false), counter: pref.get('counter', false), custom: [], sfPick: {},
    held: null, heldTimer: null, offset: 0, lastTick: -1, raf: 0, busy: false,
    reviewKey: null, resultShown: null, modal: null, layout: null, lastTurnKey: null,
  };

  // ------------------------------------------------------------ skeleton
  const backBtn = h('button', { class: 'btn small icon', title: '离开牌桌', 'aria-label': '离开牌桌', onclick: leave }, '离开');
  const soundLabel = () => (sound.enabled ? '♪' : '♪̸');
  const soundBtn = h('button', { class: 'btn small icon' + (sound.enabled ? '' : ' muted'), title: '音效开关', 'aria-label': '音效开关', onclick: () => { sound.toggle(); soundBtn.textContent = soundLabel(); soundBtn.classList.toggle('muted', !sound.enabled); } }, soundLabel());
  const lvBox = h('div', { class: 'lvbox' });
  const meta = h('div', { class: 'meta' });
  const helpBtn = h('button', { class: 'btn small icon', title: '规则与操作', 'aria-label': '规则与操作', onclick: showHelp }, '?');
  const pauseBtn = h('button', { class: 'btn small icon', title: '暂停', 'aria-label': '暂停牌局', onclick: () => send({ action: 'pause' }) }, '⏸');
  // full screen hides the phone browser's address bar (more room for the table)
  // the Android app is fullscreen already (its WebView has no fullscreen API)
  const canFull = !/GuandanApp/.test(navigator.userAgent) && !!(document.documentElement.requestFullscreen || document.documentElement.webkitRequestFullscreen);
  const fullBtn = canFull ? h('button', { class: 'btn small icon', title: '全屏', 'aria-label': '全屏', onclick: toggleFull }, '⛶') : null;
  function toggleFull() {
    const d = document;
    if (d.fullscreenElement || d.webkitFullscreenElement) {
      (d.exitFullscreen || d.webkitExitFullscreen).call(d);
      return;
    }
    const el = d.documentElement;
    const req = el.requestFullscreen || el.webkitRequestFullscreen;
    Promise.resolve(req.call(el, { navigationUI: 'hide' })).then(() => {
      try { screen.orientation && screen.orientation.lock && screen.orientation.lock('landscape').catch(() => {}); } catch {}
    }).catch(() => {});
  }
  const counterEl = h('div', { class: 'counter', title: '记牌器：其他三家手里还剩的牌（不含已出的牌和你的手牌）' });
  const topbar = h('div', { class: 'topbar' }, backBtn, h('div', { class: 'grow' }, lvBox), meta, pauseBtn, fullBtn, helpBtn, soundBtn, counterEl);
  const board = h('div', { class: 'board' });
  const seatEls = DIR.map((d) => h('div', { class: `seat ${d}` }));
  const zoneEls = DIR.map((d) => h('div', { class: `zone ${d}` }));
  const centerMsg = h('div', { class: 'center-msg' });
  const actionbar = h('div', { class: 'actionbar' });
  const toolsEl = h('div', { class: 'hand-bar' });
  board.append(...zoneEls, ...seatEls, centerMsg, actionbar);
  const hand = h('div', { class: 'hand' });
  const handWrap = h('div', { class: 'hand-wrap' }, hand);
  const side = h('div', { class: 'side-panel' });
  const root = h('div', { class: 'gt' }, topbar, board, handWrap, toolsEl, side);
  app.replaceChildren(root);

  // ------------------------------------------------------------ socket
  function connect() {
    if (S.closed) return;
    const proto = location.protocol === 'https:' ? 'wss' : 'ws';
    const qs = [wsAuth()];
    if (roomId) qs.push('room=' + encodeURIComponent(roomId));
    if (seat.token) qs.push('token=' + encodeURIComponent(seat.token));
    const q = '?' + qs.join('&');
    const ws = new WebSocket(`${proto}://${location.host}/ws${q}`);
    S.ws = ws;
    ws.onopen = () => { S.retry = 0; };
    ws.onmessage = (e) => {
      let m;
      try { m = JSON.parse(e.data); } catch { return; }
      onMessage(m);
    };
    ws.onclose = () => {
      if (S.closed || S.away) return;
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
    if (m.type === 'closed') { // the table was ended (by a player or the host)
      S.closed = true;
      if (S.modal) S.modal.close();
      toast(m.message || '牌桌已关闭');
      setTimeout(() => { location.hash = '#/'; }, S.ending ? 200 : 1500);
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
  const isPaused = () => !!(S.s && (S.s.paused || S.s.pause_by));
  const myTurn = () => S.s && you() >= 0 && !isPaused() && S.s.current === you() && S.s.status === 'playing' && (S.s.phase === 0 || S.s.phase === 1);
  const myHand = () => (S.s ? (you() >= 0 ? S.s.hand || [] : S.s.hands['0'] || []) : []);

  // ------------------------------------------------------------ state
  function onState(s) {
    const prev = S.s;
    S.offset = s.server_time - Date.now();
    const ev = s.event || {};
    // a state can be sent again without anything happening (a player joins, pauses, gets ready):
    // animate and announce only events that were not shown yet
    const isNew = !prev || (s.seq !== prev.seq && (ev.id == null || !prev.event || prev.event.id !== ev.id));
    let flyFrom = null;
    if (isNew && prev && ev.type === 'play' && ev.player === you()) flyFrom = captureHandRects(ev.cards);
    const newRound = !prev || prev.round_no !== s.round_no || prev.match_no !== s.match_no || (prev.phase >= 2 && s.phase < 2);
    S.prev = prev;
    S.s = s;
    if (newRound) {
      S.sel.clear();
      S.custom = [];
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
    placeZones();
    renderCenter();
    renderPause();
    renderActions();
    renderHand();
    renderCounter();
    renderSide();
  }

  function renderCounter() {
    const s = S.s;
    const on = S.counter && s && s.status === 'playing' && you() >= 0 && s.played;
    counterEl.classList.toggle('on', !!on);
    topbar.classList.toggle('counter-on', !!on);
    if (!on) { counterEl.replaceChildren(); return; }
    const cells = cardCounter(myHand(), s.played, s.level).map((c) => h('div', {
      class: 'cc' + (c.level ? ' lv' : '') + (c.r >= 13 ? ' jk' : '') + (c.r === 14 ? ' big' : '') + (c.left === 0 ? ' zero' : ''),
    }, h('span', { class: 'l' }, c.label), h('b', {}, String(c.left))));
    counterEl.replaceChildren(...cells);
  }

  function renderTop() {
    const s = S.s;
    const ot = ourTeam();
    const lv = (t) => RANKS[s.team_level[t]];
    const prevLevel = S.prev ? S.prev.level : s.level;
    const plaque = h('div', { class: 'plaque' + (prevLevel !== s.level ? ' pulse' : '') },
      h('div', { class: 'big' }, '打 ', h('span', { class: 'n' }, RANKS[s.level])),
      h('div', { class: 'sub' }, `红桃${RANKS[s.level]} 逢人配`));
    const multBadge = s.scoring && s.scoring !== 'none'
      ? h('div', { class: 'mult' + (S.prev && S.prev.multiplier !== s.multiplier && s.multiplier > 1 ? ' pulse' : ''), title: '本局倍数' }, h('span', {}, '倍数'), h('b', {}, `×${s.multiplier}`))
      : null;
    if (s.length === 'quick') {
      lvBox.replaceChildren(...[plaque, multBadge].filter(Boolean));
    } else {
      lvBox.replaceChildren(
        h('div', { class: 'teamlv ours' + (s.banker === ot ? ' banker' : '') }, h('span', {}, '我方'), h('b', {}, lv(ot))),
        plaque,
        h('div', { class: 'teamlv' + (s.banker === 1 - ot ? ' banker' : '') }, h('span', {}, '对方'), h('b', {}, lv(1 - ot))),
        ...(multBadge ? [multBadge] : []));
    }
    const modeName = s.mode === 'teach' ? '教学' : '实战';
    const scored = s.scoring && s.scoring !== 'none';
    const line2 = s.length === 'quick'
      ? `积分 ${s.points[ot]} : ${s.points[1 - ot]}`
      : `第 ${s.match_no} 场  胜场 ${s.score[ot]} : ${s.score[1 - ot]}` + (scored ? `  总分 ${s.totals[ot]} : ${s.totals[1 - ot]}` : '');
    meta.replaceChildren(h('div', {}, h('b', {}, `${modeName}${s.length === 'quick' ? '快速' : '竞技'}`), `  第 ${s.round_no} 局`), h('div', {}, line2 + (s.think === 'deep' ? '  AI 推演搜索' : '')));
  }

  function renderWaiting() {
    const s = S.s;
    pauseBtn.style.display = 'none';
    seatEls.forEach((e) => e.replaceChildren());
    zoneEls.forEach((e) => e.replaceChildren());
    actionbar.replaceChildren();
    toolsEl.replaceChildren();
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
        h('div', { class: 'qr', id: 'qrbox' })),
      h('div', { style: { marginTop: '14px', display: 'flex', gap: '10px', flexWrap: 'wrap' } },
        S.canManage ? h('button', { class: 'btn small', onclick: () => api.post('/api/room/fill_ai', { room: s.room }).catch((e) => toast(e.message, 'err')) }, '空位由 AI 补上，直接开始') : null,
        h('button', { class: 'btn small', onclick: () => { location.hash = '#/'; } }, '返回首页')));
    centerMsg.replaceChildren(box);
    centerMsg.style.pointerEvents = 'auto';
    api.get('/api/info').then((info) => {
      const url = (info.lan_urls && info.lan_urls[0]) || location.origin + '/';
      const u = url + '#/join/' + s.room;
      const el = document.getElementById('lanurl');
      if (el) el.textContent = u;
      const q = document.getElementById('qrbox');
      if (q) q.replaceChildren(h('img', { src: '/api/qr?text=' + encodeURIComponent(u), alt: '扫码加入' }));
    }).catch(() => {});
  }

  function ringSvg() {
    const r = 30;
    const c = 2 * Math.PI * r;
    const wrap = document.createElement('div');
    wrap.innerHTML = `<svg class="ring" viewBox="0 0 66 66" aria-hidden="true"><circle class="bg" cx="33" cy="33" r="${r}"/><circle class="fg" cx="33" cy="33" r="${r}" stroke-dasharray="${c}" stroke-dashoffset="0"/></svg>`;
    return wrap.firstChild;
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
      el.className = `seat ${DIR[r]} ${p % 2 === ourTeam() ? 'team-ours' : 'team-theirs'}${isTurn ? ' turn' : ''}${st.kind === 'ai' ? ' is-ai' : ''}`;
      el.dataset.p = p;
      const avText = st.kind === 'ai' ? (st.name || 'AI').slice(-1) : (st.name || '?').slice(0, 1);
      const av = h('div', { class: 'avatar' + (st.kind === 'ai' ? ' ai' : '') }, avText);
      if (isTurn && s.deadline) av.appendChild(ringSvg());
      // the finishing badge stamps in once; later renders keep it still
      const bkey = `${s.match_no}-${s.round_no}-${p}`;
      S.badges = S.badges || new Set();
      const justDone = place >= 0 && !S.badges.has(bkey);
      if (place >= 0) S.badges.add(bkey);
      if (place >= 0) av.appendChild(h('span', { class: 'badge' + (place === 0 ? ' p1' : '') + (justDone ? ' new' : '') }, PLACE[place]));
      const cnt = s.counts[p];
      const info = h('div', { class: 'info' },
        h('div', { class: 'nm', title: st.name }, st.name + (r === 0 && you() >= 0 && st.name !== '我' ? '（你）' : '')),
        r === 0 ? null : h('div', { class: 'count' + (cnt > 0 && cnt <= 6 ? ' low' : '') }, h('i', { class: 'cb' }), String(cnt)),
        st.kind === 'human' && !st.connected ? h('div', { class: 'off' }, s.paused || s.status !== 'playing' ? '离线' : '离线 · AI 托管') : null);
      const kids = [av, info];
      const vis = s.hands[String(p)];
      const showHand = vis && r !== 0 && (s.mode !== 'teach' || S.reveal || you() < 0);
      if (showHand && vis.length) kids.push(miniRow(vis, s.level, 'm'));
      el.replaceChildren(...kids);
    }
  }

  // landscape phones: the left / right plays start right next to that player's chip, measured
  // (names, card counts and the camera cut-out / safe area in full screen move the chips)
  function placeZones() {
    const zl = zoneEls[3], zr = zoneEls[1];
    if (!compactMQ.matches) { zl.style.left = ''; zr.style.right = ''; return; }
    const br = board.getBoundingClientRect();
    // the seat's box plus whatever sticks out of it (card counter, place badge)
    const span = (el) => {
      let a = Infinity, b = -Infinity;
      for (const e of [el, ...el.querySelectorAll('*')]) {
        const r = e.getBoundingClientRect();
        if (!r.width) continue;
        a = Math.min(a, r.left); b = Math.max(b, r.right);
      }
      return { left: a, right: b, width: b > a ? b - a : 0 };
    };
    const L = span(seatEls[3]);
    const R = span(seatEls[1]);
    if (L.width) zl.style.left = `${Math.round(L.right - br.left + 8)}px`;
    if (R.width) zr.style.right = `${Math.round(br.right - R.left + 8)}px`;
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

  function renderPause() {
    const s = S.s;
    pauseBtn.style.display = s.status === 'playing' && !isPaused() && (you() >= 0 || S.canManage) ? '' : 'none';
    if (!isPaused() || s.phase >= 2) return;
    const box = h('div', { class: 'panel pause-box' },
      h('h2', {}, '牌局已暂停'),
      h('div', { class: 'sub' }, s.pause_by ? `${s.pause_by} 按了暂停，点“继续”从原处接着打。`
        : (s.seats.some((st) => st.kind === 'human') ? '玩家都离开了，他们回来后从原处继续。' : '没人观战时这桌会暂停。')),
      s.pause_by && (you() >= 0 || S.canManage) ? h('div', { class: 'actions' }, h('button', { class: 'btn primary', onclick: () => send({ action: 'resume' }) }, '继续')) : null);
    centerMsg.replaceChildren(box);
    centerMsg.style.pointerEvents = 'auto';
  }

  function renderActions() {
    const s = S.s;
    if (!s) return;
    const kids = [];
    board.classList.toggle('myturn', !!myTurn());
    if (s.status !== 'playing') { actionbar.replaceChildren(); toolsEl.replaceChildren(); return; }
    if (isPaused()) {
      actionbar.replaceChildren();
    } else if (myTurn()) {
      if (s.deadline) kids.push(h('span', { class: 'timer-text', id: 'timer' }, ''));
      if (s.phase === 0) {
        kids.push(h('span', { class: 'yourturn' }, '还贡'),
          h('button', { class: 'btn primary', disabled: S.sel.size !== 1 || S.busy, onclick: doReturn }, '还给他'));
      } else {
        kids.push(
          h('button', { class: 'btn', disabled: s.is_leading || S.busy, onclick: doPass }, '不出'),
          s.hint_available ? h('button', { class: 'btn', disabled: S.busy, onclick: () => send({ action: 'hint' }) }, '提示') : null,
          h('button', { class: 'btn primary', disabled: S.sel.size === 0 || S.busy, onclick: doPlay }, '出牌'));
      }
    } else if (s.phase < 2) {
      const cur = s.current;
      kids.push(h('span', { class: 'wait' }, you() < 0 ? `观战中，轮到 ${nameOf(cur)}` : `等待 ${nameOf(cur)}${s.phase === 0 ? ' 还贡' : ' 出牌'}…`));
    }
    // landscape phones (short screens): the table between the top bar and the hand is only about
    // 90 px high, so the play buttons go to the tool bar under the hand (middle) instead of
    // floating above the hand, where large plays of the other players would cover them
    const playKids = kids.filter(Boolean);
    const inBar = compactMQ.matches && !!myTurn() && playKids.length > 0;
    board.classList.toggle('bar-play', inBar);
    actionbar.replaceChildren(...(inBar ? [] : playKids));
    // tools: a bar under the hand (left: arrange, middle: straight flushes or the play buttons,
    // right: selection tools and switches)
    const L = [], M = [], R = [];
    const cards = myHand();
    if (you() >= 0 && cards.length) {
      const arranged = S.arrange || S.custom.length;
      L.push(h('button', { class: 'btn small' + (arranged ? ' primary' : ''), title: arranged ? '恢复按点数排列' : '把炸弹、同花顺、顺子、三带二等组合各放一列（双击一列可整列选中）',
        onclick: () => {
          if (arranged) { S.arrange = false; S.custom = []; } else S.arrange = true;
          pref.set('arrange', S.arrange); S.sel.clear(); renderHand(); renderActions();
        } }, arranged ? '恢复' : '一键理牌'));
      if (!arranged) L.push(h('button', { class: 'btn small', onclick: () => { S.desc = !S.desc; renderHand(); renderActions(); } }, S.desc ? '大→小' : '小→大'));
      const sf = h('span', { class: 'sfpick', title: '选出这个花色的同花顺（再点换下一个）' }, h('span', { class: 'lab' }, '同花顺'));
      for (let su = 0; su < 4; su++) {
        const list = straightFlushes(cards, s.level, su);
        sf.appendChild(h('button', { class: 'sfb' + (su === 1 || su === 3 ? ' red' : ''), disabled: !list.length, title: SUITS[su], 'aria-label': SUITS[su], onclick: () => pickSF(su), html: suitSvg(su) }));
      }
      if (inBar) R.push(sf); else M.push(sf);
      if (S.sel.size) {
        R.push(h('button', { class: 'btn small', title: '把选中的牌放成一列', onclick: groupSelected }, '理牌'));
        R.push(h('button', { class: 'btn small', onclick: () => { S.sel.clear(); renderHand(); renderActions(); } }, '重选'));
      }
    }
    if (you() >= 0 && s.played) {
      R.push(h('button', { class: 'btn small' + (S.counter ? ' primary' : ''), title: '显示其他三家还剩哪些牌',
        onclick: () => { S.counter = !S.counter; pref.set('counter', S.counter); renderCounter(); renderActions(); } }, '记牌器'));
    }
    if (s.mode === 'teach' && you() >= 0) {
      R.push(h('button', { class: 'btn small' + (S.reveal ? ' primary' : ''), onclick: () => { S.reveal = !S.reveal; renderSeats(); renderActions(); } }, S.reveal ? '盖牌' : '明牌'));
    }
    if (inBar) M.push(h('div', { class: 'play-group' }, ...playKids));
    toolsEl.replaceChildren(h('div', { class: 'g l' }, ...L), h('div', { class: 'g m' }, ...M), h('div', { class: 'g r' }, ...R));
    toolsEl.classList.toggle('blank', !(L.length || M.length || R.length));
  }

  function groupSelected() {
    if (!S.sel.size) return;
    const types = selectedTypes();
    const take = {};
    for (const t of types) take[t] = (take[t] || 0) + 1;
    S.custom = S.custom.map((g) => g.filter((t) => { if (take[t] > 0) { take[t]--; return false; } return true; })).filter((g) => g.length);
    S.custom.push(types);
    S.sel.clear();
    renderHand();
    renderActions();
  }
  function pickSF(suit) {
    const list = straightFlushes(myHand(), S.s.level, suit);
    if (!list.length) return;
    const i = (S.sfPick[suit] || 0) % list.length;
    S.sfPick[suit] = i + 1;
    selectCards(list[i]);
  }

  // ------------------------------------------------------------ hand
  // Hand layout in the style of mobile card apps: one column per rank (or per combination after
  // 理牌), columns stand on a common baseline and grow upwards; every covered card shows a strip
  // with its rank and suit, so each card is a comfortable tap target. On a tall narrow screen
  // the columns wrap into two rows.
  function handLayout(cols) {
    const vh = window.innerHeight;
    const vw = window.innerWidth;
    const short = vh < 560;
    const n = cols.length;
    const rows = vh > vw * 1.1 && n > 8 ? 2 : 1;
    const per = Math.ceil(n / rows);
    const rowCols = Array.from({ length: rows }, (_, r) => cols.slice(r * per, (r + 1) * per));
    const stacks = rowCols.map((rc) => Math.max(1, ...rc.map((c) => c.length)));
    const W = Math.max(200, handWrap.clientWidth - 16);
    const cap = short ? Math.min(62, vh * 0.16) : vw < 480 ? 66 : 92;
    let cw = Math.max(short ? 40 : 42, Math.min(cap, W / (1 + Math.max(0, per - 1) * 0.78)));
    let ch = cw * 1.4;
    let sy = ch * 0.34;
    // leave the table enough room to show everyone's plays (two bands of played cards on phones)
    const boardMin = short ? 112 : 220;
    const avail = vh - topbar.offsetHeight - (toolsEl.offsetHeight || 34) - 8 - boardMin;
    const maxH = Math.max(short ? 120 : 160, Math.min(vh * (vh < 420 ? 0.44 : 0.48), avail));
    // the second row may cover the lower part (big suit pip) of the first row's bottom cards
    let over = 0;
    const total = (c, y) => stacks.reduce((a, m) => a + c + (m - 1) * y, 0) - (rows - 1) * over * c;
    if (rows > 1 && total(ch, sy) > maxH) over = Math.min(0.45, (total(ch, sy) - maxH) / ch);
    if (total(ch, sy) > maxH) {
      const minSy = ch * 0.26;
      const free = stacks.reduce((a, m) => a + m - 1, 0);
      sy = Math.max(minSy, (maxH - total(ch, 0)) / Math.max(1, free));
      if (total(ch, sy) > maxH) {
        const k = maxH / total(ch, sy);
        cw *= k; ch *= k; sy *= k;
      }
    }
    cw = Math.round(cw); ch = Math.round(ch); sy = Math.round(sy);
    const sx = per > 1 ? Math.min(cw + 3, (W - cw) / (per - 1)) : 0;
    const rowTop = [0];
    for (let r = 1; r < rows; r++) rowTop.push(rowTop[r - 1] + ch + (stacks[r - 1] - 1) * sy - over * ch);
    const height = rowTop[rows - 1] + ch + (stacks[rows - 1] - 1) * sy;
    const width = (Math.max(...rowCols.map((rc) => rc.length)) - 1) * sx + cw;
    return { cw, ch, sx, sy, per, stacks, rowTop, height, width };
  }

  // Columns to show: manual groups (理牌 on selected cards) first, then the rest either as
  // combinations (一键理牌) or by rank.
  function handColumns(cards) {
    const s = S.s;
    const left = {};
    for (const t of cards) left[t] = (left[t] || 0) + 1;
    const cols = [];
    const kinds = [];
    const custom = [];
    for (const g of S.custom) {
      const got = [];
      for (const t of g) if (left[t] > 0) { left[t]--; got.push(t); }
      if (got.length) { custom.push(got); cols.push(sortCards(got, s.level, true)); kinds.push(got.length > 1 ? '自选' : null); }
    }
    S.custom = custom;
    const rest = [];
    for (const [t, k] of Object.entries(left)) for (let i = 0; i < k; i++) rest.push(+t);
    if (S.arrange) {
      for (const g of arrangeHand(rest, s.level)) { cols.push(g.cards); kinds.push(g.cards.length > 1 ? g.kind : null); }
    } else {
      for (const c of columns(rest, s.level, S.desc)) { cols.push(c); kinds.push(null); }
    }
    return { cols, kinds };
  }

  function renderHand() {
    const s = S.s;
    const cards = myHand();
    const interactive = you() >= 0 && myTurn();
    hand.classList.toggle('disabled', !interactive);
    hand.classList.toggle('arranged', !!(S.arrange || S.custom.length));
    if (!cards.length) { hand.replaceChildren(); hand.style.height = '8px'; return; }
    const ids = withIds(cards);
    const byType = {};
    for (const c of ids) (byType[c.t] = byType[c.t] || []).push(c.id);
    for (const id of [...S.sel]) if (!ids.some((c) => c.id === id)) S.sel.delete(id);
    const { cols, kinds } = handColumns(cards);
    const { cw, ch, sx, sy, per, stacks, rowTop, height, width } = handLayout(cols);
    hand.style.width = `${width}px`;
    hand.style.height = `${height}px`;
    hand.style.setProperty('--cw', `${cw}px`);
    hand.style.setProperty('--ch', `${ch}px`);
    hand.style.setProperty('--sy', `${sy}px`);
    const used = {};
    const els = [];
    cols.forEach((col, i) => {
      const row = Math.floor(i / per);
      const x = (i % per) * sx;
      const y0 = rowTop[row] + (stacks[row] - col.length) * sy;  // columns stand on the row's baseline
      col.forEach((t, j) => {
        used[t] = (used[t] || 0);
        const id = byType[t][used[t]++];
        const el = cardEl(t, { level: s.level });
        el.dataset.id = id;
        el.dataset.col = i;
        el.style.left = `${x}px`;
        el.style.top = `${y0 + j * sy}px`;
        el.style.zIndex = String(row * 1000 + i * 10 + j + 1);
        if (S.sel.has(id)) el.classList.add('sel');
        if (kinds[i] && j === col.length - 1) el.dataset.group = kinds[i];
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
    const now = Date.now();
    S.tap2 = S.tap1 && S.tap1.id === c.dataset.id && now - S.tap1.t < 450;  // second tap on the same card
    S.tap1 = { id: c.dataset.id, t: now };
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
    // two quick taps on two different cards of one column are two selections, not a double tap
    if (!c || !S.tap2 || (S.tap1 && S.tap1.id !== c.dataset.id)) return;
    S.tap2 = false;
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
    const box = h('div', { class: 'hints' }, h('h3', {}, h('span', {}, S.s && S.s.think === 'deep' ? 'AI 建议（推演后的期望升级数）' : 'AI 建议（期望升级数）'), h('button', { class: 'btn small', onclick: () => { S.hints = null; renderSide(); } }, '收起')));
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
    const below = handWrap.offsetHeight + toolsEl.offsetHeight + 8;
    side.style.bottom = `${below}px`;
    // never taller than the space between the top bar and the hand (short phone screens: a
    // compact list that scrolls)
    box.style.maxHeight = `${Math.max(72, Math.min(board.offsetHeight + 40, window.innerHeight - below - topbar.offsetHeight - 4))}px`;
    box.style.overflow = 'auto';
    side.replaceChildren(box);
  }

  function showReview(rv) {
    if (!rv || rv.rank == null) return;
    const node = rv.rank === 1
      ? h('span', {}, `AI 点评：与 AI 首选一致（共 ${rv.total} 种出法）`)
      : rv.searched === false
        ? h('span', {}, `AI 点评：这一手不在 AI 重点推演的几手之内（网络估计 ${fmt(rv.chosen_q)}）；AI 首选「${rv.best.desc}」期望 ${fmt(rv.best.q)}`)
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
      const title = rr.match_over && s2.length === 'full' ? (weWin ? '我方过 A，赢下整场！' : '对方过 A，赢下整场') : (weWin ? (rr.gain === 3 ? '我方双上！' : '我方赢了这一局') : (rr.gain === 3 ? '对方双上' : '对方赢了这一局'));
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
      const scoreLine = rr.score != null
        ? h('div', { class: 'scoreline' }, h('span', {}, `底分 ${rr.gain}`), h('span', { class: 'x' }, '×'), h('span', {}, `倍数 ${s2.multiplier}`), h('span', { class: 'x' }, '='),
          h('b', { class: weWin ? '' : 'neg' }, `${weWin ? '+' : '-'}${rr.score} 分`))
        : null;
      if (s2.length === 'quick') {
        change = h('div', {}, scoreLine, h('div', { class: 'sub' }, rr.score != null
          ? `总积分 ${s2.points[ot]} : ${s2.points[1 - ot]}`
          : `${weWin ? '我方' : '对方'} +${rr.gain} 分　总积分 ${s2.points[ot]} : ${s2.points[1 - ot]}`));
      } else if (scoreLine) {
        change = h('div', {},
          h('div', { class: 'lvchange' }, h('span', {}, RANKS[rr.before[wt]]), h('span', { class: 'arrow' }, `${wt === ot ? '我方' : '对方'}升 ${rr.gain} 级`), h('span', { class: 'to' }, RANKS[rr.after[wt]])),
          scoreLine, h('div', { class: 'sub' }, `总分 ${s2.totals[ot]} : ${s2.totals[1 - ot]}`));
      } else {
        change = h('div', {},
          h('div', { class: 'lvchange' }, h('span', {}, RANKS[rr.before[wt]]), h('span', { class: 'arrow' }, `${wt === ot ? '我方' : '对方'}升 ${rr.gain} 级`), h('span', { class: 'to' }, RANKS[rr.after[wt]])),
          rr.a_reset ? h('div', { class: 'sub' }, '打 A 三次未过，退回打 2') : null);
      }
      const readyHumans = s2.seats.filter((x) => x.kind === 'human');
      const waiting = readyHumans.filter((x) => !x.ready).map((x) => x.name);
      const meReady = you() >= 0 && s2.seats[you()].ready;
      const btnText = rr.match_over && s2.length === 'full' ? '再来一场' : '下一局';
      box.replaceChildren(...[
        rr.match_over && s2.length === 'full' ? h('div', { class: 'win-banner' }, weWin ? '胜' : '负') : null,
        h('h2', {}, title),
        h('div', { class: 'sub' }, s2.length === 'quick' ? `本局打 ${RANKS[s2.level]}` : `本局打 ${RANKS[s2.level]}`),
        rows, change,
        h('div', { class: 'actions' },
          you() >= 0 ? h('button', { class: 'btn primary', disabled: meReady, onclick: () => send({ action: 'ready' }) }, meReady ? '已准备' : btnText) : null,
          canEnd() ? h('button', { class: 'btn danger', onclick: endTable }, '结束牌局') : null,
          h('button', { class: 'btn', onclick: leave }, '离开')),
        waiting.length && (meReady || you() < 0) ? h('div', { class: 'sub', style: { marginTop: '10px' } }, `等待：${waiting.join('、')}`) : null,
      ].filter(Boolean));
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
  function showHelp() {
    const lv = S.s ? RANKS[S.s.level] : '2';
    const row = (a, b) => h('div', { class: 'result-row', style: { gridTemplateColumns: '6em 1fr' } }, h('span', { class: 'pl' }, a), h('span', { class: 'rd', style: { color: 'var(--text)' } }, b));
    const box = h('div', {},
      h('h2', {}, '规则速查'),
      h('div', { class: 'sub' }, `本局打 ${lv}：${lv} 排在 A 之上，红桃 ${lv} 是逢人配，可以当除王以外的任何牌`),
      h('div', { class: 'result-list' },
        row('单张对子', '大王 > 小王 > 级牌 > A > K … > 2'),
        row('三带二', '三张同点带一对，比三张的大小'),
        row('顺子', '正好 5 张，A2345 最小、10JQKA 最大'),
        row('三连对', '3 个连续对子，如 334455'),
        row('钢板', '2 个连续三张，如 333444'),
        row('炸弹', '天王炸 > 6 张以上炸弹 > 同花顺 > 5 张炸弹 > 4 张炸弹'),
        row('接风', '出完最后一手牌没人要时，由队友接着出'),
        row('升级', '双上升 3 级，队友三游升 2 级，队友末游升 1 级')),
      h('div', { class: 'sub', style: { marginTop: '14px', textAlign: 'left' } },
        '随打随停：离开牌桌（或手机切到后台几秒）牌局就暂停，回来从原处接着打；有别的玩家在时，轮到你由 AI 托管。也可以点 ⏸ 暂停。操作：点牌或按住拖动选牌（选中的牌变成金色），双击选中一整列，右键清空。回车出牌，空格不出，Esc 取消选择。选几张牌再点“理牌”，把它们放成一列；“一键理牌”自动把炸弹、同花顺、顺子、三带二等组合各放一列，“恢复”回到按点数排列；“同花顺 ♠♥♣♦”一键选出该花色的同花顺（含逢人配，再点换下一个）；“记牌器”显示其他三家还剩的各点数张数。'),
      h('div', { class: 'actions' }, h('button', { class: 'btn primary', onclick: () => m.close() }, '知道了')));
    const m = modal(box, { onClose: () => {} });
  }

  function leave() {
    const s = S.s;
    let sub = '之后可以从首页回到这张牌桌。';
    if (s && you() >= 0 && s.status === 'playing') {
      const others = s.seats.some((st, i) => i !== you() && st.kind === 'human' && st.connected);
      sub = others ? '其他玩家会接着打，轮到你时由 AI 替你出牌（托管）。之后从首页“回到座位”接着打。'
        : '牌局会暂停，之后从首页“回到座位”，从原处接着打。';
    } else if (s && you() < 0 && !s.seats.some((st) => st.kind === 'human')) {
      sub = '没人观战时这桌会暂停，之后从首页“观战”接着看。';
    }
    const box = h('div', {}, h('h2', {}, '离开牌桌？'), h('div', { class: 'sub' }, sub));
    let m;
    box.appendChild(h('div', { class: 'actions' },
      h('button', { class: 'btn', onclick: () => m.close() }, '留下'),
      h('button', { class: 'btn primary', onclick: () => { m.close(); location.hash = '#/'; } }, '离开'),
      canEnd() ? h('button', { class: 'btn danger', onclick: () => { m.close(); endTable(); } }, '结束牌局') : null));
    m = modal(box, { onClose: () => {} });
  }

  // the host, or the device that opened this table, can end it for good
  function canEnd() { return !!S.canManage; }
  function loadPerm() {
    api.get('/api/room' + (roomId ? '?id=' + encodeURIComponent(roomId) : '')).then((r) => {
      S.canManage = !!r.can_manage;
      if (S.s) render();
    }).catch(() => {});
  }
  function endTable() {
    const s = S.s;
    const others = s && s.seats.some((st, i) => i !== you() && st.kind === 'human' && st.taken);
    const box = h('div', {}, h('h2', {}, '结束牌局？'),
      h('div', { class: 'sub' }, '这张牌桌会关闭，进度不再保留' + (others ? '，桌上其他玩家也会回到首页。' : '。')));
    let m;
    box.appendChild(h('div', { class: 'actions' },
      h('button', { class: 'btn', onclick: () => { m.close(); if (fromResult) showResult(); } }, '取消'),
      h('button', { class: 'btn danger', onclick: () => { m.close(); S.ending = true; send({ action: 'end' }); } }, '结束牌局')));
    const fromResult = !!(S.modal && s && s.phase >= 2);
    if (S.modal) { S.modal.close(); S.modal = null; }
    m = modal(box, { onClose: () => {} });
  }

  // tell the host the real usable screen size of this device (phones differ; this is what the
  // layout has to fit) - at most once a few seconds after it settles
  let sizeTimer = 0;
  const reportSize = () => {
    clearTimeout(sizeTimer);
    sizeTimer = setTimeout(() => {
      api.post('/api/client', {
        w: window.innerWidth, h: window.innerHeight, sw: screen.width, sh: screen.height,
        dpr: window.devicePixelRatio || 1, fullscreen: !!(document.fullscreenElement || document.webkitFullscreenElement),
        ua: navigator.userAgent.slice(0, 200),
      }).catch(() => {});
    }, 2500);
  };
  reportSize();
  const onResize = () => { reportSize(); if (S.s && S.s.status === 'playing') { renderActions(); renderHand(); placeZones(); } };
  window.addEventListener('resize', onResize);
  const onKey = (e) => {
    if (!myTurn() || e.target.tagName === 'INPUT') return;
    if (e.key === 'Enter') doPlay();
    else if (e.key === ' ' && !S.s.is_leading) { e.preventDefault(); doPass(); }
    else if (e.key === 'Escape') { S.sel.clear(); renderHand(); renderActions(); }
  };
  window.addEventListener('keydown', onKey);
  // a phone that goes to the home screen or locks keeps the connection alive for a while; give
  // the seat up after a few seconds so that the table pauses (or the AI plays for you) instead of
  // the turn clock running out, and reconnect when the page is shown again
  let hideTimer = 0;
  const goAway = () => { if (you() >= 0 && S.ws && !S.away) { S.away = true; try { S.ws.close(); } catch {} } };
  const onVis = () => {
    clearTimeout(hideTimer);
    if (document.hidden) hideTimer = setTimeout(() => { if (document.hidden) goAway(); }, 3000);
    else if (S.away) { S.away = false; connect(); }
  };
  const onHide = () => goAway();
  // the Android app tells the page when it goes to the background or comes back (its WebView
  // does not always report it as a hidden page)
  const onApp = (e) => {
    clearTimeout(hideTimer);
    if (e.detail && e.detail.visible === false) goAway();
    else if (S.away) { S.away = false; connect(); }
  };
  document.addEventListener('visibilitychange', onVis);
  window.addEventListener('pagehide', onHide);
  window.addEventListener('gd-app-visibility', onApp);
  loadPerm();
  connect();

  return () => {
    S.closed = true;
    cancelAnimationFrame(S.raf);
    clearTimeout(S.heldTimer);
    window.removeEventListener('resize', onResize);
    window.removeEventListener('keydown', onKey);
    document.removeEventListener('visibilitychange', onVis);
    window.removeEventListener('pagehide', onHide);
    window.removeEventListener('gd-app-visibility', onApp);
    clearTimeout(hideTimer);
    clearTimeout(sizeTimer);
    if (S.modal) S.modal.close();
    try { S.ws && S.ws.close(); } catch {}
    document.querySelectorAll('.modal-back, .flycard, .boom').forEach((x) => x.remove());
  };
}

void cardName; void rankOf;
