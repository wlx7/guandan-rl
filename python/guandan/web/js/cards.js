// Card helpers. Card type t: rank*4+suit for 0..51 (rank 0="2" .. 12="A"; suit 0♠ 1♥ 2♣ 3♦), 52 small joker, 53 big joker.
import { T, isEn } from './i18n.js';

export const RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A'];
export const SUITS = ['♠', '♥', '♣', '♦'];
export const rankOf = (t) => (t >= 52 ? t - 52 + 13 : t >> 2);
export const suitOf = (t) => (t >= 52 ? 4 : t & 3);
export const strength = (r, level) => (r >= 13 ? r + 1 : r === level ? 13 : r);
export const rankName = (r) => (r === 13 ? T('小王') : r === 14 ? T('大王') : RANKS[r]);
export const levelName = (lv) => RANKS[lv];

// Suit shapes drawn as SVG: phones otherwise render ♠♥♣♦ with an emoji font of a different size,
// which overflows the visible strip of stacked cards.
const SUIT_PATHS = [
  // spade
  'M50 4C80 30 96 44 96 60C96 74 85 82 73 82C64 82 57 77 53 70C54 80 58 88 66 96H34C42 88 46 80 47 70C43 77 36 82 27 82C15 82 4 74 4 60C4 44 20 30 50 4Z',
  // heart
  'M50 92C20 68 4 50 4 31C4 15 16 5 29 5C39 5 46 11 50 19C54 11 61 5 71 5C84 5 96 15 96 31C96 50 80 68 50 92Z',
  // club
  'M50 6A21 21 0 0 1 69 37A21 21 0 1 1 57 72C58 82 62 89 68 96H32C38 89 42 82 43 72A21 21 0 1 1 31 37A21 21 0 0 1 50 6Z',
  // diamond
  'M50 2L90 50L50 98L10 50Z',
];
export function suitSvg(s, cls = '') {
  return `<svg class="sv${cls ? ' ' + cls : ''}" viewBox="0 0 100 100" aria-hidden="true"><path d="${SUIT_PATHS[s]}"/></svg>`;
}

export function cardName(t) {
  const r = rankOf(t);
  if (r >= 13) return rankName(r);
  return SUITS[suitOf(t)] + RANKS[r];
}

export function cardEl(t, opts = {}) {
  const { level = 0, size = '', back = false } = opts;
  const el = document.createElement('div');
  el.className = 'card' + (size ? ' ' + size : '');
  el.dataset.t = t;
  if (back) {
    el.classList.add('back');
    return el;
  }
  const r = rankOf(t);
  const s = suitOf(t);
  if (r >= 13) {
    el.classList.add('joker', r === 14 ? 'big' : 'small');
    el.innerHTML = isEn
      ? `<div class="idx">${r === 14 ? 'B<br>J' : 'S<br>J'}</div><div class="pip">★</div>`
      : `<div class="idx">${r === 14 ? '大<br>王' : '小<br>王'}</div><div class="pip">王</div>`;
    el.title = rankName(r);
    return el;
  }
  if (s === 1 || s === 3) el.classList.add('red');
  if (r === level) el.classList.add('lv');
  const isWild = r === level && s === 1;
  if (isWild) el.classList.add('wild');
  const rn = RANKS[r];
  el.innerHTML = `<div class="idx"><span class="r${rn === '10' ? ' ten' : ''}">${rn}</span><span class="s">${suitSvg(s)}</span></div><div class="pip">${suitSvg(s)}</div>${isWild ? `<span class="tagw">${isEn ? 'W' : '配'}</span>` : ''}`;
  el.title = cardName(t) + (isWild ? T('（逢人配）') : '');
  return el;
}

// Sort key: strength desc, wild first within level, then suit.
export function sortCards(cards, level, desc = true) {
  const wild = level * 4 + 1;
  const key = (t) => {
    const r = rankOf(t);
    let k = strength(r, level) * 10 + (t === wild ? 9 : 4 - suitOf(t));
    return desc ? -k : k;
  };
  return [...cards].sort((a, b) => key(a) - key(b));
}

// Columns of same rank (strength order). Wild cards get their own column.
export function columns(cards, level, desc = true) {
  const wild = level * 4 + 1;
  const sorted = sortCards(cards, level, desc);
  const cols = [];
  let cur = null;
  let curKey = null;
  for (const t of sorted) {
    const k = t === wild ? 'W' : rankOf(t);
    if (k !== curKey) {
      cur = [];
      cols.push(cur);
      curKey = k;
    }
    cur.push(t);
  }
  return cols;
}

// Give every physical card a stable id: "t:k" where k counts duplicates of type t.
export function withIds(cards) {
  const seen = {};
  return cards.map((t) => {
    seen[t] = (seen[t] || 0) + 1;
    return { t, id: `${t}:${seen[t]}` };
  });
}

export function miniRow(cards, level, size = 'm', max = 30) {
  const row = document.createElement('div');
  row.className = 'mini-hand';
  for (const t of sortCards(cards, level).slice(0, max)) row.appendChild(cardEl(t, { level, size }));
  return row;
}

// ---------------------------------------------------------------- 理牌 (one-click arrange)
// Natural-order windows for straights: A-2-3-4-5 up to 10-J-Q-K-A (rank indices, '2' = 0, 'A' = 12).
const WINDOWS = [[12, 0, 1, 2, 3], ...Array.from({ length: 9 }, (_, i) => [i, i + 1, i + 2, i + 3, i + 4])];

// Group a hand into playable combinations, one column each: wild cards, 天王炸, bombs, 同花顺,
// 钢板, 三连对, 顺子, 三带二, 三张, 对子, 单张. Greedy and deterministic; it only rearranges
// the display, the player still chooses what to play.
export function arrangeHand(cards, level) {
  const wild = level * 4 + 1;
  const cnt = new Array(54).fill(0);
  let wilds = 0;
  for (const t of cards) { if (t === wild) wilds++; else cnt[t]++; }
  const rankCount = (r) => cnt[r * 4] + cnt[r * 4 + 1] + cnt[r * 4 + 2] + cnt[r * 4 + 3];
  const take = (t) => { cnt[t]--; return t; };
  const takeRank = (r, n) => {  // n cards of rank r, preferring suits that are not needed elsewhere
    const out = [];
    for (let k = 0; k < n; k++) {
      let best = -1;
      for (let s = 0; s < 4; s++) if (cnt[r * 4 + s] > 0 && (best < 0 || cnt[r * 4 + s] > cnt[r * 4 + best])) best = s;
      out.push(take(r * 4 + best));
    }
    return out;
  };
  const str = (r) => strength(r, level);
  const groups = [];
  const add = (kind, cs, order) => groups.push({ kind, cards: cs, order });
  if (wilds) add('配', Array(wilds).fill(wild), 0);
  if (cnt[52] === 2 && cnt[53] === 2) add('天王炸', [take(53), take(53), take(52), take(52)], 1);
  // bombs of 6+ outrank a straight flush: keep them whole
  for (let r = 0; r < 13; r++) if (rankCount(r) >= 6) add('炸弹', takeRank(r, rankCount(r)), 2);
  // straight flushes: cheapest first (the fewest pairs / triples / bombs broken; taking one card
  // from a 5-card bomb still leaves a bomb)
  const cost = (c) => (c >= 5 ? 0 : c === 4 ? 3 : c === 3 ? 2 : c === 2 ? 1 : 0);
  for (;;) {
    let best = null;
    for (let s = 0; s < 4; s++) {
      for (const w of WINDOWS) {
        if (!w.every((r) => cnt[r * 4 + s] > 0)) continue;
        const pen = w.reduce((a, r) => a + cost(rankCount(r)), 0);
        if (!best || pen < best.pen || (pen === best.pen && w[4] > best.w[4])) best = { s, w, pen };
      }
    }
    if (!best || best.pen > 4) break;
    add('同花顺', best.w.map((r) => take(r * 4 + best.s)), 3);
  }
  for (let r = 0; r < 13; r++) if (rankCount(r) >= 4) add('炸弹', takeRank(r, rankCount(r)), 2);
  // plain straights, only from cards that are not part of a pair or triple (at most one broken pair)
  for (;;) {
    let best = null;
    for (const w of WINDOWS) {
      if (!w.every((r) => rankCount(r) > 0)) continue;
      const pen = w.reduce((a, r) => a + (rankCount(r) === 1 ? 0 : rankCount(r) === 2 ? 1 : 3), 0);
      if (!best || pen < best.pen || (pen === best.pen && w[4] > best.w[4])) best = { w, pen };
    }
    if (!best || best.pen > 1) break;
    add('顺子', best.w.map((r) => takeRank(r, 1)[0]), 6);
  }
  const ranksWith = (n) => Array.from({ length: 13 }, (_, r) => r).filter((r) => rankCount(r) === n);
  // 钢板: two triples of consecutive natural ranks (A can follow K or precede 2)
  const next = (r) => (r === 12 ? 0 : r + 1);
  for (const r of ranksWith(3)) {
    const r2 = next(r);
    if (rankCount(r) === 3 && rankCount(r2) === 3) add('钢板', [...takeRank(r, 3), ...takeRank(r2, 3)], 4);
  }
  // 三连对: three pairs of consecutive natural ranks (A-2-3 up to Q-K-A; K-A-2 does not wrap)
  for (const r of ranksWith(2)) {
    if (r === 11) continue;
    const r2 = next(r), r3 = next(r2);
    if (rankCount(r) === 2 && rankCount(r2) === 2 && rankCount(r3) === 2) {
      add('三连对', [...takeRank(r, 2), ...takeRank(r2, 2), ...takeRank(r3, 2)], 5);
    }
  }
  // 三带二: each triple takes the weakest remaining pair
  const triples = ranksWith(3).sort((a, b) => str(b) - str(a));
  const pairs = ranksWith(2).sort((a, b) => str(a) - str(b));
  for (const r of triples) {
    const p = pairs.shift();
    if (p !== undefined) add('三带二', [...takeRank(r, 3), ...takeRank(p, 2)], 7);
    else add('三张', takeRank(r, 3), 8);
  }
  for (const r of ranksWith(2)) add('对子', takeRank(r, 2), 9);
  for (const t of [53, 52]) if (cnt[t] === 2) add('对子', [take(t), take(t)], 9);
  for (let r = 0; r < 13; r++) while (rankCount(r) > 0) add('单张', takeRank(r, 1), 10);
  for (const t of [53, 52]) while (cnt[t] > 0) add('单张', [take(t)], 10);
  // display order: by kind, then strongest first
  const top = (g) => Math.max(...g.cards.map((t) => (t >= 52 ? t - 52 + 14 : str(rankOf(t)))));
  groups.sort((a, b) => a.order - b.order || b.cards.length - a.cards.length || top(b) - top(a));
  return groups;
}

// Cards of every rank not yet seen by this player (not in their hand and not played this round).
// Returns [{label, left, total, level}] from 大王 down to 2, the level rank right after the jokers.
export function cardCounter(hand, played, level) {
  const seen = new Array(15).fill(0);
  for (const t of hand) seen[rankOf(t)]++;
  for (const list of played) for (const t of list) seen[rankOf(t)]++;
  const out = [{ label: isEn ? 'BJ' : '大', r: 14 }, { label: isEn ? 'SJ' : '小', r: 13 }, { label: RANKS[level], r: level, level: true }];
  for (let r = 12; r >= 0; r--) if (r !== level) out.push({ label: RANKS[r], r });
  return out.map((c) => {
    const total = c.r >= 13 ? 2 : 8;
    return { ...c, total, left: Math.max(0, total - seen[c.r]) };
  });
}

// Straight flushes of one suit that the hand can make (wild cards fill gaps), strongest first.
// Each is a list of card types to select.
export function straightFlushes(cards, level, suit) {
  const wild = level * 4 + 1;
  const have = new Set(cards.filter((t) => t !== wild && t < 52));
  const wilds = cards.filter((t) => t === wild).length;
  const out = [];
  for (let i = WINDOWS.length - 1; i >= 0; i--) {
    const w = WINDOWS[i];
    const types = w.map((r) => r * 4 + suit);
    const missing = types.filter((t) => !have.has(t));
    if (missing.length > wilds) continue;
    // a straight flush made entirely of the wild cards' own suit/rank is still fine
    out.push(types.map((t) => (have.has(t) ? t : wild)));
  }
  return out;
}
