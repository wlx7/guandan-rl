// Card helpers. Card type t: rank*4+suit for 0..51 (rank 0="2" .. 12="A"; suit 0♠ 1♥ 2♣ 3♦), 52 small joker, 53 big joker.
export const RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A'];
export const SUITS = ['♠', '♥', '♣', '♦'];
export const rankOf = (t) => (t >= 52 ? t - 52 + 13 : t >> 2);
export const suitOf = (t) => (t >= 52 ? 4 : t & 3);
export const strength = (r, level) => (r >= 13 ? r + 1 : r === level ? 13 : r);
export const rankName = (r) => (r === 13 ? '小王' : r === 14 ? '大王' : RANKS[r]);
export const levelName = (lv) => RANKS[lv];

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
    el.innerHTML = `<div class="idx">${r === 14 ? '大<br>王' : '小<br>王'}</div><div class="pip">${r === 14 ? '王' : '王'}</div>`;
    el.title = rankName(r);
    return el;
  }
  if (s === 1 || s === 3) el.classList.add('red');
  if (r === level) el.classList.add('lv');
  const isWild = r === level && s === 1;
  if (isWild) el.classList.add('wild');
  const rn = RANKS[r];
  el.innerHTML = `<div class="idx"><span class="r${rn === '10' ? ' ten' : ''}">${rn}</span><span class="s">${SUITS[s]}</span></div><div class="pip">${SUITS[s]}</div>${isWild ? '<span class="tagw">配</span>' : ''}`;
  el.title = cardName(t) + (isWild ? '（逢人配）' : '');
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
