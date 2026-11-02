import { cardEl } from './cards.js';
import { sound } from './sound.js';
import { mountTable } from './table.js';
import { mountTrain } from './train.js';

import { api, toast, h, seatStore, nameStore, modal } from './util.js';

const SEAT_POS = ['下（你）', '右（下家）', '上（对家）', '左（上家）'];
const app = document.getElementById('app');
let cleanup = null;
document.addEventListener('pointerdown', () => sound.unlock(), { once: true });

// ---------------------------------------------------------------- home
async function viewHome() {
  const fan = h('div', { class: 'fan', 'aria-hidden': 'true' });
  const fanCards = [1, 52, 52, 53, 53];
  fanCards.forEach((t, i) => {
    const c = cardEl(t, { level: 0 });
    const ang = (i - 2) * 13;
    c.style.transform = `translateX(-50%) rotate(${ang}deg)`;
    c.style.animationDelay = `${0.1 + i * 0.06}s`;
    fan.appendChild(c);
  });
  const joinSlot = h('div');
  const foot = h('div', { class: 'home-foot' });
  app.replaceChildren(h('div', { class: 'home' }, h('div', { class: 'home-inner' },
    fan,
    h('h1', { class: 'brand' }, '掼蛋'),
    h('p', { class: 'tagline' }, '纯自对弈强化学习训练的掼蛋 AI，同一 WiFi 下多人联机'),
    h('div', { class: 'home-actions' },
      h('a', { class: 'btn primary', href: '#/play' }, '开一桌'),
      h('a', { class: 'btn', href: '#/train' }, '训练中心'),
    ),
    joinSlot,
    foot,
  )));
  try {
    const [info, room] = await Promise.all([api.get('/api/info'), api.get('/api/room')]);
    const mine = seatStore.get();
    if (room.exists) {
      const open = room.seats.filter((s) => s.kind === 'human' && !s.taken).length;
      const isMine = mine && mine.room === room.id;
      const modeName = room.mode === 'teach' ? '教学模式' : '实战模式';
      const lenName = room.length === 'quick' ? '快速对局' : '竞技模式';
      joinSlot.appendChild(h('div', { class: 'join-strip' },
        h('span', {}, `当前牌桌：${modeName}，${lenName}，`, open ? `还差 ${open} 位玩家` : (room.status === 'playing' ? '对局进行中' : '等待开始')),
        isMine ? h('a', { class: 'btn small primary', href: '#/table' }, '回到牌桌')
          : open ? h('a', { class: 'btn small primary', href: '#/join' }, '加入')
            : h('a', { class: 'btn small', href: '#/join' }, '观战'),
      ));
    }
    const dev = info.device || {};
    foot.append(...[
      h('span', {}, dev.cuda ? `GPU：${dev.gpu}` : (dev.torch ? 'CPU 模式' : '未安装 PyTorch')),
      h('span', {}, info.default_model ? `模型：${info.default_model.split(/[\\/]/).slice(-2).join('/')}` : '还没有可用模型'),
      info.lan_urls && info.lan_urls[0] ? h('span', {}, `局域网地址：${info.lan_urls[0]}`) : null,
      h('span', {}, `v${info.version}`),
    ].filter(Boolean));
  } catch (e) {
    foot.append(h('span', {}, '无法连接服务：' + e.message));
  }
}

// ---------------------------------------------------------------- setup
async function viewSetup() {
  const cfg = { think: 'deep', scoring: 'none', mode: 'teach', length: 'full', seats: ['human', 'ai', 'ai', 'ai'], model: '', seatModels: ['', '', '', ''], ai_speed: 1, name: nameStore.get() || '房主' };
  let models = [];
  let room = null;
  try { [models, room] = await Promise.all([api.get('/api/models'), api.get('/api/room')]); } catch (e) { toast(e.message, 'err'); }
  if (models.length) cfg.model = models[0].id;

  const root = h('div', { class: 'page' });
  app.replaceChildren(root);

  function modelSelect(value, onchange, small) {
    const sel = h('select', { onchange: (e) => onchange(e.target.value), 'aria-label': '选择模型' });
    if (small) sel.appendChild(h('option', { value: '' }, '同默认模型'));
    for (const m of models) {
      const o = h('option', { value: m.id }, `${m.source}：${m.id.split('/').slice(-1)[0]}`);
      if (m.id === value) o.selected = true;
      sel.appendChild(o);
    }
    return sel;
  }

  function render() {
    const teach = cfg.mode === 'teach';
    if (teach) cfg.seats = ['human', 'ai', 'ai', 'ai'];
    const choice = (on, title, desc, click) => h('button', { class: 'choice' + (on ? ' on' : ''), onclick: click, type: 'button' }, h('b', {}, title), h('span', {}, desc));
    const mini = h('div', { class: 'mini-table' }, h('div', { class: 'center' }, '牌桌'));
    for (let i = 0; i < 4; i++) {
      const isHuman = cfg.seats[i] === 'human';
      const chip = h('div', { class: `seat-chip sc-${i} team${i % 2}` },
        h('div', { class: 'who' }, h('b', {}, SEAT_POS[i]), i % 2 === 0 ? '  我方' : '  对方'),
      );
      if (teach) {
        chip.appendChild(h('div', {}, i === 0 ? '你' : 'AI'));
      } else {
        const seg = h('div', { class: 'seg' },
          h('button', { class: isHuman ? 'on' : '', type: 'button', onclick: () => { cfg.seats[i] = 'human'; render(); } }, '玩家'),
          h('button', { class: !isHuman ? 'on' : '', type: 'button', onclick: () => { cfg.seats[i] = 'ai'; render(); } }, 'AI'));
        chip.appendChild(seg);
        if (!isHuman && models.length > 1) chip.appendChild(modelSelect(cfg.seatModels[i], (v) => { cfg.seatModels[i] = v; }, true));
      }
      mini.appendChild(chip);
    }
    const humans = cfg.seats.filter((s) => s === 'human').length;
    const nameInput = h('input', { type: 'text', value: cfg.name, maxlength: 10, oninput: (e) => { cfg.name = e.target.value; } });
    const speed = h('input', { type: 'range', min: 0.5, max: 3, step: 0.25, value: cfg.ai_speed, oninput: (e) => { cfg.ai_speed = +e.target.value; spd.textContent = `${cfg.ai_speed}×`; } });
    const spd = h('span', { class: 'dim' }, `${cfg.ai_speed}×`);
    const warn = room && room.exists && room.status === 'playing' ? h('p', { class: 'dim' }, '提示：当前已有一桌在进行，开新桌会结束它。') : null;
    root.replaceChildren(
      h('div', { class: 'nav' }, h('a', { class: 'btn small', href: '#/' }, '返回'), h('span', { class: 'title' }, '开一桌'), h('span', { class: 'spacer' })),
      h('div', { class: 'setup' },
        h('div', { class: 'opts' },
          h('div', { class: 'panel' }, h('h2', {}, '模式'),
            h('div', { class: 'choice-row' },
              choice(teach, '教学模式', '你和三个 AI 对局。不限时，可以让 AI 给出出牌建议、点评你的出牌，也可以明牌查看所有人的手牌。', () => { cfg.mode = 'teach'; render(); }),
              choice(!teach, '实战模式', '与 AI 或局域网玩家对局。每步限时 20 秒，超时随机出牌；看不到别人的牌，自己出完后可以看队友的牌。', () => { cfg.mode = 'real'; render(); }),
            )),
          h('div', { class: 'panel' }, h('h2', {}, '对局长度'),
            h('div', { class: 'choice-row' },
              choice(cfg.length === 'quick', '快速对局', '每局随机抽一个级牌，单局结算积分，不进贡。适合随时开打。', () => { cfg.length = 'quick'; render(); }),
              choice(cfg.length === 'full', '竞技模式', '从 2 打到 A：双上升三级、进贡还贡、抗贡、接风，过 A 才算赢。', () => { cfg.length = 'full'; render(); }),
            )),
          cfg.length === 'quick' ? h('div', { class: 'panel' }, h('h2', {}, '计分'),
            h('div', { class: 'choice-row', style: { gridTemplateColumns: '1fr 1fr 1fr' } },
              choice(cfg.scoring === 'none', '只算升级', '按升级数结算：双上 3、一三游 2、一四游 1。', () => { cfg.scoring = 'none'; render(); }),
              choice(cfg.scoring === 'standard', '标准倍数', '底分同升级数；6 张以上炸弹、同花顺、天王炸每出一个翻一倍，封顶 16 倍。', () => { cfg.scoring = 'standard'; render(); }),
              choice(cfg.scoring === 'all', '全炸翻倍', '每出一个炸弹（含同花顺、天王炸）翻一倍，封顶 64 倍。', () => { cfg.scoring = 'all'; render(); }),
            ),
            cfg.scoring !== 'none' ? h('p', { class: 'dim', style: { fontSize: '12.5px', margin: '10px 0 0' } },
              '倍数只影响每局结算的积分，AI 的打法不变，仍以赢下本局、多升级为目标。') : null) : null,
          h('div', { class: 'panel' }, h('h2', {}, 'AI 思考方式'),
            h('div', { class: 'choice-row' },
              choice(cfg.think === 'deep', '深度思考', '推荐。网络先挑出最好的几手，再在随机推测的对手手牌里各模拟打下去比较结果，更强，每步多想不到一秒。', () => { cfg.think = 'deep'; render(); }),
              choice(cfg.think === 'fast', '快速', '只用神经网络直接出牌，几乎不用等待，水平稍弱。', () => { cfg.think = 'fast'; render(); }),
            )),
          h('div', { class: 'panel' }, h('h2', {}, 'AI 与玩家'),
            h('div', { class: 'field' }, h('label', {}, 'AI 模型'), models.length ? modelSelect(cfg.model, (v) => { cfg.model = v; }) : h('div', { class: 'dim' }, '还没有模型：请先到训练中心训练，或下载 Release 中的模型。')),
            h('div', { class: 'grid2', style: { marginTop: '12px' } },
              h('div', { class: 'field' }, h('label', {}, '你的昵称'), nameInput),
              h('div', { class: 'field' }, h('label', {}, h('span', {}, 'AI 出牌速度 '), spd), speed)),
          ),
        ),
        h('div', { class: 'panel' }, h('h2', {}, '座位'),
          h('p', { class: 'muted', style: { margin: '0 0 6px', fontSize: '13px' } }, teach ? '教学模式固定为你与三个 AI。' : '对家是你的队友。设为“玩家”的座位，可由同一 WiFi 下的手机或电脑扫码加入。'),
          mini,
          h('p', { class: 'dim', style: { fontSize: '13px' } }, humans === 0 ? '四个座位都是 AI：你将以观战者身份旁观（可看到所有手牌）。' : humans > 1 ? `需要另外 ${humans - 1} 位玩家通过局域网加入后才会开始。` : ''),
          warn,
          h('div', { style: { display: 'flex', justifyContent: 'flex-end', marginTop: '10px' } },
            h('button', { class: 'btn primary', disabled: !models.length, onclick: start }, humans === 0 ? '开始观战' : '开桌')),
        ),
      ),
    );
  }

  async function start() {
    nameStore.set(cfg.name);
    try {
      const body = {
        mode: cfg.mode, length: cfg.length, think: cfg.think, scoring: cfg.scoring, model: cfg.model, ai_speed: cfg.ai_speed, name: cfg.name,
        seats: cfg.seats.map((k, i) => ({ kind: k, model: cfg.seatModels[i] || cfg.model })),
      };
      const r = await api.post('/api/room/create', body);
      if (r.token) seatStore.set({ room: r.room, seat: r.seat, token: r.token });
      else seatStore.set({ room: r.room, seat: -1, token: null });
      location.hash = '#/table';
    } catch (e) {
      toast(e.message, 'err');
    }
  }
  render();
}

// ---------------------------------------------------------------- join (LAN players)
async function viewJoin() {
  const root = h('div', { class: 'page' });
  app.replaceChildren(root);
  let room;
  async function load() {
    try { room = await api.get('/api/room'); } catch (e) { room = { exists: false }; }
    render();
  }
  function render() {
    const nav = h('div', { class: 'nav' }, h('a', { class: 'btn small', href: '#/' }, '返回'), h('span', { class: 'title' }, '加入牌桌'));
    if (!room.exists) {
      root.replaceChildren(nav, h('div', { class: 'panel' }, h('h2', {}, '主机还没有开桌'),
        h('p', { class: 'muted' }, '请在主机上点“开一桌”，把座位设为“玩家”，然后点下面的按钮刷新。'),
        h('button', { class: 'btn', onclick: load }, '刷新')));
      return;
    }
    const mine = seatStore.get();
    const nameInput = h('input', { type: 'text', value: nameStore.get(), maxlength: 10, placeholder: '你的昵称' });
    const list = h('div', { class: 'seat-list' });
    room.seats.forEach((s, i) => {
      let action;
      if (mine && mine.room === room.id && mine.seat === i) action = h('a', { class: 'btn small primary', href: '#/table' }, '回到座位');
      else if (s.kind === 'human' && !s.taken) action = h('button', { class: 'btn small primary', onclick: () => join(i, nameInput.value) }, '坐这里');
      else action = h('span', { class: 'dim' }, s.kind === 'ai' ? 'AI' : '已有人');
      list.appendChild(h('div', { class: 'seat-row' },
        h('span', { class: 'pos' }, ['南', '东', '北', '西'][i]),
        h('span', { class: 'nm' }, s.kind === 'ai' ? `${s.name}` : (s.taken ? s.name : '空位'), h('span', { class: 'dim' }, i % 2 === 0 ? '  （与南家同队）' : '  （与东家同队）')),
        s.kind === 'human' ? h('span', { class: 'dot' + (s.connected ? ' on' : '') }) : null,
        action));
    });
    root.replaceChildren(nav, h('div', { class: 'panel' },
      h('h2', {}, `${room.mode === 'teach' ? '教学' : '实战'}模式，${room.length === 'quick' ? '快速对局' : '竞技模式（打到 A）'}`),
      h('div', { class: 'field', style: { maxWidth: '280px' } }, h('label', {}, '昵称'), nameInput),
      list,
      h('div', { style: { marginTop: '14px', display: 'flex', gap: '10px' } },
        h('button', { class: 'btn', onclick: load }, '刷新'),
        h('button', { class: 'btn', onclick: () => { seatStore.set({ room: room.id, seat: -1, token: null }); location.hash = '#/table'; } }, '观战')),
    ));
  }
  async function join(seat, name) {
    nameStore.set(name);
    try {
      const r = await api.post('/api/room/join', { seat, name });
      seatStore.set({ room: r.room, seat: r.seat, token: r.token });
      location.hash = '#/table';
    } catch (e) { toast(e.message, 'err'); load(); }
  }
  load();
}

// ---------------------------------------------------------------- router
function route() {
  if (cleanup) { try { cleanup(); } catch {} cleanup = null; }
  document.body.classList.remove('in-game');
  const hash = location.hash || '#/';
  if (hash.startsWith('#/play')) viewSetup();
  else if (hash.startsWith('#/join')) viewJoin();
  else if (hash.startsWith('#/train')) cleanup = mountTrain(app);
  else if (hash.startsWith('#/table')) { document.body.classList.add('in-game'); cleanup = mountTable(app); }
  else viewHome();
}
window.addEventListener('hashchange', route);
route();
