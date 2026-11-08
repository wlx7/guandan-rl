import { cardEl } from './cards.js';
import { sound } from './sound.js';
import { mountTable } from './table.js';
import { mountTrain } from './train.js';

import { api, toast, h, seatStore, nameStore, modal, fillModelSelect, getMe, adminToken, host } from './util.js';
import { T, tr, lang, setLang } from './i18n.js';

const SEAT_POS = ['下（你）', '右（下家）', '上（对家）', '左（上家）'];
const app = document.getElementById('app');
let cleanup = null;
document.title = T('掼蛋 AI');
document.addEventListener('pointerdown', () => sound.unlock(), { once: true });

// "this computer" / "this phone": the device that runs the server
const dev = () => T(host.device);

// ---------------------------------------------------------------- home
function langSwitch() {
  const b = (code, text) => h('button', { class: lang === code ? 'on' : '', type: 'button', 'aria-pressed': lang === code ? 'true' : 'false',
    onclick: () => { if (lang !== code) setLang(code); } }, text);
  return h('div', { class: 'seg lang-switch', role: 'group', 'aria-label': 'Language / 语言' }, b('zh', '中文'), b('en', 'English'));
}

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
  const me = await getMe();
  const canCreate = me.admin || me.guest_create;
  app.replaceChildren(h('div', { class: 'home' }, langSwitch(), h('div', { class: 'home-inner' },
    fan,
    h('h1', { class: 'brand' }, T('掼蛋')),
    h('p', { class: 'tagline' }, T('纯自对弈强化学习训练的掼蛋 AI，同一 WiFi 下多人联机')),
    h('div', { class: 'home-actions' },
      canCreate ? h('a', { class: 'btn primary', href: '#/play' }, T('开一桌'))
        : h('button', { class: 'btn primary', title: T('房主没有开放其他设备开桌'), onclick: () => toast(T('房主没有开放其他设备开桌，请让房主开桌后扫码加入'), 'err') }, T('开一桌')),
      me.train !== false && (me.admin || me.guest_train) ? h('a', { class: 'btn', href: '#/train' }, T('训练中心')) : null,
    ),
    joinSlot,
    foot,
    h('div', { class: 'home-perm' }, ...permLinks(me)),
  )));
  let timer = 0;
  let stopped = false;
  window.addEventListener('hashchange', () => { stopped = true; clearTimeout(timer); }, { once: true });
  async function refresh() {
    if (stopped) return;
    try {
      const { rooms, max } = await api.get('/api/rooms');
      seatStore.keep(rooms.map((r) => r.id));
      joinSlot.replaceChildren(...(rooms.length ? [tableList(rooms, max, refresh)] : []));
    } catch {}
    if (!stopped) timer = setTimeout(refresh, 3000);
  }
  refresh();
  try {
    const info = await api.get('/api/info');
    const d = info.device || {};
    foot.append(...[
      h('span', {}, info.app === 'android' ? T('安卓版（只能对战）') : info.portable ? T('免安装版（只能对战）') : d.cuda ? `GPU${T('：')}${d.gpu}` : (d.torch ? T('CPU 模式') : T('未安装 PyTorch（只能对战）'))),
      h('span', {}, info.default_model ? T('模型：{m}', { m: info.default_model.split(/[\\/]/).slice(-2).join('/') }) : T('还没有可用模型')),
      info.lan_urls && info.lan_urls[0] ? h('span', {}, T('局域网地址：{u}', { u: info.lan_urls[0] })) : null,
      h('span', {}, `v${info.version}`),
    ].filter(Boolean));
  } catch (e) {
    foot.append(h('span', {}, T('无法连接服务：') + tr(e.message)));
  }
}

// ---------------------------------------------------------------- permissions
// 房主 (the host computer, or a device logged in with the host password) manages everything;
// other devices manage only the tables they opened
function permLinks(me) {
  const link = (text, fn) => h('a', { href: '#', onclick: (e) => { e.preventDefault(); fn(); } }, text);
  if (me.admin) {
    return [h('span', {}, me.local ? T('你是房主（运行游戏的{d}）', { d: dev() }) : T('你已用房主密码登录')),
      link(T('权限设置'), () => permSettings(me)),
      me.local ? null : link(T('退出房主'), async () => { try { await api.post('/api/perm/logout'); } catch {} adminToken.clear(); route(); })].filter(Boolean);
  }
  return [h('span', {}, me.guest_create ? T('你是普通玩家：可以加入、观战，管理自己开的牌桌') : T('你是普通玩家：可以加入、观战')),
    me.pin_set ? link(T('房主登录'), permLogin) : null].filter(Boolean);
}

function permLogin() {
  let m;
  const input = h('input', { type: 'password', inputmode: 'numeric', autocomplete: 'off', placeholder: T('房主密码'), maxlength: 32 });
  const go = async () => {
    try {
      const r = await api.post('/api/perm/login', { pin: input.value });
      adminToken.set(r.token);
      m.close();
      toast(T('已用房主密码登录，这台设备现在可以管理所有牌桌'));
      route();
    } catch (e) { toast(tr(e.message), 'err'); }
  };
  input.addEventListener('keydown', (e) => { if (e.key === 'Enter') go(); });
  const box = h('div', {}, h('h2', {}, T('房主登录')),
    h('div', { class: 'sub' }, host.device === '电脑'
      ? T('输入在运行游戏的{d}上设置的房主密码，这台设备就能管理所有牌桌和训练中心。', { d: dev() })
      : T('输入在运行游戏的{d}上设置的房主密码，这台设备就能管理所有牌桌。', { d: dev() })),
    h('div', { class: 'field', style: { margin: '12px auto 0', maxWidth: '240px' } }, input),
    h('div', { class: 'actions' }, h('button', { class: 'btn', onclick: () => m.close() }, T('取消')), h('button', { class: 'btn primary', onclick: go }, T('登录'))));
  m = modal(box, { onClose: () => {} });
  setTimeout(() => input.focus(), 50);
}

function permSettings(me) {
  let m;
  const st = { guest_create: me.guest_create, guest_train: me.guest_train };
  const check = (key, text, sub) => {
    const cb = h('input', { type: 'checkbox', onchange: (e) => { st[key] = e.target.checked; } });
    cb.checked = !!st[key];
    return h('label', { class: 'perm-row' }, cb, h('span', {}, h('b', {}, text), h('span', { class: 'dim' }, sub)));
  };
  const pin = h('input', { type: 'password', inputmode: 'numeric', autocomplete: 'new-password', maxlength: 32,
    placeholder: me.pin_set ? T('已设置（留空不改）') : T('至少 4 位，不设就只有这台{d}是房主', { d: dev() }) });
  const save = async (body) => {
    try {
      await api.post('/api/perm/settings', body);
      m.close();
      toast(T('权限设置已保存'));
      route();
    } catch (e) { toast(tr(e.message), 'err'); }
  };
  const box = h('div', { style: { textAlign: 'left' } }, h('h2', { style: { textAlign: 'center' } }, T('权限设置')),
    h('div', { class: 'sub', style: { textAlign: 'center' } }, me.train === false
      ? T('运行游戏的这台{d}始终是房主，可以管理所有牌桌。', { d: dev() })
      : T('运行游戏的这台{d}始终是房主，可以管理所有牌桌、使用训练中心。', { d: dev() })),
    h('div', { class: 'perm-list' },
      check('guest_create', T('其他设备可以开桌'), T('手机等设备可以自己开一桌，并且只能管理（暂停、结束、关闭）自己开的牌桌')),
      me.train === false ? null : check('guest_train', T('其他设备可以使用训练中心'), T('关闭时只有房主能开始、停止训练和整理模型'))),
    h('div', { class: 'field', style: { marginTop: '12px' } }, h('label', {}, T('房主密码')), pin,
      h('span', { class: 'hint' }, T('其他设备在首页点“房主登录”输入它，就和这台{d}一样是房主。修改或清除密码会让已登录的设备全部退出。', { d: dev() }))),
    h('div', { class: 'actions' },
      me.pin_set ? h('button', { class: 'btn danger', onclick: () => save({ ...st, pin: '' }) }, T('清除密码')) : null,
      h('button', { class: 'btn', onclick: () => m.close() }, T('取消')),
      h('button', { class: 'btn primary', onclick: () => save(pin.value ? { ...st, pin: pin.value } : st) }, T('保存'))));
  m = modal(box, { onClose: () => {} });
}

const DIRS = ['南', '东', '北', '西'];
const LV = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A'];

function tableStatus(r) {
  if (r.status === 'waiting') return ['wait', T('等人加入（{a}/{b}）', { a: r.connected, b: r.humans })];
  if (r.pause_by) return ['pause', T('已暂停（{who}）', { who: tr(r.pause_by) })];
  if (r.paused) return ['pause', r.humans ? T('已暂停：玩家都离开了') : T('已暂停：没人观战')];
  return ['play', T('进行中')];
}

function tableList(rooms, max, refresh) {
  const box = h('div', { class: 'tables' },
    h('div', { class: 'tables-head' }, h('b', {}, T('牌桌（{n}/{max}）', { n: rooms.length, max })),
      h('span', { class: 'dim' }, T('离开牌桌就自动暂停，回来从原处接着打；可以同时开几桌'))));
  for (const r of rooms) {
    const mine = seatStore.get(r.id);
    const [cls, text] = tableStatus(r);
    const players = r.seats.map((st, i) => st.kind === 'ai' ? null
      : h('span', { class: 'pl' + (st.connected ? ' on' : '') }, `${T(DIRS[i])} ${st.taken ? tr(st.name) : T('空位')}`)).filter(Boolean);
    const ais = r.seats.filter((st) => st.kind === 'ai').length;
    const prog = r.status === 'waiting' ? T('还没开始')
      : r.length === 'quick' ? T('第 {n} 副 · 打 {lv} · 积分 {a} : {b}', { n: r.round_no, lv: LV[r.level], a: r.points[0], b: r.points[1] })
        : T('第 {m} 场第 {n} 局 · 南北 {a} / 东西 {b}', { m: r.match_no, n: r.round_no, a: LV[r.team_level[0]], b: LV[r.team_level[1]] });
    const free = r.seats.some((st) => st.kind === 'human' && !st.taken);
    const enter = mine && mine.token
      ? h('a', { class: 'btn small primary', href: `#/table/${r.id}` }, T('回到座位'))
      : free ? h('a', { class: 'btn small primary', href: `#/join/${r.id}` }, T('加入'))
        : null;
    const watch = h('a', { class: 'btn small', href: `#/table/${r.id}/watch` }, T('观战'));
    const manage = r.can_manage ? [
      r.status === 'playing' ? h('button', { class: 'btn small', onclick: async () => {
        try { await api.post('/api/room/pause', { room: r.id, paused: !r.pause_by }); refresh(); } catch (e) { toast(tr(e.message), 'err'); }
      } }, r.pause_by ? T('继续') : T('暂停')) : null,
      h('button', { class: 'btn small danger', onclick: () => closeTable(r, refresh) }, T('关闭')),
    ] : [];
    box.appendChild(h('div', { class: 'table-card' },
      h('div', { class: 'tc-top' },
        h('b', {}, `${r.mode === 'teach' ? T('教学') : T('实战')} · ${r.length === 'quick' ? T('快速对局') : T('竞技（打到 A）')}`),
        h('span', { class: 'chip ' + cls }, text)),
      h('div', { class: 'tc-line' }, ...players, ais ? h('span', { class: 'pl ai' }, `AI ×${ais}`) : null,
        r.spectators ? h('span', { class: 'dim' }, T('{n} 人观战', { n: r.spectators })) : null),
      h('div', { class: 'tc-line dim' }, prog),
      h('div', { class: 'tc-actions' }, enter, watch, ...manage)));
  }
  return box;
}

function closeTable(r, refresh) {
  let m;
  const box = h('div', {}, h('h2', {}, T('关闭这张牌桌？')),
    h('div', { class: 'sub' }, T('牌局会结束，桌上的玩家会回到首页。')),
    h('div', { class: 'actions' },
      h('button', { class: 'btn', onclick: () => m.close() }, T('取消')),
      h('button', { class: 'btn danger', onclick: async () => {
        m.close();
        try { await api.post('/api/room/close', { room: r.id }); refresh(); } catch (e) { toast(tr(e.message), 'err'); }
      } }, T('关闭'))));
  m = modal(box, { onClose: () => {} });
}

// ---------------------------------------------------------------- setup
async function viewSetup() {
  const cfg = { think: 'fast', scoring: 'none', mode: 'teach', length: 'full', seats: ['human', 'ai', 'ai', 'ai'], model: '', seatModels: ['', '', '', ''], ai_speed: 1, name: nameStore.get() || T('房主'), hostPlays: true };
  let models = [];
  let room = null;
  let info = {};
  let me = {};
  try { [models, room, info, me] = await Promise.all([api.get('/api/models'), api.get('/api/room'), api.get('/api/info'), getMe()]); } catch (e) { toast(tr(e.message), 'err'); }
  if (!nameStore.get() && !me.admin) cfg.name = T('玩家'); // "房主" is the host computer's default name
  const canSearch = !!(info.device && info.device.torch);  // look-ahead search needs PyTorch
  if (models.length) cfg.model = models[0].id;

  const root = h('div', { class: 'page' });
  app.replaceChildren(root);

  function modelSelect(value, onchange, small) {
    const sel = h('select', { onchange: (e) => onchange(e.target.value), 'aria-label': T('选择模型') });
    if (small) sel.appendChild(h('option', { value: '' }, T('同默认模型')));
    fillModelSelect(sel, models, value);
    return sel;
  }

  function render() {
    const teach = cfg.mode === 'teach';
    if (teach) cfg.seats = ['human', 'ai', 'ai', 'ai'];
    const choice = (on, title, desc, click) => h('button', { class: 'choice' + (on ? ' on' : ''), onclick: click, type: 'button' }, h('b', {}, title), h('span', {}, desc));
    const mini = h('div', { class: 'mini-table' }, h('div', { class: 'center' }, T('牌桌')));
    for (let i = 0; i < 4; i++) {
      const isHuman = cfg.seats[i] === 'human';
      const chip = h('div', { class: `seat-chip sc-${i} team${i % 2}` },
        h('div', { class: 'who' }, h('b', {}, T(SEAT_POS[i])), i % 2 === 0 ? '  ' + T('我方') : '  ' + T('对方')),
      );
      if (teach) {
        chip.appendChild(h('div', {}, i === 0 ? T('你') : 'AI'));
      } else {
        const seg = h('div', { class: 'seg' },
          h('button', { class: isHuman ? 'on' : '', type: 'button', onclick: () => { cfg.seats[i] = 'human'; render(); } }, T('玩家')),
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
    const warn = room && room.exists && room.status === 'playing' ? h('p', { class: 'dim' }, T('提示：当前已有一桌在进行，开新桌会结束它。')) : null;
    root.replaceChildren(
      h('div', { class: 'nav' }, h('a', { class: 'btn small', href: '#/' }, T('返回')), h('span', { class: 'title' }, T('开一桌')), h('span', { class: 'spacer' })),
      h('div', { class: 'setup' },
        h('div', { class: 'opts' },
          h('div', { class: 'panel' }, h('h2', {}, T('模式')),
            h('div', { class: 'choice-row' },
              choice(teach, T('教学模式'), T('你和三个 AI 对局。不限时，可以让 AI 给出出牌建议、点评你的出牌，也可以明牌查看所有人的手牌。'), () => { cfg.mode = 'teach'; render(); }),
              choice(!teach, T('实战模式'), T('与 AI 或局域网玩家对局。每步限时 20 秒，超时随机出牌；看不到别人的牌，自己出完后可以看队友的牌。'), () => { cfg.mode = 'real'; render(); }),
            )),
          h('div', { class: 'panel' }, h('h2', {}, T('对局长度')),
            h('div', { class: 'choice-row' },
              choice(cfg.length === 'quick', T('快速对局'), T('每局随机抽一个级牌，单局结算积分，不进贡。适合随时开打。'), () => { cfg.length = 'quick'; render(); }),
              choice(cfg.length === 'full', T('竞技模式'), T('从 2 打到 A：双上升三级、进贡还贡、抗贡、接风，过 A 才算赢。'), () => { cfg.length = 'full'; render(); }),
            )),
          cfg.length === 'quick' ? h('div', { class: 'panel' }, h('h2', {}, T('计分')),
            h('div', { class: 'choice-row', style: { gridTemplateColumns: '1fr 1fr 1fr' } },
              choice(cfg.scoring === 'none', T('只算升级'), T('按升级数结算：双上 3、一三游 2、一四游 1。'), () => { cfg.scoring = 'none'; render(); }),
              choice(cfg.scoring === 'standard', T('标准倍数'), T('底分同升级数；6 张以上炸弹、同花顺、天王炸每出一个翻一倍，封顶 16 倍。'), () => { cfg.scoring = 'standard'; render(); }),
              choice(cfg.scoring === 'all', T('全炸翻倍'), T('每出一个炸弹（含同花顺、天王炸）翻一倍，封顶 256 倍。'), () => { cfg.scoring = 'all'; render(); }),
            ),
            cfg.scoring !== 'none' ? h('p', { class: 'dim', style: { fontSize: '12.5px', margin: '10px 0 0' } },
              T('倍数只影响每局结算的积分，AI 的打法不变，仍以赢下本局、多升级为目标。')) : null) : null,
          !canSearch ? null : h('div', { class: 'panel' }, h('h2', {}, T('AI 思考方式')),
            h('div', { class: 'choice-row' },
              choice(cfg.think === 'fast', T('神经网络'), T('推荐。直接用训练好的网络出牌，不用等待。对随包模型这是最强的方式。'), () => { cfg.think = 'fast'; render(); }),
              choice(cfg.think === 'deep', T('推演搜索（实验）'), T('网络先挑出最好的几手，再在随机推测的对手手牌里各模拟打下去比较。对训练不久的模型有帮助；对随包模型反而略弱，每步也要多想半秒左右。'), () => { cfg.think = 'deep'; render(); }),
            )),
          h('div', { class: 'panel' }, h('h2', {}, T('AI 与玩家')),
            h('div', { class: 'field' }, h('label', {}, T('AI 模型')), models.length ? modelSelect(cfg.model, (v) => { cfg.model = v; }) : h('div', { class: 'dim' }, T('还没有模型：请先到训练中心训练，或下载 Release 中的模型。'))),
            h('div', { class: 'grid2', style: { marginTop: '12px' } },
              h('div', { class: 'field' }, h('label', {}, T('你的昵称')), nameInput),
              h('div', { class: 'field' }, h('label', {}, h('span', {}, T('AI 出牌速度') + ' '), spd), speed)),
          ),
        ),
        h('div', { class: 'panel' }, h('h2', {}, T('座位')),
          h('p', { class: 'muted', style: { margin: '0 0 6px', fontSize: '13px' } }, teach ? T('教学模式固定为你与三个 AI。') : T('对家是你的队友。设为“玩家”的座位，可由同一 WiFi 下的手机或电脑扫码加入。')),
          mini,
          humans > 0 ? h('div', { class: 'field', style: { marginTop: '8px' } }, h('label', {}, T('这台{d}', { d: dev() })),
            h('div', { class: 'choice-row' },
              choice(cfg.hostPlays, T('入座打牌'), teach ? T('在这台{d}上打。', { d: dev() }) : T('你在这台{d}上坐南家（第一个“玩家”座位）。', { d: dev() }), () => { cfg.hostPlays = true; render(); }),
              choice(!cfg.hostPlays, T('只做主机'), T('这台{d}只运行 AI 和显示牌桌，用手机等其他设备扫码入座（看不到任何人的手牌）。', { d: dev() }), () => { cfg.hostPlays = false; render(); }),
            )) : null,
          h('p', { class: 'dim', style: { fontSize: '13px' } }, humans === 0 ? T('四个座位都是 AI：你将以观战者身份旁观（可看到所有手牌）。')
            : !cfg.hostPlays ? T('开桌后用手机扫牌桌上的二维码入座（共 {n} 个玩家座位）。', { n: humans })
            : humans > 1 ? (humans === 2 ? T('需要另外 1 位玩家通过局域网加入后才会开始。') : T('需要另外 {n} 位玩家通过局域网加入后才会开始。', { n: humans - 1 })) : ''),
          warn,
          h('div', { class: 'start-row', style: { display: 'flex', justifyContent: 'flex-end', marginTop: '10px' } },
            h('button', { class: 'btn primary', disabled: !models.length, onclick: start }, humans === 0 ? T('开始观战') : T('开桌'))),
        ),
      ),
    );
  }

  async function start() {
    nameStore.set(cfg.name);
    try {
      const body = {
        mode: cfg.mode, length: cfg.length, think: cfg.think, scoring: cfg.scoring, model: cfg.model, ai_speed: cfg.ai_speed, name: cfg.name, host_plays: cfg.hostPlays,
        seats: cfg.seats.map((k, i) => ({ kind: k, model: cfg.seatModels[i] || cfg.model })),
      };
      const r = await api.post('/api/room/create', body);
      if (r.token) seatStore.set({ room: r.room, seat: r.seat, token: r.token });
      location.hash = r.token ? `#/table/${r.room}` : `#/table/${r.room}/watch`;
    } catch (e) {
      toast(tr(e.message), 'err');
    }
  }
  render();
}

// ---------------------------------------------------------------- join (LAN players)
async function viewJoin(roomId) {
  const root = h('div', { class: 'page' });
  app.replaceChildren(root);
  let room;
  let rooms = [];
  async function load() {
    try {
      rooms = (await api.get('/api/rooms')).rooms;
      const open = rooms.filter((r) => r.seats.some((st) => st.kind === 'human' && (!st.taken || (seatStore.get(r.id) || {}).token)));
      const id = roomId || (open.length === 1 ? open[0].id : (rooms.length === 1 ? rooms[0].id : null));
      room = id ? (await api.get('/api/room?id=' + encodeURIComponent(id))) : null;
    } catch (e) { room = { exists: false }; }
    render();
  }
  function render() {
    const nav = h('div', { class: 'nav' }, h('a', { class: 'btn small', href: '#/' }, T('返回')), h('span', { class: 'title' }, T('加入牌桌')));
    if (!room && rooms.length) { // several tables: pick one
      root.replaceChildren(nav, h('div', { class: 'panel' }, h('h2', {}, T('选一张牌桌')), tableList(rooms, rooms.length, load)));
      return;
    }
    if (!room || !room.exists) {
      root.replaceChildren(nav, h('div', { class: 'panel' }, h('h2', {}, roomId ? T('这张牌桌已经关闭了') : T('主机还没有开桌')),
        h('p', { class: 'muted' }, T('请在主机上点“开一桌”，把座位设为“玩家”，然后点下面的按钮刷新。')),
        h('button', { class: 'btn', onclick: load }, T('刷新'))));
      return;
    }
    const mine = seatStore.get(room.id);
    const nameInput = h('input', { type: 'text', value: nameStore.get(), maxlength: 10, placeholder: T('你的昵称') });
    const list = h('div', { class: 'seat-list' });
    room.seats.forEach((s, i) => {
      let action;
      if (mine && mine.token && mine.seat === i) action = h('a', { class: 'btn small primary', href: `#/table/${room.id}` }, T('回到座位'));
      else if (s.kind === 'human' && !s.taken) action = h('button', { class: 'btn small primary', onclick: () => join(i, nameInput.value) }, T('坐这里'));
      else action = h('span', { class: 'dim' }, s.kind === 'ai' ? 'AI' : T('已有人'));
      list.appendChild(h('div', { class: 'seat-row' },
        h('span', { class: 'pos' }, T(DIRS[i])),
        h('span', { class: 'nm' }, s.kind === 'ai' ? tr(s.name) : (s.taken ? tr(s.name) : T('空位')), h('span', { class: 'dim' }, i % 2 === 0 ? '  ' + T('（与南家同队）') : '  ' + T('（与东家同队）'))),
        s.kind === 'human' ? h('span', { class: 'dot' + (s.connected ? ' on' : '') }) : null,
        action));
    });
    root.replaceChildren(nav, h('div', { class: 'panel' },
      h('h2', {}, `${room.mode === 'teach' ? T('教学模式') : T('实战模式')}${T('，')}${room.length === 'quick' ? T('快速对局') : T('竞技模式（打到 A）')}`),
      h('div', { class: 'field', style: { maxWidth: '280px' } }, h('label', {}, T('昵称')), nameInput),
      list,
      h('div', { style: { marginTop: '14px', display: 'flex', gap: '10px' } },
        h('button', { class: 'btn', onclick: load }, T('刷新')),
        h('a', { class: 'btn', href: `#/table/${room.id}/watch` }, T('观战'))),
    ));
  }
  async function join(seat, name) {
    nameStore.set(name);
    try {
      const r = await api.post('/api/room/join', { room: room.id, seat, name });
      seatStore.set({ room: r.room, seat: r.seat, token: r.token });
      location.hash = `#/table/${r.room}`;
    } catch (e) { toast(tr(e.message), 'err'); load(); }
  }
  load();
}

// ---------------------------------------------------------------- router
function route() {
  if (cleanup) { try { cleanup(); } catch {} cleanup = null; }
  document.body.classList.remove('in-game');
  const hash = location.hash || '#/';
  const part = hash.split('/'); // #/table/<room>[/watch], #/join/<room>
  if (hash.startsWith('#/play')) viewSetup();
  else if (hash.startsWith('#/join')) viewJoin(part[2] || null);
  else if (hash.startsWith('#/train')) cleanup = mountTrain(app);
  else if (hash.startsWith('#/table')) {
    document.body.classList.add('in-game');
    cleanup = mountTable(app, part[2] || seatStore.current(), part[3] === 'watch');
  } else viewHome();
}
window.addEventListener('hashchange', route);
route();
