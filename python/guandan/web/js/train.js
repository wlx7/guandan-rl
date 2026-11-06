import { api, toast, h, modal, fillModelSelect, modelLabel, getMe } from './util.js';
import { LineChart, SERIES_COLORS, fmtFrames } from './chart.js';
import { T, tr } from './i18n.js';

const OPP = [
  { key: 'rule', name: T('对规则 AI') },
  { key: 'base', name: T('对基准模型') },
  { key: 'random', name: T('对随机出牌') },
];

function fmtDur(s) {
  s = Math.round(s || 0);
  const hh = Math.floor(s / 3600);
  const mm = Math.floor((s % 3600) / 60);
  return hh ? T('{h} 小时 {m} 分', { h: hh, m: mm }) : T('{m} 分 {s} 秒', { m: mm, s: s % 60 });
}

export function mountTrain(app) {
  const st = { info: null, models: [], runs: [], sel: null, rows: [], since: 0, timer: 0, charts: {}, closed: false, logOpen: false, stopping: {}, showHidden: false };
  const now = new Date();
  const pad = (n) => String(n).padStart(2, '0');
  const form = {
    name: `run-${now.getFullYear()}${pad(now.getMonth() + 1)}${pad(now.getDate())}-${pad(now.getHours())}${pad(now.getMinutes())}`,
    preset: 'medium', base: '', device: 'auto', actor_device: 'cpu', actors: 4, envs: 64, batch: 0,
    lr: 0, epsilon: 0.02, league: 0, aux: 1.0, eval_interval: 10, eval_rounds: 400, max_hours: 0,
  };
  const root = h('div', { class: 'page' });
  const left = h('div', { class: 'panel' });
  const right = h('div', { class: 'panel' });
  root.append(h('div', { class: 'nav' }, h('a', { class: 'btn small', href: '#/' }, T('返回')), h('span', { class: 'title' }, T('训练中心')), h('span', { class: 'spacer' })),
    h('div', { class: 'train' }, left, right));
  app.replaceChildren(root);

  async function init() {
    const me = await getMe();
    if (!(me.admin || me.guest_train)) {
      left.replaceChildren(h('h2', {}, T('训练中心只有房主可以使用')),
        h('p', {}, T('在运行游戏的那台电脑上打开训练中心；或者在首页点“房主登录”，输入房主密码后这台设备也能使用。')),
        h('a', { class: 'btn', href: '#/' }, T('返回首页')));
      right.replaceChildren();
      return;
    }
    try {
      [st.info, st.models, st.runs] = await Promise.all([api.get('/api/info'), api.get('/api/models'), api.get('/api/train/runs')]);
    } catch (e) { toast(tr(e.message), 'err'); return; }
    form.preset = st.info.recommended_preset || 'medium';
    const cpu = st.info.cpu_count || 4;
    if (st.info.device.cuda) { form.actor_device = 'cuda'; form.actors = 3; form.envs = 256; }
    else { form.actor_device = 'cpu'; form.actors = Math.max(1, Math.min(16, cpu - (cpu >= 12 ? 4 : 2))); form.envs = 64; }
    form.batch = st.info.device.cuda ? 4096 : 512;
    const running = st.runs.find((r) => r.status === 'running');
    st.sel = running ? running.name : ((st.runs.find((r) => !r.hidden) || st.runs[0] || {}).name || null);
    renderForm();
    renderRuns();
    if (st.sel) loadRun(st.sel, true);
    st.timer = setInterval(poll, 3000);
  }

  // ------------------------------------------------------------ form
  function renderForm() {
    const info = st.info;
    const dev = info.device || {};
    if (info.can_train === false) {
      left.replaceChildren(h('h2', {}, T('这个版本只能对战')),
        h('p', {}, info.portable
          ? T('你打开的是免安装版：一个文件、双击就能打牌和局域网联机，但不带训练需要的 PyTorch（显卡版有好几 GB）。')
          : T('没有检测到 PyTorch，训练需要它。')),
        h('p', {}, T('想自己训练模型，请用完整安装版：在 GitHub Release 页面下载 install_windows.bat 等 4 个文件放在一起，双击 install_windows.bat，它会自动安装 Python 和显卡版 PyTorch。')),
        h('p', { class: 'dim', style: { fontSize: '12.5px' } }, T('已有的训练记录在 {home}，右侧仍可查看。', { home: info.home })));
      return;
    }
    const presets = h('div', { class: 'presets' });
    for (const p of info.presets) {
      presets.appendChild(h('div', { class: 'preset' + (form.preset === p.name ? ' on' : ''), role: 'button', tabindex: 0, onclick: () => { form.preset = p.name; renderForm(); } },
        h('span', { class: 'nm' }, p.name),
        h('span', {}, p.recommended ? h('span', { class: 'tag' }, T('推荐')) : ''),
        h('span', { class: 'pr' }, T('{p}M 参数  {mb}MB', { p: (p.params / 1e6).toFixed(2), mb: p.size_fp16_mb })),
        h('span', { class: 'nt' }, tr(p.note))));
    }
    const num = (k, attrs = {}) => h('input', { type: 'number', value: form[k], ...attrs, oninput: (e) => { form[k] = e.target.value === '' ? '' : +e.target.value; } });
    const baseSel = h('select', { onchange: (e) => { form.base = e.target.value; } }, h('option', { value: '' }, T('从零开始（随机初始化）')));
    fillModelSelect(baseSel, st.models, form.base);
    const devSel = h('select', { onchange: (e) => { form.device = e.target.value; } },
      ...[['auto', T('自动')], ['cuda', 'GPU (CUDA)'], ['cpu', 'CPU']].map(([v, t]) => h('option', { value: v, selected: form.device === v }, t)));
    const actDevSel = h('select', { onchange: (e) => { form.actor_device = e.target.value; clampForm(); renderForm(); } },
      ...[['cuda', T('GPU（有显卡时推荐）')], ['cpu', T('CPU（多进程）')]].map(([v, t]) => h('option', { value: v, selected: form.actor_device === v }, t)));
    const cpuN = info.cpu_count || 4;
    left.replaceChildren(
      h('h2', {}, T('新建训练')),
      h('div', { class: 'form' },
        h('div', { class: 'field' }, h('label', {}, T('名称')), h('input', { type: 'text', value: form.name, oninput: (e) => { form.name = e.target.value; } }),
          h('span', { class: 'hint' }, T('用已有名称会从上次的进度继续训练'))),
        h('div', { class: 'field' }, h('span', { class: 'label' }, T('模型大小')), presets,
          h('span', { class: 'hint' }, T('选了基准模型时：选比它更大的尺寸，会先把基准模型“加宽”成大模型（开局打法和原模型完全一样，保留已学到的东西）再继续训练；否则沿用基准模型的大小'))),
        h('div', { class: 'field' }, h('label', {}, T('基准模型')), baseSel, h('span', { class: 'hint' }, T('在已有模型上继续自对弈，评估时会和它对比胜率。'),
          h('a', { href: '#', onclick: (e) => { e.preventDefault(); manageModels(); } }, T('整理模型列表')))),
        h('div', { class: 'grid2' },
          h('div', { class: 'field' }, h('label', {}, T('学习设备')), devSel),
          h('div', { class: 'field' }, h('label', {}, T('自对弈设备')), actDevSel)),
        h('div', { class: 'hint dim', style: { fontSize: '12px' } }, dev.cuda
          ? T('检测到 {gpu}（{mem} GB），CPU {n} 线程', { gpu: dev.gpu, mem: dev.gpu_mem_gb, n: info.cpu_count })
          : T('未检测到 CUDA，将使用 CPU（{n} 线程）', { n: info.cpu_count })),
        h('div', { class: 'grid2' },
          h('div', { class: 'field' }, h('label', {}, T('自对弈进程数')), num('actors', { min: 1, max: 64 }), h('span', { class: 'hint' }, dev.cuda
            ? T('推荐：GPU 自对弈 3 个进程 × 256 局')
            : T('推荐 {n}（留几个线程给学习和评估）', { n: Math.max(1, Math.min(16, cpuN - (cpuN >= 12 ? 4 : 2))) }))),
          h('div', { class: 'field' }, h('label', {}, T('每进程并行对局')), num('envs', { min: 1, max: 512 }),
            h('span', { class: 'hint' }, T('本机上限：{a} 个进程，合计 {e} 局', { a: limitsFor(form.actor_device).max_actors, e: limitsFor(form.actor_device).max_total_envs })))),
        h('details', {}, h('summary', {}, T('更多参数')),
          h('div', { class: 'grid2', style: { marginTop: '8px' } },
            h('div', { class: 'field' }, h('label', {}, T('批大小')), num('batch', { min: 64, step: 64 })),
            h('div', { class: 'field' }, h('label', {}, T('学习率')), num('lr', { step: 0.00005, min: 0 }),
              h('span', { class: 'hint' }, T('0 为自动：从头训练 2e-4，在已有模型上继续训练 5e-5（实测继续训练时小学习率更好）'))),
            h('div', { class: 'field' }, h('label', {}, T('探索率 ε')), num('epsilon', { step: 0.005, min: 0, max: 1 })),
            h('div', { class: 'field' }, h('label', {}, T('辅助任务权重')), num('aux', { step: 0.25, min: 0, max: 4 }),
              h('span', { class: 'hint' }, T('训练时让网络顺便预测其他三家的手牌（对战时不用）。实测明显更强；0 为关闭'))),
            h('div', { class: 'field' }, h('label', {}, T('对手池比例')), num('league', { step: 0.1, min: 0, max: 1 }),
              h('span', { class: 'hint' }, T('这部分对局的对手由早先版本的模型来打。默认 0（纯自对弈）：实测纯自对弈每小时样本多一倍，效果更好'))),
            h('div', { class: 'field' }, h('label', {}, T('评估间隔（分钟）')), num('eval_interval', { min: 1 })),
            h('div', { class: 'field' }, h('label', {}, T('每次评估局数')), num('eval_rounds', { min: 50, step: 50 })),
            h('div', { class: 'field' }, h('label', {}, T('最长训练（小时，0 不限）')), num('max_hours', { min: 0, step: 0.5 })))),
        busyRun() ? h('div', { class: 'hint', style: { fontSize: '12.5px', color: 'var(--brass-hi)' } }, T('训练「{name}」正在运行。显卡和内存只够同时跑一个训练，要开新训练请先停止它。', { name: busyRun() })) : null,
        h('button', { class: 'btn primary', onclick: startRun, disabled: !!busyRun() }, T('开始训练')),
      ));
  }

  function busyRun() {
    const r = st.runs.find((x) => x.status === 'running');
    return r ? r.name : null;
  }

  function limitsFor(dev) {
    const l = (st.info.limits || {})[dev === 'cuda' ? 'cuda' : 'cpu'];
    return l || { max_actors: 64, max_total_envs: 100000 };
  }

  function clampForm() {
    const dev = form.actor_device === 'cuda' ? 'cuda' : 'cpu';
    const lim = limitsFor(dev);
    const msgs = [];
    if (+form.actors > lim.max_actors) { form.actors = lim.max_actors; msgs.push(T('自对弈进程数已限制为 {n}', { n: lim.max_actors })); }
    if (+form.actors * +form.envs > lim.max_total_envs) {
      form.envs = Math.max(16, Math.floor(lim.max_total_envs / form.actors));
      msgs.push(T('每进程并行对局已限制为 {n}', { n: form.envs }));
    }
    return msgs;
  }

  async function startRun() {
    const msgs = clampForm();
    if (msgs.length) { renderForm(); toast(T('{msgs}（受显存和内存限制）', { msgs: msgs.join(T('，')) })); }
    const body = { ...form, eval_interval: Math.round((+form.eval_interval || 10) * 60) };
    if (!body.base) delete body.base;
    try {
      const r = await api.post('/api/train/start', body);
      toast(T('已开始训练 {name}', { name: r.name }));
      st.sel = r.name;
      st.rows = [];
      st.since = 0;
      st.runs = await api.get('/api/train/runs');
      renderRuns();
      loadRun(r.name, true);
    } catch (e) { toast(tr(e.message), 'err'); }
  }

  // ------------------------------------------------------------ runs
  function renderRuns() {
    if (!st.runs.length) {
      right.replaceChildren(h('h2', {}, T('训练记录')), h('div', { class: 'chart-empty' }, st.info && st.info.can_train === false ? T('还没有训练记录。') : T('还没有训练。左侧选择模型大小后点“开始训练”，这里会实时显示胜率变化。')));
      return;
    }
    const tabs = h('div', { class: 'runs-bar' });
    const shown = st.runs.filter((r) => st.showHidden || !r.hidden || r.status === 'running' || r.name === st.sel);
    const nHidden = st.runs.filter((r) => r.hidden && r.status !== 'running').length;
    for (const r of (shown.length ? shown : st.runs)) {
      tabs.appendChild(h('button', { class: 'run-tab' + (r.name === st.sel ? ' on' : '') + (r.hidden ? ' hidden-run' : ''), title: r.name, onclick: () => { st.sel = r.name; st.rows = []; st.since = 0; renderRuns(); loadRun(r.name, true); } },
        h('span', { class: 'dot' + (r.status === 'running' ? ' on' : '') }), r.label || r.name));
    }
    if (nHidden) tabs.appendChild(h('button', { class: 'btn small ghost', onclick: () => { st.showHidden = !st.showHidden; renderRuns(); } },
      st.showHidden ? T('收起已隐藏的训练') : T('显示已隐藏的训练（{n}）', { n: nHidden })));
    const run = st.runs.find((r) => r.name === st.sel) || shown[0] || st.runs[0];
    st.sel = run.name;
    const running = run.status === 'running';
    const head = h('div', { style: { display: 'flex', alignItems: 'center', gap: '10px', flexWrap: 'wrap', marginBottom: '12px' } },
      h('h2', { style: { margin: 0 } }, run.label || run.name),
      run.label ? h('span', { class: 'dim', style: { fontSize: '13px' } }, run.name) : null,
      h('span', { class: 'tag ' + (running ? 'on' : 'off') }, running ? T('训练中') : T('已停止')),
      h('button', { class: 'btn small ghost', title: T('改一个好记的显示名称（文件夹名不变）'), onclick: () => renameEntry(`runs/${run.name}`, run.label || '', run.name) }, T('改名')),
      h('button', { class: 'btn small ghost', title: T('隐藏后不再出现在模型列表里，文件不会删除，随时可以恢复'), onclick: () => hideEntry(`runs/${run.name}`, !run.hidden) }, run.hidden ? T('取消隐藏') : T('隐藏')),
      h('span', { class: 'dim', style: { fontSize: '13px' } }, `${run.preset || ''} ${run.params ? T('{p}M 参数', { p: (run.params / 1e6).toFixed(2) }) : ''} ${run.device ? '· ' + run.device : ''}`),
      h('span', { class: 'spacer', style: { flex: 1 } }),
      running ? stopButtons(run)
        : h('button', { class: 'btn small', onclick: () => resumeRun(run), disabled: !!busyRun() }, T('继续训练')));
    const statsBox = h('div', { class: 'stats', id: 'stats' });
    const charts = h('div', { class: 'charts' });
    st.charts = {};
    const mk = (id, title, legend, wide, opts) => {
      const canvas = h('canvas');
      const tip = h('div', { class: 'chart-tip' });
      const lg = legend ? h('div', { class: 'legend' }, ...legend.map((l) => h('span', {}, h('i', { style: { background: l.color } }), l.name))) : null;
      const box = h('div', { class: 'chart-box' + (wide ? ' wide' : '') }, h('h3', {}, title), lg, canvas, tip);
      charts.appendChild(box);
      st.charts[id] = new LineChart(canvas, tip, opts);
      st.charts[id].box = box;
    };
    const legend = OPP.map((o, i) => ({ name: o.name, color: SERIES_COLORS[i] }));
    mk('win', T('每局胜率（头游所在方获胜）'), legend, true, { yMin: 0, yMax: 1, refLine: 0.5, yFormat: (v) => `${Math.round(v * 100)}%` });
    mk('score', T('平均每局升级数'), legend, false, { refLine: 0, yFormat: (v) => (v >= 0 ? '+' : '') + v.toFixed(1) });
    mk('loss', T('训练损失'), null, false, { yFormat: (v) => v.toFixed(3) });
    mk('fps', T('自对弈速度（样本/秒）'), null, false, { yFormat: (v) => fmtFrames(v) });
    mk('rps', T('每秒完成对局数'), null, false, { yFormat: (v) => v.toFixed(0) });
    const evalTable = h('div', { id: 'evaltable' });
    const ck = h('div', { class: 'result-list' });
    for (const f of run.checkpoints || []) {
      const what = f === 'model.pt' ? T('最新模型，每 5 分钟自动保存。开桌时在“AI 模型”里选「训练 {name}：model.pt」即可用它对战', { name: run.name })
        : T('第 {k} 千样本时的快照（每小时保留一份，可用来比较不同阶段的水平）', { k: f.replace('model_', '').replace('k.pt', '') });
      ck.appendChild(h('div', { class: 'result-row', style: { gridTemplateColumns: 'minmax(8em, auto) 1fr auto' } },
        h('span', { class: 'pl', style: { fontFamily: 'var(--num)', fontWeight: 600 } }, f),
        h('span', { class: 'rd', style: { color: 'var(--text-2)' } }, what),
        h('button', { class: 'btn small', title: T('复制到模型库并起个名字'), onclick: () => exportModel(run.name, f) }, T('加入模型库'))));
    }
    const log = h('pre', { class: 'log', id: 'trainlog' }, '');
    const det = h('details', { open: st.logOpen, ontoggle: (e) => { st.logOpen = e.target.open; if (st.logOpen) loadLog(); } }, h('summary', {}, T('训练日志')), log);
    const sep = st.info.home.includes('\\') ? '\\' : '/';
    right.replaceChildren(tabs, head, statsBox, charts,
      h('h3', { style: { fontSize: '14px', margin: '16px 0 6px' } }, T('评估记录')),
      h('div', { class: 'dim', style: { fontSize: '12px', margin: '0 0 6px' } }, T('复式评估：每副牌打两遍、双方交换座位，每次评估用同一批牌。“升级”为平均每局升级数，± 为标准误。')),
      evalTable,
      h('h3', { style: { fontSize: '14px', margin: '16px 0 6px' } }, T('这个训练输出的模型')),
      h('div', { class: 'dim', style: { fontSize: '12.5px', marginBottom: '8px' } }, T('保存在 {path}。另有 ckpt_latest.pt（含优化器，用于“继续训练”）。', { path: `${st.info.home}${sep}runs${sep}${run.name}` })),
      ck.childNodes.length ? ck : h('div', { class: 'dim', style: { fontSize: '13px' } }, T('训练开始约 5 分钟后会自动保存第一个模型。')),
      det);
    drawAll();
  }

  function stopButtons(run) {
    const since = st.stopping[run.name];
    if (!since) return h('button', { class: 'btn small danger', onclick: () => stopRun(run.name) }, T('停止训练'));
    const waited = (Date.now() - since) / 1000;
    return h('span', { style: { display: 'inline-flex', gap: '8px', alignItems: 'center' } },
      h('span', { class: 'dim', style: { fontSize: '13px' } }, T('正在保存并停止…')),
      waited > 20 ? h('button', { class: 'btn small danger', onclick: () => forceStop(run.name) }, T('强制结束')) : null);
  }
  async function stopRun(name) {
    try {
      await api.post('/api/train/stop', { name });
      st.stopping[name] = Date.now();
      toast(T('正在保存模型并停止；1 分钟内没有响应会自动强制结束'));
      renderRuns();
    } catch (e) { toast(tr(e.message), 'err'); }
  }
  async function forceStop(name) {
    try { await api.post('/api/train/stop', { name, force: true }); toast(T('已强制结束（最近一次自动保存的模型仍然可用）')); setTimeout(refreshRuns, 800); } catch (e) { toast(tr(e.message), 'err'); }
  }
  async function resumeRun(run) {
    try {
      const r = await api.post('/api/train/start', { name: run.name, preset: run.preset || form.preset, actors: form.actors, envs: form.envs, batch: form.batch, device: form.device, actor_device: form.actor_device, eval_interval: Math.round(form.eval_interval * 60), eval_rounds: form.eval_rounds, base: run.base ? undefined : undefined });
      toast(T('继续训练 {name}', { name: r.name }));
      setTimeout(refreshRuns, 1500);
    } catch (e) { toast(tr(e.message), 'err'); }
  }
  async function refreshLists() {
    try { [st.models, st.runs] = await Promise.all([api.get('/api/models'), api.get('/api/train/runs')]); } catch (e) { toast(tr(e.message), 'err'); }
    renderForm();
    renderRuns();
  }
  async function hideEntry(key, hidden) {
    try { await api.post('/api/models/manage', { key, hidden }); toast(hidden ? T('已隐藏（文件仍保留，可在“整理模型列表”里恢复）') : T('已恢复显示')); } catch (e) { toast(tr(e.message), 'err'); }
    refreshLists();
  }
  function renameEntry(key, current, fallback) {
    const input = h('input', { type: 'text', value: current, placeholder: fallback, maxlength: 60 });
    let m;
    const save = async () => {
      try { await api.post('/api/models/manage', { key, label: input.value }); m.close(); toast(T('已改名')); refreshLists(); } catch (e) { toast(tr(e.message), 'err'); }
    };
    input.addEventListener('keydown', (e) => { if (e.key === 'Enter') save(); });
    const box = h('div', {}, h('h2', {}, T('改名')), h('div', { class: 'sub' }, T('只改显示名称，文件和文件夹不变。留空恢复默认名称。')), input,
      h('div', { class: 'actions' }, h('button', { class: 'btn', onclick: () => m.close() }, T('取消')), h('button', { class: 'btn primary', onclick: save }, T('保存'))));
    m = modal(box, { onClose: () => {} });
    setTimeout(() => input.focus(), 0);
  }
  async function manageModels() {
    let all;
    try { all = await api.get('/api/models?all=true'); } catch (e) { toast(tr(e.message), 'err'); return; }
    // one row per model library file and per training run (its latest model); snapshots stay out of the way
    const rows = all.filter((m) => m.group !== '训练' || m.id.endsWith('/model.pt'));
    const keyOf = (m) => (m.group === '训练' ? m.id.replace(/\/model\.pt$/, '') : m.id);
    const list = h('div', { class: 'result-list', style: { maxHeight: '60vh', overflow: 'auto' } });
    // labels are shown translated; "shown" is what a label counts as unchanged against, so an untouched
    // (translated) default name is never saved back as a custom name
    const state = new Map(rows.map((m) => [keyOf(m), { label: tr(m.label), shown: tr(m.label), hidden: !!m.hidden, orig: m }]));
    for (const m of rows) {
      const k = keyOf(m);
      const v = state.get(k);
      const input = h('input', { type: 'text', value: v.shown, maxlength: 60, oninput: (e) => { v.label = e.target.value; } });
      const chk = h('input', { type: 'checkbox', checked: !m.hidden, disabled: m.group === '内置', onchange: (e) => { v.hidden = !e.target.checked; } });
      list.appendChild(h('div', { class: 'result-row', style: { gridTemplateColumns: '4.5em 1fr auto' } },
        h('span', { class: 'rd' }, tr(m.group)),
        input,
        h('label', { class: 'dim', style: { fontSize: '12.5px', display: 'inline-flex', gap: '4px', alignItems: 'center' }, title: m.id }, chk, T('显示'))));
    }
    let md;
    const save = async () => {
      try {
        for (const [key, v] of state) {
          const body = { key };
          if (v.label !== v.shown) body.label = v.label;
          if (v.hidden !== !!v.orig.hidden) body.hidden = v.hidden;
          if (Object.keys(body).length > 1) await api.post('/api/models/manage', body);
        }
        md.close(); toast(T('模型列表已更新')); refreshLists();
      } catch (e) { toast(tr(e.message), 'err'); }
    };
    const box = h('div', {}, h('h2', {}, T('整理模型列表')),
      h('div', { class: 'sub' }, T('改名只改显示名称；取消“显示”后不再出现在选择列表里，文件不会删除。每个训练只列出最新模型，各阶段快照可在训练中心“加入模型库”。')),
      list,
      h('div', { class: 'actions' }, h('button', { class: 'btn', onclick: () => md.close() }, T('取消')), h('button', { class: 'btn primary', onclick: save }, T('保存'))));
    md = modal(box, { onClose: () => {} });
    const mbox = md.el.querySelector('.modal');
    if (mbox) mbox.style.width = 'min(640px, 100%)';
  }

  function exportModel(run, file) {
    const input = h('input', { type: 'text', value: `${run}-${file.replace('.pt', '')}` });
    const box = h('div', {}, h('h2', {}, T('加入模型库')), h('div', { class: 'sub' }, T('起个名字，之后开桌时可以选择它')), input);
    let m;
    box.appendChild(h('div', { class: 'actions' }, h('button', { class: 'btn', onclick: () => m.close() }, T('取消')),
      h('button', { class: 'btn primary', onclick: async () => {
        try { await api.post('/api/train/export', { run, file, as: input.value }); toast(T('已加入模型库')); m.close(); st.models = await api.get('/api/models'); renderForm(); } catch (e) { toast(tr(e.message), 'err'); }
      } }, T('保存'))));
    m = modal(box, { onClose: () => {} });
  }

  async function loadRun(name, full) {
    if (full) { st.rows = []; st.since = 0; }
    try {
      const r = await api.get(`/api/train/metrics?name=${encodeURIComponent(name)}&since=${st.since}`);
      if (name !== st.sel) return;
      if (r.rows.length) {
        st.rows.push(...r.rows);
        st.since = r.rows[r.rows.length - 1].t;
      }
      drawAll();
      if (st.logOpen) loadLog();
    } catch (e) { /* ignore */ }
  }
  async function loadLog() {
    try {
      const r = await api.get(`/api/train/log?name=${encodeURIComponent(st.sel)}`);
      const el = document.getElementById('trainlog');
      if (el) { el.textContent = r.log || T('（暂无日志）'); el.scrollTop = el.scrollHeight; }
    } catch {}
  }
  async function refreshRuns() {
    try {
      const runs = await api.get('/api/train/runs');
      const changed = JSON.stringify(runs.map((r) => [r.name, r.status, r.checkpoints])) !== JSON.stringify(st.runs.map((r) => [r.name, r.status, r.checkpoints]));
      const wasBusy = busyRun();
      st.runs = runs;
      for (const r of runs) if (r.status !== 'running') delete st.stopping[r.name];
      if (changed) renderRuns();
      if (wasBusy !== busyRun()) renderForm();
      else if (Object.keys(st.stopping).length) renderRuns();
    } catch {}
  }
  let pollN = 0;
  function poll() {
    if (st.closed || document.hidden) return;
    if (st.sel) loadRun(st.sel, false);
    if (++pollN % 2 === 0) refreshRuns();
  }

  function drawAll() {
    const rows = st.rows;
    const trains = rows.filter((r) => r.type === 'train');
    const evals = rows.filter((r) => r.type === 'eval');
    const statsBox = document.getElementById('stats');
    if (statsBox) {
      const lt = trains[trains.length - 1];
      const le = evals[evals.length - 1];
      const stat = (k, v) => h('div', { class: 'stat' }, h('div', { class: 'k' }, k), h('div', { class: 'v' }, v));
      statsBox.replaceChildren(
        stat(T('训练样本'), lt ? fmtFrames(lt.frames) : '0'),
        stat(T('速度'), lt ? `${fmtFrames(lt.fps)}/s` : '—'),
        stat(T('训练时长'), lt ? fmtDur(lt.train_seconds) : '—'),
        stat(T('对规则 AI 胜率'), le && le.rule ? `${(le.rule.win_rate * 100).toFixed(1)}%` : '—'));
    }
    const C = st.charts;
    if (!C.win) return;
    const evalSeries = (field) => OPP.map((o, i) => ({ name: o.name, color: SERIES_COLORS[i], points: evals.filter((e) => e[o.key]).map((e) => ({ x: e.frames, y: e[o.key][field] })) }))
      .filter((s) => s.points.length);
    const empty = (chart, msg) => {
      let ph = chart.box.querySelector('.chart-empty');
      if (msg) {
        chart.c.style.display = 'none';
        if (!ph) { ph = h('div', { class: 'chart-empty' }); chart.box.appendChild(ph); }
        ph.textContent = msg;
      } else {
        chart.c.style.display = 'block';
        if (ph) ph.remove();
      }
    };
    const ws = evalSeries('win_rate');
    empty(C.win, ws.length ? null : T('第一次评估会在训练开始约 10 分钟后完成（可在“评估间隔”调整）'));
    C.win.set(ws);
    const ss = evalSeries('score');
    empty(C.score, ss.length ? null : T('等待评估结果'));
    C.score.set(ss);
    const one = (key) => [{ name: '', color: SERIES_COLORS[0], points: trains.filter((r) => r[key] != null && r.updates !== 0).map((r) => ({ x: r.frames, y: r[key] })) }];
    const ls = one('loss');
    empty(C.loss, ls[0].points.length ? null : T('等待训练数据'));
    C.loss.set(ls);
    const fs = one('fps');
    empty(C.fps, fs[0].points.length ? null : T('等待训练数据'));
    C.fps.set(fs);
    const rs = one('rounds_per_sec');
    empty(C.rps, rs[0].points.length ? null : T('等待训练数据'));
    C.rps.set(rs);
    const tb = document.getElementById('evaltable');
    if (tb) {
      if (!evals.length) { tb.replaceChildren(h('div', { class: 'dim', style: { fontSize: '13px' } }, T('暂无评估'))); }
      else {
        const t = h('table', { class: 'evals' }, h('tr', {}, h('th', {}, T('样本数')), ...OPP.map((o) => h('th', {}, T('{opp} 胜率 / 升级', { opp: o.name })))));
        for (const e of evals.slice(-8).reverse()) {
          const cell = (r) => {
            if (!r) return '—';
            const sc = `${r.score >= 0 ? '+' : ''}${r.score.toFixed(2)}${r.score_se != null ? ` ±${r.score_se.toFixed(2)}` : ''}`;
            return `${(r.win_rate * 100).toFixed(1)}% / ${sc}`;
          };
          t.appendChild(h('tr', {}, h('td', {}, fmtFrames(e.frames)), ...OPP.map((o) => h('td', {}, cell(e[o.key])))));
        }
        tb.replaceChildren(t);
      }
    }
  }

  const onResize = () => Object.values(st.charts).forEach((c) => c.draw());
  window.addEventListener('resize', onResize);
  init();
  return () => { st.closed = true; clearInterval(st.timer); window.removeEventListener('resize', onResize); };
}
