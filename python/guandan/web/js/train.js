import { api, toast, h, modal } from './util.js';
import { LineChart, SERIES_COLORS, fmtFrames } from './chart.js';

const OPP = [
  { key: 'rule', name: '对规则 AI' },
  { key: 'base', name: '对基准模型' },
  { key: 'random', name: '对随机出牌' },
];

function fmtDur(s) {
  s = Math.round(s || 0);
  const hh = Math.floor(s / 3600);
  const mm = Math.floor((s % 3600) / 60);
  return hh ? `${hh} 小时 ${mm} 分` : `${mm} 分 ${s % 60} 秒`;
}

export function mountTrain(app) {
  const st = { info: null, models: [], runs: [], sel: null, rows: [], since: 0, timer: 0, charts: {}, closed: false, logOpen: false, stopping: {} };
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
  root.append(h('div', { class: 'nav' }, h('a', { class: 'btn small', href: '#/' }, '返回'), h('span', { class: 'title' }, '训练中心'), h('span', { class: 'spacer' })),
    h('div', { class: 'train' }, left, right));
  app.replaceChildren(root);

  async function init() {
    try {
      [st.info, st.models, st.runs] = await Promise.all([api.get('/api/info'), api.get('/api/models'), api.get('/api/train/runs')]);
    } catch (e) { toast(e.message, 'err'); return; }
    form.preset = st.info.recommended_preset || 'medium';
    const cpu = st.info.cpu_count || 4;
    if (st.info.device.cuda) { form.actor_device = 'cuda'; form.actors = 3; form.envs = 256; }
    else { form.actor_device = 'cpu'; form.actors = Math.max(1, Math.min(16, cpu - (cpu >= 12 ? 4 : 2))); form.envs = 64; }
    form.batch = st.info.device.cuda ? 4096 : 512;
    const running = st.runs.find((r) => r.status === 'running');
    st.sel = running ? running.name : (st.runs[0] && st.runs[0].name) || null;
    renderForm();
    renderRuns();
    if (st.sel) loadRun(st.sel, true);
    st.timer = setInterval(poll, 3000);
  }

  // ------------------------------------------------------------ form
  function renderForm() {
    const info = st.info;
    const dev = info.device || {};
    const presets = h('div', { class: 'presets' });
    for (const p of info.presets) {
      presets.appendChild(h('div', { class: 'preset' + (form.preset === p.name ? ' on' : ''), role: 'button', tabindex: 0, onclick: () => { form.preset = p.name; renderForm(); } },
        h('span', { class: 'nm' }, p.name),
        h('span', {}, p.recommended ? h('span', { class: 'tag' }, '推荐') : ''),
        h('span', { class: 'pr' }, `${(p.params / 1e6).toFixed(2)}M 参数  ${p.size_fp16_mb}MB`),
        h('span', { class: 'nt' }, p.note)));
    }
    const num = (k, attrs = {}) => h('input', { type: 'number', value: form[k], ...attrs, oninput: (e) => { form[k] = e.target.value === '' ? '' : +e.target.value; } });
    const baseSel = h('select', { onchange: (e) => { form.base = e.target.value; } }, h('option', { value: '' }, '从零开始（随机初始化）'));
    for (const m of st.models) baseSel.appendChild(h('option', { value: m.id, selected: form.base === m.id }, `${m.source}：${m.id.split('/').slice(-1)[0]}`));
    const devSel = h('select', { onchange: (e) => { form.device = e.target.value; } },
      ...[['auto', '自动'], ['cuda', 'GPU (CUDA)'], ['cpu', 'CPU']].map(([v, t]) => h('option', { value: v, selected: form.device === v }, t)));
    const actDevSel = h('select', { onchange: (e) => { form.actor_device = e.target.value; clampForm(); renderForm(); } },
      ...[['cuda', 'GPU（有显卡时推荐）'], ['cpu', 'CPU（多进程）']].map(([v, t]) => h('option', { value: v, selected: form.actor_device === v }, t)));
    left.replaceChildren(
      h('h2', {}, '新建训练'),
      h('div', { class: 'form' },
        h('div', { class: 'field' }, h('label', {}, '名称'), h('input', { type: 'text', value: form.name, oninput: (e) => { form.name = e.target.value; } }),
          h('span', { class: 'hint' }, '用已有名称会从上次的进度继续训练')),
        h('div', { class: 'field' }, h('span', { class: 'label' }, '模型大小'), presets,
          h('span', { class: 'hint' }, '从基准模型继续训练时，沿用基准模型的大小')),
        h('div', { class: 'field' }, h('label', {}, '基准模型'), baseSel, h('span', { class: 'hint' }, '在已有模型上继续自对弈，评估时会和它对比胜率')),
        h('div', { class: 'grid2' },
          h('div', { class: 'field' }, h('label', {}, '学习设备'), devSel),
          h('div', { class: 'field' }, h('label', {}, '自对弈设备'), actDevSel)),
        h('div', { class: 'hint dim', style: { fontSize: '12px' } }, dev.cuda ? `检测到 ${dev.gpu}（${dev.gpu_mem_gb} GB），CPU ${info.cpu_count} 线程` : `未检测到 CUDA，将使用 CPU（${info.cpu_count} 线程）`),
        h('div', { class: 'grid2' },
          h('div', { class: 'field' }, h('label', {}, '自对弈进程数'), num('actors', { min: 1, max: 64 }), h('span', { class: 'hint' }, dev.cuda ? '推荐：GPU 自对弈 3 个进程 × 256 局' : `推荐 ${Math.max(1, Math.min(16, (info.cpu_count || 4) - ((info.cpu_count || 4) >= 12 ? 4 : 2)))}（留几个线程给学习和评估）`)),
          h('div', { class: 'field' }, h('label', {}, '每进程并行对局'), num('envs', { min: 1, max: 512 }),
            h('span', { class: 'hint' }, `本机上限：${limitsFor(form.actor_device).max_actors} 个进程，合计 ${limitsFor(form.actor_device).max_total_envs} 局`))),
        h('details', {}, h('summary', {}, '更多参数'),
          h('div', { class: 'grid2', style: { marginTop: '8px' } },
            h('div', { class: 'field' }, h('label', {}, '批大小'), num('batch', { min: 64, step: 64 })),
            h('div', { class: 'field' }, h('label', {}, '学习率'), num('lr', { step: 0.00005, min: 0 }),
              h('span', { class: 'hint' }, '0 为自动：从头训练 2e-4，在已有模型上继续训练 5e-5（实测继续训练时小学习率更好）')),
            h('div', { class: 'field' }, h('label', {}, '探索率 ε'), num('epsilon', { step: 0.005, min: 0, max: 1 })),
            h('div', { class: 'field' }, h('label', {}, '辅助任务权重'), num('aux', { step: 0.25, min: 0, max: 4 }),
              h('span', { class: 'hint' }, '训练时让网络顺便预测其他三家的手牌（对战时不用）。实测明显更强；0 为关闭')),
            h('div', { class: 'field' }, h('label', {}, '对手池比例'), num('league', { step: 0.1, min: 0, max: 1 }),
              h('span', { class: 'hint' }, '这部分对局的对手由早先版本的模型来打。默认 0（纯自对弈）：实测纯自对弈每小时样本多一倍，效果更好')),
            h('div', { class: 'field' }, h('label', {}, '评估间隔（分钟）'), num('eval_interval', { min: 1 })),
            h('div', { class: 'field' }, h('label', {}, '每次评估局数'), num('eval_rounds', { min: 50, step: 50 })),
            h('div', { class: 'field' }, h('label', {}, '最长训练（小时，0 不限）'), num('max_hours', { min: 0, step: 0.5 })))),
        busyRun() ? h('div', { class: 'hint', style: { fontSize: '12.5px', color: 'var(--brass-hi)' } }, `训练「${busyRun()}」正在运行。显卡和内存只够同时跑一个训练，要开新训练请先停止它。`) : null,
        h('button', { class: 'btn primary', onclick: startRun, disabled: !!busyRun() }, '开始训练'),
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
    if (+form.actors > lim.max_actors) { form.actors = lim.max_actors; msgs.push(`自对弈进程数已限制为 ${lim.max_actors}`); }
    if (+form.actors * +form.envs > lim.max_total_envs) {
      form.envs = Math.max(16, Math.floor(lim.max_total_envs / form.actors));
      msgs.push(`每进程并行对局已限制为 ${form.envs}`);
    }
    return msgs;
  }

  async function startRun() {
    const msgs = clampForm();
    if (msgs.length) { renderForm(); toast(`${msgs.join('，')}（受显存和内存限制）`); }
    const body = { ...form, eval_interval: Math.round((+form.eval_interval || 10) * 60) };
    if (!body.base) delete body.base;
    try {
      const r = await api.post('/api/train/start', body);
      toast(`已开始训练 ${r.name}`);
      st.sel = r.name;
      st.rows = [];
      st.since = 0;
      st.runs = await api.get('/api/train/runs');
      renderRuns();
      loadRun(r.name, true);
    } catch (e) { toast(e.message, 'err'); }
  }

  // ------------------------------------------------------------ runs
  function renderRuns() {
    if (!st.runs.length) {
      right.replaceChildren(h('h2', {}, '训练记录'), h('div', { class: 'chart-empty' }, '还没有训练。左侧选择模型大小后点“开始训练”，这里会实时显示胜率变化。'));
      return;
    }
    const tabs = h('div', { class: 'runs-bar' });
    for (const r of st.runs) {
      tabs.appendChild(h('button', { class: 'run-tab' + (r.name === st.sel ? ' on' : ''), onclick: () => { st.sel = r.name; st.rows = []; st.since = 0; renderRuns(); loadRun(r.name, true); } },
        h('span', { class: 'dot' + (r.status === 'running' ? ' on' : '') }), r.name));
    }
    const run = st.runs.find((r) => r.name === st.sel) || st.runs[0];
    st.sel = run.name;
    const running = run.status === 'running';
    const head = h('div', { style: { display: 'flex', alignItems: 'center', gap: '10px', flexWrap: 'wrap', marginBottom: '12px' } },
      h('h2', { style: { margin: 0 } }, run.name),
      h('span', { class: 'tag ' + (running ? 'on' : 'off') }, running ? '训练中' : '已停止'),
      h('span', { class: 'dim', style: { fontSize: '13px' } }, `${run.preset || ''} ${run.params ? (run.params / 1e6).toFixed(2) + 'M 参数' : ''} ${run.device ? '· ' + run.device : ''}`),
      h('span', { class: 'spacer', style: { flex: 1 } }),
      running ? stopButtons(run)
        : h('button', { class: 'btn small', onclick: () => resumeRun(run), disabled: !!busyRun() }, '继续训练'));
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
    mk('win', '每局胜率（头游所在方获胜）', legend, true, { yMin: 0, yMax: 1, refLine: 0.5, yFormat: (v) => `${Math.round(v * 100)}%` });
    mk('score', '平均每局升级数', legend, false, { refLine: 0, yFormat: (v) => (v >= 0 ? '+' : '') + v.toFixed(1) });
    mk('loss', '训练损失', null, false, { yFormat: (v) => v.toFixed(3) });
    mk('fps', '自对弈速度（样本/秒）', null, false, { yFormat: (v) => fmtFrames(v) });
    mk('rps', '每秒完成对局数', null, false, { yFormat: (v) => v.toFixed(0) });
    const evalTable = h('div', { id: 'evaltable' });
    const ck = h('div', { class: 'result-list' });
    for (const f of run.checkpoints || []) {
      const what = f === 'model.pt' ? '最新模型，每 5 分钟自动保存。开桌时在“AI 模型”里选「训练 ' + run.name + '：model.pt」即可用它对战'
        : `第 ${f.replace('model_', '').replace('k.pt', '')} 千样本时的快照（每小时保留一份，可用来比较不同阶段的水平）`;
      ck.appendChild(h('div', { class: 'result-row', style: { gridTemplateColumns: 'minmax(8em, auto) 1fr auto' } },
        h('span', { class: 'pl', style: { fontFamily: 'var(--num)', fontWeight: 600 } }, f),
        h('span', { class: 'rd', style: { color: 'var(--text-2)' } }, what),
        h('button', { class: 'btn small', title: '复制到模型库并起个名字', onclick: () => exportModel(run.name, f) }, '加入模型库')));
    }
    const log = h('pre', { class: 'log', id: 'trainlog' }, '');
    const det = h('details', { open: st.logOpen, ontoggle: (e) => { st.logOpen = e.target.open; if (st.logOpen) loadLog(); } }, h('summary', {}, '训练日志'), log);
    right.replaceChildren(tabs, head, statsBox, charts,
      h('h3', { style: { fontSize: '14px', margin: '16px 0 6px' } }, '评估记录'),
      h('div', { class: 'dim', style: { fontSize: '12px', margin: '0 0 6px' } }, '复式评估：每副牌打两遍、双方交换座位，每次评估用同一批牌。“升级”为平均每局升级数，± 为标准误。'),
      evalTable,
      h('h3', { style: { fontSize: '14px', margin: '16px 0 6px' } }, '这个训练输出的模型'),
      h('div', { class: 'dim', style: { fontSize: '12.5px', marginBottom: '8px' } }, `保存在 ${st.info.home}${st.info.home.includes('\\') ? '\\' : '/'}runs${st.info.home.includes('\\') ? '\\' : '/'}${run.name}。另有 ckpt_latest.pt（含优化器，用于“继续训练”）。`),
      ck.childNodes.length ? ck : h('div', { class: 'dim', style: { fontSize: '13px' } }, '训练开始约 5 分钟后会自动保存第一个模型。'),
      det);
    drawAll();
  }

  function stopButtons(run) {
    const since = st.stopping[run.name];
    if (!since) return h('button', { class: 'btn small danger', onclick: () => stopRun(run.name) }, '停止训练');
    const waited = (Date.now() - since) / 1000;
    return h('span', { style: { display: 'inline-flex', gap: '8px', alignItems: 'center' } },
      h('span', { class: 'dim', style: { fontSize: '13px' } }, '正在保存并停止…'),
      waited > 20 ? h('button', { class: 'btn small danger', onclick: () => forceStop(run.name) }, '强制结束') : null);
  }
  async function stopRun(name) {
    try {
      await api.post('/api/train/stop', { name });
      st.stopping[name] = Date.now();
      toast('正在保存模型并停止；1 分钟内没有响应会自动强制结束');
      renderRuns();
    } catch (e) { toast(e.message, 'err'); }
  }
  async function forceStop(name) {
    try { await api.post('/api/train/stop', { name, force: true }); toast('已强制结束（最近一次自动保存的模型仍然可用）'); setTimeout(refreshRuns, 800); } catch (e) { toast(e.message, 'err'); }
  }
  async function resumeRun(run) {
    try {
      const r = await api.post('/api/train/start', { name: run.name, preset: run.preset || form.preset, actors: form.actors, envs: form.envs, batch: form.batch, device: form.device, actor_device: form.actor_device, eval_interval: Math.round(form.eval_interval * 60), eval_rounds: form.eval_rounds, base: run.base ? undefined : undefined });
      toast(`继续训练 ${r.name}`);
      setTimeout(refreshRuns, 1500);
    } catch (e) { toast(e.message, 'err'); }
  }
  function exportModel(run, file) {
    const input = h('input', { type: 'text', value: `${run}-${file.replace('.pt', '')}` });
    const box = h('div', {}, h('h2', {}, '加入模型库'), h('div', { class: 'sub' }, '起个名字，之后开桌时可以选择它'), input);
    let m;
    box.appendChild(h('div', { class: 'actions' }, h('button', { class: 'btn', onclick: () => m.close() }, '取消'),
      h('button', { class: 'btn primary', onclick: async () => {
        try { await api.post('/api/train/export', { run, file, as: input.value }); toast('已加入模型库'); m.close(); st.models = await api.get('/api/models'); renderForm(); } catch (e) { toast(e.message, 'err'); }
      } }, '保存')));
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
      if (el) { el.textContent = r.log || '（暂无日志）'; el.scrollTop = el.scrollHeight; }
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
        stat('训练样本', lt ? fmtFrames(lt.frames) : '0'),
        stat('速度', lt ? `${fmtFrames(lt.fps)}/s` : '—'),
        stat('训练时长', lt ? fmtDur(lt.train_seconds) : '—'),
        stat('对规则 AI 胜率', le && le.rule ? `${(le.rule.win_rate * 100).toFixed(1)}%` : '—'));
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
    empty(C.win, ws.length ? null : '第一次评估会在训练开始约 10 分钟后完成（可在“评估间隔”调整）');
    C.win.set(ws);
    const ss = evalSeries('score');
    empty(C.score, ss.length ? null : '等待评估结果');
    C.score.set(ss);
    const one = (key) => [{ name: '', color: SERIES_COLORS[0], points: trains.filter((r) => r[key] != null && r.updates !== 0).map((r) => ({ x: r.frames, y: r[key] })) }];
    const ls = one('loss');
    empty(C.loss, ls[0].points.length ? null : '等待训练数据');
    C.loss.set(ls);
    const fs = one('fps');
    empty(C.fps, fs[0].points.length ? null : '等待训练数据');
    C.fps.set(fs);
    const rs = one('rounds_per_sec');
    empty(C.rps, rs[0].points.length ? null : '等待训练数据');
    C.rps.set(rs);
    const tb = document.getElementById('evaltable');
    if (tb) {
      if (!evals.length) { tb.replaceChildren(h('div', { class: 'dim', style: { fontSize: '13px' } }, '暂无评估')); }
      else {
        const t = h('table', { class: 'evals' }, h('tr', {}, h('th', {}, '样本数'), ...OPP.map((o) => h('th', {}, `${o.name} 胜率 / 升级`))));
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
