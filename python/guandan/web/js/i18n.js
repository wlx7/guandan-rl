// Interface language: Chinese (default) or English.
//
// The Chinese text stays in the code and is the lookup key; English comes from the dictionaries
// below. T('中文', {n: 3}) translates interface text (placeholders {name} are filled in both
// languages); tr(s) translates text that comes from the server (errors, seat names, move names,
// model labels), first by exact match, then by the patterns at the end of this file.
// The choice is stored per browser; switching reloads the page (seats and tables are kept).
import MAIN from './lang_main.js';
import TABLE from './lang_table.js';
import TRAIN from './lang_train.js';

const KEY = 'gd_lang';
function initial() {
  try {
    const v = localStorage.getItem(KEY);
    if (v === 'en' || v === 'zh') return v;
  } catch {}
  return 'zh';
}
export const lang = initial();
export const isEn = lang === 'en';
document.documentElement.lang = isEn ? 'en' : 'zh-CN';
// the Android app shows its notification in the same language
try { if (window.GuandanApp && window.GuandanApp.setLang) window.GuandanApp.setLang(lang); } catch {}

export function setLang(l) {
  try { localStorage.setItem(KEY, l); } catch {}
  location.reload();
}

// ------------------------------------------------------------------ shared words and server text
const CORE = {
  // seats, teams, devices
  '南': 'S', '东': 'E', '北': 'N', '西': 'W',
  '南家': 'South', '东家': 'East', '北家': 'North', '西家': 'West',
  '南北': 'N–S', '东西': 'E–W',
  '电脑': 'computer', '手机': 'phone',
  '房主': 'Host', '有人': 'Someone', '我': 'Me', '你': 'You', '玩家': 'Player',
  // move kinds and cards (from the rules engine)
  '不出': 'Pass', '单张': 'Single', '对子': 'Pair', '三同张': 'Triple', '三张': 'Triple', '三带二': 'Full house',
  '顺子': 'Straight', '三连对': 'Tube', '钢板': 'Plate', '炸弹': 'Bomb', '同花顺': 'Straight flush',
  '天王炸': 'Joker bomb', '还贡': 'Return tribute', '配': 'Wild',
  '小王': 'Small joker', '大王': 'Big joker',
  // scoring options (scoring.py / android room.rs)
  '标准倍数': 'Standard multiplier',
  '6 张及以上炸弹、同花顺、天王炸每出一个翻一倍，封顶 16 倍': 'Each bomb of 6+ cards, straight flush or joker bomb doubles the score, up to 16×',
  '全炸翻倍': 'Every bomb doubles',
  '每出一个炸弹（含同花顺、天王炸）翻一倍，封顶 256 倍': 'Every bomb (incl. straight flushes and joker bombs) doubles the score, up to 256×',
  // model sizes (model.py)
  '最快，适合 CPU 调试': 'Fastest, for debugging on a CPU',
  '轻量，CPU 也能训练': 'Light, trains on a CPU too',
  '推荐：容量与体积的最佳平衡（约 2.5M 参数，fp16 约 5MB）': 'Recommended: best balance of capacity and size (about 2.5M parameters, about 5 MB in fp16)',
  '更大容量，需要 GPU 与更长训练': 'More capacity, needs a GPU and longer training',
  '实验用，收益递减': 'Experimental, diminishing returns',
  // model groups (paths.py / android app.rs)
  '内置': 'Built-in', '模型库': 'Model library', '训练': 'Training',
  // server errors (python server / android server)
  '训练中心只有房主可以使用': 'Only the host can use the training center',
  '房主密码至少 4 位': 'The host password needs at least 4 characters',
  '无效的模型': 'Invalid model',
  '内置模型不能隐藏': 'The built-in model cannot be hidden',
  '找不到基准模型': 'Base model not found',
  '只有房主或开这张桌的人可以管理这张牌桌': 'Only the host or whoever opened this table can manage it',
  '房主没有开放其他设备开桌，请让房主开桌后扫码加入': 'The host does not let other devices open tables. Ask the host to open one, then scan its QR code to join',
  '没有可用的模型，请先训练或下载模型': 'No model available. Train one or download one first',
  '这张牌桌已经关闭了': 'This table has been closed',
  '这张牌桌不存在或已经关闭': 'This table does not exist or has been closed',
  '牌桌已经开始或不存在': 'The table has already started or does not exist',
  '房主关闭了这张牌桌': 'The host closed this table',
  '当前没有牌桌': 'There is no table right now',
  '座位已失效，请重新加入': 'Your seat is no longer valid. Please join again',
  '牌桌已关闭': 'The table is closed',
  '这个版本只能对战，不能训练：训练需要 PyTorch。请用完整安装版（install_windows.bat）训练。': 'This version can only play, not train: training needs PyTorch. Use the full install (install_windows.bat) to train.',
  '没有这个训练': 'No such training run',
  '找不到模型文件': 'Model file not found',
  '只能加宽到更大的模型': 'A model can only be widened to a larger size',
  '没有空余的玩家座位': 'No free player seat',
  '座位不存在': 'No such seat',
  '该座位是 AI': 'That seat is an AI',
  '该座位已有人': 'That seat is taken',
  '还贡牌必须是 10 或以下的牌': 'The returned card must be a 10 or lower',
  '请先选择一张牌还贡': 'Select a card to return first',
  '你是首家，必须出牌': 'You lead, so you must play',
  '管不上上家的牌': 'That does not beat the last play',
  '不是有效牌型': 'Not a valid combination',
  '实战模式不能使用 AI 提示': 'AI hints are not available in real mode',
  '还没轮到你': 'It is not your turn yet',
  '只有房主或开这张桌的人可以结束牌局': 'Only the host or whoever opened this table can end the game',
  '观战时不能暂停牌局': 'Spectators cannot pause the game',
  '密码错误次数太多，请 5 分钟后再试': 'Too many wrong passwords. Try again in 5 minutes',
  '房主还没有设置房主密码': 'The host has not set a host password',
  '房主密码不对': 'Wrong host password',
  '安卓版只能对战，训练请用电脑版': 'The Android app can only play. Use the computer version to train',
  '服务启动时出错': 'The server failed to start',
  '模型与当前程序的输入格式不一致': 'The model does not match this program\'s input format',
  // what need_admin() refuses ("只有房主可以…")
  '修改权限设置': 'change the permission settings',
  '查看设备屏幕信息': 'view device screen info',
  '整理模型列表': 'organize the model list',
  '关闭所有牌桌': 'close all tables',
};

const EN = Object.assign(Object.create(null), CORE, MAIN, TABLE, TRAIN);

function fill(s, vars) {
  return vars ? s.replace(/\{(\w+)\}/g, (m, k) => (k in vars ? String(vars[k]) : m)) : s;
}

// interface text
export function T(zh, vars) {
  const s = isEn && EN[zh] !== undefined ? EN[zh] : zh;
  return fill(s, vars);
}

// ------------------------------------------------------------------ server text
const RANK = { SJ: 'SJ', BJ: 'BJ' };
const word = (s) => (EN[s] !== undefined ? EN[s] : s);
const card = (s) => word(s);  // ♠7 stays, 小王 / 大王 are translated
const PATTERNS = [
  [/^(单张|对子|三同张|三带二|顺子|三连对|钢板|同花顺) (.+)$/, (_, k, r) => `${word(k)} ${RANK[r] || r}`],
  [/^(\d+)张(.+)炸弹$/, (_, n, r) => `${n}-card bomb of ${RANK[r] || r}`],
  [/^还贡 (.+)$/, (_, c) => `Return ${card(c)}`],
  [/^AI ([南东北西])$/, (_, d) => `AI ${word(d + '家')}`],
  [/^玩家(\d+)$/, (_, n) => `Player ${n}`],
  [/^(.+) 结束了这张牌桌$/, (_, who) => `${tr(who)} ended this table`],
  [/^只有房主可以(.+)$/, (_, what) => `Only the host can ${word(what)}`],
  [/^最多同时开 (\d+) 桌，请先在首页关掉不用的牌桌$/, (_, n) => `At most ${n} tables at a time. Close one you no longer use on the home page first`],
  [/^训练「(.+)」正在运行。显卡和内存只够跑一个训练，请先停止它再开始新的训练。$/, (_, n) => `Training "${n}" is running. The GPU and memory can only handle one training run: stop it before starting a new one.`],
  [/^内置模型 v(.+)（推荐）$/, (_, v) => `Built-in model v${v} (recommended)`],
  [/^内置：(.+)$/, (_, n) => `Built-in: ${n}`],
  [/^训练 (.+) · (.+) 样本快照$/, (_, n, f) => `Training ${n} · snapshot at ${f} samples`],
  [/^(.+) · (.+) 样本快照$/, (_, n, f) => `${n} · snapshot at ${f} samples`],
  [/^训练 (.+)$/, (_, n) => `Training ${n}`],
  [/^读不到模型文件 (.+)：(.*)$/, (_, f, e) => `Cannot read the model file ${f}: ${e}`],
  [/^端口 (\d+) 起都被占用了$/, (_, p) => `All ports from ${p} up are in use`],
];

export function tr(s) {
  if (!isEn || typeof s !== 'string' || !s) return s;
  if (EN[s] !== undefined) return EN[s];
  for (const [re, f] of PATTERNS) {
    const m = s.match(re);
    if (m) return f(...m);
  }
  return s;
}
