//! The Q-network forward pass (GuandanNet's inference path) in plain Rust.
//!
//! Same computation as python/guandan/npnet.py: an LSTM over the move history, an MLP over
//! [state, lstm output] giving z, and a Q head over [z, action] for every legal move. The
//! inputs (state, history, action features) are small integer vectors that are mostly zero,
//! so the layers that read them skip the zeros; everything else is dense float32.

use std::collections::HashMap;

use guandan_core::features::{ACTION_DIM, HIST_DIM, HIST_LEN, STATE_DIM};

/// A linear layer stored both ways: `w` row-major [out][in] for dense inputs (dot products)
/// and `wt` [in][out] for sparse inputs (add the rows of the non-zero inputs).
struct Lin {
    n_in: usize,
    n_out: usize,
    w: Vec<f32>,
    wt: Vec<f32>,
    b: Vec<f32>,
}

impl Lin {
    fn new(w: Vec<f32>, b: Vec<f32>, n_out: usize, n_in: usize) -> Lin {
        let mut wt = vec![0f32; n_in * n_out];
        for o in 0..n_out {
            for i in 0..n_in {
                wt[i * n_out + o] = w[o * n_in + i];
            }
        }
        Lin { n_in, n_out, w, wt, b }
    }

    /// y = W x + b for a dense x.
    fn dense(&self, x: &[f32], y: &mut [f32]) {
        for o in 0..self.n_out {
            y[o] = self.b[o] + dot(&self.w[o * self.n_in..(o + 1) * self.n_in], x);
        }
    }

    /// y += W[:, off..off+len(x)] x, skipping zeros of x.
    fn add_sparse(&self, x: &[f32], off: usize, y: &mut [f32]) {
        for (i, &v) in x.iter().enumerate() {
            if v != 0.0 {
                axpy(v, &self.wt[(off + i) * self.n_out..(off + i + 1) * self.n_out], y);
            }
        }
    }

    /// y += W[:, off..off+len(x)] x for a dense x (row by row of the transposed weights).
    fn add_dense_cols(&self, x: &[f32], off: usize, y: &mut [f32]) {
        self.add_sparse(x, off, y)
    }
}

#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    // 8 independent sums so that the compiler can vectorize
    let mut s = [0f32; 8];
    let n = a.len().min(b.len());
    let (ca, ra) = a[..n].split_at(n - n % 8);
    let (cb, rb) = b[..n].split_at(n - n % 8);
    for (x, y) in ca.chunks_exact(8).zip(cb.chunks_exact(8)) {
        for k in 0..8 {
            s[k] += x[k] * y[k];
        }
    }
    let mut t: f32 = s.iter().sum();
    for (x, y) in ra.iter().zip(rb) {
        t += x * y;
    }
    t
}

#[inline]
fn axpy(a: f32, x: &[f32], y: &mut [f32]) {
    for (yi, xi) in y.iter_mut().zip(x) {
        *yi += a * xi;
    }
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    0.5 * ((0.5 * x).tanh() + 1.0)
}

fn relu(x: &mut [f32]) {
    for v in x {
        if *v < 0.0 {
            *v = 0.0;
        }
    }
}

pub struct Net {
    l: usize,
    w_ih: Lin,           // [4L][HIST_DIM]
    w_hh: Lin,           // [4L][L]
    b_lstm: Vec<f32>,    // bias_ih + bias_hh
    state_mlp: Vec<Lin>, // ReLU after every layer
    head: Vec<Lin>,      // ReLU after every layer but the last
    pub config: String,
}

pub fn f16_to_f32(h: u16) -> f32 {
    half::f16::from_bits(h).to_f32()
}

fn read_tensors(bytes: &[u8]) -> Result<(HashMap<String, (Vec<usize>, Vec<f32>)>, serde_json::Value), String> {
    if bytes.len() < 8 {
        return Err("模型文件太短".into());
    }
    let n = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    if 8 + n > bytes.len() {
        return Err("模型文件不完整".into());
    }
    let header: serde_json::Value = serde_json::from_slice(&bytes[8..8 + n]).map_err(|e| format!("模型文件头错误：{e}"))?;
    let data = &bytes[8 + n..];
    let mut out = HashMap::new();
    let obj = header.as_object().ok_or("模型文件头错误")?;
    for (k, v) in obj {
        if k == "__metadata__" {
            continue;
        }
        let dtype = v["dtype"].as_str().unwrap_or("");
        let shape: Vec<usize> = v["shape"].as_array().ok_or("shape")?.iter().map(|x| x.as_u64().unwrap_or(0) as usize).collect();
        let offs = v["data_offsets"].as_array().ok_or("offsets")?;
        let (a, b) = (offs[0].as_u64().unwrap_or(0) as usize, offs[1].as_u64().unwrap_or(0) as usize);
        if b > data.len() || a > b {
            return Err("模型文件数据不完整".into());
        }
        let raw = &data[a..b];
        let vals: Vec<f32> = match dtype {
            "F16" => raw.chunks_exact(2).map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect(),
            "F32" => raw.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
            _ => return Err(format!("不支持的数据类型 {dtype}")),
        };
        if vals.len() != shape.iter().product::<usize>() {
            return Err(format!("{k} 的大小不对"));
        }
        out.insert(k.clone(), (shape, vals));
    }
    let meta = obj.get("__metadata__").cloned().unwrap_or(serde_json::Value::Null);
    Ok((out, meta))
}

/// Linear layers under `prefix` (nn.Sequential, possibly nested) in forward order.
fn linears(t: &HashMap<String, (Vec<usize>, Vec<f32>)>, prefix: &str) -> Result<Vec<Lin>, String> {
    let mut keys: Vec<(Vec<u32>, String)> = t
        .keys()
        .filter(|k| k.starts_with(&format!("{prefix}.")) && k.ends_with(".weight"))
        .map(|k| {
            let base = k[..k.len() - ".weight".len()].to_string();
            let nums = base[prefix.len()..].split('.').filter_map(|x| x.parse::<u32>().ok()).collect();
            (nums, base)
        })
        .collect();
    keys.sort();
    let mut out = vec![];
    for (_, base) in keys {
        let (ws, w) = t.get(&format!("{base}.weight")).ok_or("weight")?;
        let (_, b) = t.get(&format!("{base}.bias")).ok_or_else(|| format!("{base}.bias 缺失"))?;
        out.push(Lin::new(w.clone(), b.clone(), ws[0], ws[1]));
    }
    if out.is_empty() {
        return Err(format!("模型缺少 {prefix}"));
    }
    Ok(out)
}

impl Net {
    pub fn load(bytes: &[u8]) -> Result<Net, String> {
        let (t, meta) = read_tensors(bytes)?;
        let get = |k: &str| t.get(k).ok_or_else(|| format!("模型缺少 {k}"));
        let (sih, wih) = get("lstm.weight_ih_l0")?;
        let (shh, whh) = get("lstm.weight_hh_l0")?;
        let (_, bih) = get("lstm.bias_ih_l0")?;
        let (_, bhh) = get("lstm.bias_hh_l0")?;
        let l = shh[1];
        if sih[1] != HIST_DIM || sih[0] != 4 * l {
            return Err("模型与当前程序的输入格式不一致".into());
        }
        let state_mlp = linears(&t, "state_mlp")?;
        let head = linears(&t, "head")?;
        if state_mlp[0].n_in != STATE_DIM + l || head[0].n_in != state_mlp.last().unwrap().n_out + ACTION_DIM {
            return Err("模型与当前程序的输入格式不一致".into());
        }
        Ok(Net {
            l,
            w_ih: Lin::new(wih.clone(), vec![0.0; 4 * l], 4 * l, HIST_DIM),
            w_hh: Lin::new(whh.clone(), vec![0.0; 4 * l], 4 * l, l),
            b_lstm: bih.iter().zip(bhh).map(|(a, b)| a + b).collect(),
            state_mlp,
            head,
            config: meta.get("config").and_then(|c| c.as_str()).unwrap_or("{}").to_string(),
        })
    }

    /// z from the state features [STATE_DIM] and the history [HIST_LEN][HIST_DIM].
    pub fn encode_state(&self, state: &[u8], hist: &[u8]) -> Vec<f32> {
        let l = self.l;
        let mut h = vec![0f32; l];
        let mut c = vec![0f32; l];
        let mut g = vec![0f32; 4 * l];
        let mut x = vec![0f32; HIST_DIM];
        let mut hh = vec![0f32; 4 * l];
        for t in 0..HIST_LEN {
            for (i, v) in hist[t * HIST_DIM..(t + 1) * HIST_DIM].iter().enumerate() {
                x[i] = *v as f32;
            }
            g.copy_from_slice(&self.b_lstm);
            self.w_ih.add_sparse(&x, 0, &mut g);
            self.w_hh.dense(&h, &mut hh); // bias of w_hh is zero (folded into b_lstm)
            for k in 0..4 * l {
                g[k] += hh[k];
            }
            // PyTorch gate order: input, forget, cell, output
            for k in 0..l {
                let i = sigmoid(g[k]);
                let f = sigmoid(g[l + k]);
                let gg = g[2 * l + k].tanh();
                let o = sigmoid(g[3 * l + k]);
                c[k] = f * c[k] + i * gg;
                h[k] = o * c[k].tanh();
            }
        }
        let s: Vec<f32> = state.iter().map(|&v| v as f32).collect();
        let first = &self.state_mlp[0];
        let mut y = first.b.clone();
        first.add_sparse(&s, 0, &mut y);
        first.add_dense_cols(&h, STATE_DIM, &mut y);
        relu(&mut y);
        for lin in &self.state_mlp[1..] {
            let mut out = vec![0f32; lin.n_out];
            lin.dense(&y, &mut out);
            relu(&mut out);
            y = out;
        }
        y
    }

    /// Q of every action (rows of `actions`, ACTION_DIM each) for the state encoded as z.
    pub fn score(&self, z: &[f32], actions: &[u8]) -> Vec<f32> {
        let n = actions.len() / ACTION_DIM;
        let first = &self.head[0];
        let mut base = first.b.clone();
        first.add_dense_cols(z, 0, &mut base);
        let zl = z.len();
        let mut out = Vec::with_capacity(n);
        let mut a = vec![0f32; ACTION_DIM];
        for k in 0..n {
            for (i, v) in actions[k * ACTION_DIM..(k + 1) * ACTION_DIM].iter().enumerate() {
                a[i] = *v as f32;
            }
            let mut y = base.clone();
            first.add_sparse(&a, zl, &mut y);
            let last = self.head.len() - 1;
            if last > 0 {
                relu(&mut y);
            }
            for (j, lin) in self.head[1..].iter().enumerate() {
                let mut o = vec![0f32; lin.n_out];
                lin.dense(&y, &mut o);
                if j + 1 < last {
                    relu(&mut o);
                }
                y = o;
            }
            out.push(y[0]);
        }
        out
    }
}
