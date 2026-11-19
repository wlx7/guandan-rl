//! Pure-Rust neural network inference, incremental-LSTM rollout search, and root-cached SMC
//! action-likelihood world filtering (`FastNet`).
//!
//! Zero external ML dependencies: runs identically across Python (`_core.FastNet`), the
//! PyTorch-free portable Windows `.exe` (`NumpyNet` -> `FastNet`), and Android (`guandan_server`).
//!
//! Key speedups (~200-500x faster than naive CPU sequence Rollouts):
//! 1. Binary `{0, 1}` sparse input layers (`STATE_DIM`, `HIST_DIM`, `ACTION_DIM`) add rows of
//!    transposed weights directly without `u8 -> f32` conversion or multiplications.
//! 2. ReLU-sparse hidden layers skip zero activations (~55-65% zeros after ReLU) and stream
//!    transposed rows via contiguous 8-way SIMD `axpy`.
//! 3. Incremental `T=1` lazy per-seat LSTM during rollouts: root `(h, c)` for all 4 seats is
//!    computed once before world sampling; during rollouts, forced moves (`legal_moves == 1`,
//!    ~40% of turns) skip network evaluation completely, finished seats never update again,
//!    and active seats catch up `T=1` per move.
//! 4. Public-history-invariant SMC world weighting: `g.history` is identical across all sampled
//!    worlds, so historical LSTM states and public `state_mlp[0]` contributions are computed
//!    once at the root before scoring candidate worlds in parallel across threads.

use crate::cards::*;
use crate::endgame::Solver;
use crate::features::*;
use crate::game::*;
use crate::moves::*;

/// Linear layer stored both transposed (`wt: [n_in * n_out]` for sparse / ReLU-sparse row
/// accumulation) and row-major (`w: [n_out * n_in]` for dense small-output dot products).
#[derive(Clone)]
pub struct FastLin {
    pub n_in: usize,
    pub n_out: usize,
    pub w: Vec<f32>,
    pub wt: Vec<f32>,
    pub b: Vec<f32>,
}

impl FastLin {
    pub fn new(w: Vec<f32>, b: Vec<f32>, n_out: usize, n_in: usize) -> Self {
        assert_eq!(w.len(), n_out * n_in);
        assert_eq!(b.len(), n_out);
        let mut wt = vec![0f32; n_in * n_out];
        for o in 0..n_out {
            for i in 0..n_in {
                wt[i * n_out + o] = w[o * n_in + i];
            }
        }
        FastLin { n_in, n_out, w, wt, b }
    }

    /// `y += W[:, off + idx]` for binary `{0, 1}` feature slices.
    #[inline]
    pub fn add_bits(&self, bits: &[u8], off: usize, y: &mut [f32]) {
        let n_out = self.n_out;
        for (i, &v) in bits.iter().enumerate() {
            if v != 0 {
                let row = &self.wt[(off + i) * n_out..(off + i + 1) * n_out];
                add_assign_slice(y, row);
            }
        }
    }

    /// `y += W[:, off..off+x.len()] * x`, skipping exact zeros (ideal after ReLU).
    #[inline]
    pub fn add_sparse_f32(&self, x: &[f32], off: usize, y: &mut [f32]) {
        let n_out = self.n_out;
        for (i, &v) in x.iter().enumerate() {
            if v != 0.0 {
                let row = &self.wt[(off + i) * n_out..(off + i + 1) * n_out];
                axpy_slice(v, row, y);
            }
        }
    }

    /// `y = W * x + b` skipping zeros in `x` (fast when `n_out >= 64` and `x` is post-ReLU).
    #[inline]
    pub fn forward_sparse(&self, x: &[f32], y: &mut [f32]) {
        y[..self.n_out].copy_from_slice(&self.b);
        self.add_sparse_f32(x, 0, &mut y[..self.n_out]);
    }

    /// Scalar output (`n_out == 1`) dot product.
    #[inline]
    pub fn dot1(&self, x: &[f32]) -> f32 {
        debug_assert_eq!(self.n_out, 1);
        self.b[0] + dot_slice(&self.w[..self.n_in], &x[..self.n_in])
    }

    /// Dense `y = W * x + b` (used for `belief_logits` where `x` is dense/post-ReLU).
    #[inline]
    pub fn forward_dense(&self, x: &[f32], y: &mut [f32]) {
        for o in 0..self.n_out {
            y[o] = self.b[o] + dot_slice(&self.w[o * self.n_in..(o + 1) * self.n_in], x);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
fn has_avx2_fma() -> bool {
    use std::sync::atomic::{AtomicU8, Ordering};
    static FLAG: AtomicU8 = AtomicU8::new(0);
    let v = FLAG.load(Ordering::Relaxed);
    if v != 0 {
        return v == 2;
    }
    let ok = std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma");
    FLAG.store(if ok { 2 } else { 1 }, Ordering::Relaxed);
    ok
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn add_assign_avx2(y: &mut [f32], x: &[f32]) {
    use std::arch::x86_64::*;
    let n = y.len().min(x.len());
    let yp = y.as_mut_ptr();
    let xp = x.as_ptr();
    let mut i = 0usize;
    while i + 16 <= n {
        let vy0 = _mm256_loadu_ps(yp.add(i));
        let vx0 = _mm256_loadu_ps(xp.add(i));
        _mm256_storeu_ps(yp.add(i), _mm256_add_ps(vy0, vx0));
        let vy1 = _mm256_loadu_ps(yp.add(i + 8));
        let vx1 = _mm256_loadu_ps(xp.add(i + 8));
        _mm256_storeu_ps(yp.add(i + 8), _mm256_add_ps(vy1, vx1));
        i += 16;
    }
    while i + 8 <= n {
        let vy = _mm256_loadu_ps(yp.add(i));
        let vx = _mm256_loadu_ps(xp.add(i));
        _mm256_storeu_ps(yp.add(i), _mm256_add_ps(vy, vx));
        i += 8;
    }
    while i < n {
        *yp.add(i) += *xp.add(i);
        i += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn axpy_avx2(a: f32, x: &[f32], y: &mut [f32]) {
    use std::arch::x86_64::*;
    let n = y.len().min(x.len());
    let yp = y.as_mut_ptr();
    let xp = x.as_ptr();
    let va = _mm256_set1_ps(a);
    let mut i = 0usize;
    while i + 16 <= n {
        let vy0 = _mm256_loadu_ps(yp.add(i));
        let vx0 = _mm256_loadu_ps(xp.add(i));
        _mm256_storeu_ps(yp.add(i), _mm256_fmadd_ps(va, vx0, vy0));
        let vy1 = _mm256_loadu_ps(yp.add(i + 8));
        let vx1 = _mm256_loadu_ps(xp.add(i + 8));
        _mm256_storeu_ps(yp.add(i + 8), _mm256_fmadd_ps(va, vx1, vy1));
        i += 16;
    }
    while i + 8 <= n {
        let vy = _mm256_loadu_ps(yp.add(i));
        let vx = _mm256_loadu_ps(xp.add(i));
        _mm256_storeu_ps(yp.add(i), _mm256_fmadd_ps(va, vx, vy));
        i += 8;
    }
    while i < n {
        *yp.add(i) += a * *xp.add(i);
        i += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let n = a.len().min(b.len());
    let ap = a.as_ptr();
    let bp = b.as_ptr();
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut i = 0usize;
    while i + 16 <= n {
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(ap.add(i)), _mm256_loadu_ps(bp.add(i)), acc0);
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(ap.add(i + 8)), _mm256_loadu_ps(bp.add(i + 8)), acc1);
        i += 16;
    }
    while i + 8 <= n {
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(ap.add(i)), _mm256_loadu_ps(bp.add(i)), acc0);
        i += 8;
    }
    let sum8 = _mm256_add_ps(acc0, acc1);
    let mut tmp = [0f32; 8];
    _mm256_storeu_ps(tmp.as_mut_ptr(), sum8);
    let mut t = tmp.iter().sum::<f32>();
    while i < n {
        t += *ap.add(i) * *bp.add(i);
        i += 1;
    }
    t
}

#[inline]
fn add_assign_slice(y: &mut [f32], x: &[f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2_fma() {
            unsafe { add_assign_avx2(y, x) };
            return;
        }
    }
    let n = y.len().min(x.len());
    let (cy, ry) = y[..n].split_at_mut(n - n % 8);
    let (cx, rx) = x[..n].split_at(n - n % 8);
    for (ys, xs) in cy.chunks_exact_mut(8).zip(cx.chunks_exact(8)) {
        for k in 0..8 {
            ys[k] += xs[k];
        }
    }
    for (yi, &xi) in ry.iter_mut().zip(rx.iter()) {
        *yi += xi;
    }
}

#[inline]
fn axpy_slice(a: f32, x: &[f32], y: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2_fma() {
            unsafe { axpy_avx2(a, x, y) };
            return;
        }
    }
    let n = y.len().min(x.len());
    let (cy, ry) = y[..n].split_at_mut(n - n % 8);
    let (cx, rx) = x[..n].split_at(n - n % 8);
    for (ys, xs) in cy.chunks_exact_mut(8).zip(cx.chunks_exact(8)) {
        for k in 0..8 {
            ys[k] += a * xs[k];
        }
    }
    for (yi, &xi) in ry.iter_mut().zip(rx.iter()) {
        *yi += a * xi;
    }
}

#[inline]
fn dot_slice(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2_fma() {
            return unsafe { dot_avx2(a, b) };
        }
    }
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
    for (&x, &y) in ra.iter().zip(rb.iter()) {
        t += x * y;
    }
    t
}

#[inline]
fn fast_sigmoid(x: f32) -> f32 {
    0.5 * ((0.5 * x).tanh() + 1.0)
}

#[inline]
fn relu_inplace(x: &mut [f32]) {
    for v in x {
        if *v < 0.0 {
            *v = 0.0;
        }
    }
}

#[inline]
fn erff_exact(x: f32) -> f32 {
    let sign = if x < 0.0 { -1.0f32 } else { 1.0f32 };
    let ax = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * ax);
    let poly = (((((1.061_405_4 * t - 1.453_152) * t) + 1.421_413_7) * t - 0.284_496_74) * t + 0.254_829_6) * t;
    sign * (1.0 - poly * (-ax * ax).exp())
}

#[inline]
fn layernorm_gelu_inplace(x: &mut [f32], w: &[f32], b: &[f32]) {
    let n = x.len() as f32;
    let mean = x.iter().sum::<f32>() / n;
    let var = x.iter().map(|&v| { let d = v - mean; d * d }).sum::<f32>() / n;
    let inv_std = 1.0 / (var + 1e-5).sqrt();
    let inv_sqrt2 = std::f32::consts::FRAC_1_SQRT_2;
    for i in 0..x.len() {
        let xn = (x[i] - mean) * inv_std * w[i] + b[i];
        x[i] = 0.5 * xn * (1.0 + erff_exact(xn * inv_sqrt2));
    }
}

#[derive(Clone)]
pub struct OracleHead {
    fc0: FastLin,
    ln0_w: Vec<f32>,
    ln0_b: Vec<f32>,
    fc1: FastLin,
    ln1_w: Vec<f32>,
    ln1_b: Vec<f32>,
    fc2: FastLin,
}

/// Compact representation of one history move: the seat `player` who played it, plus the
/// non-zero indices in `0..136` (`cards108 + kind + key`, at most 14 active bits).
/// Seat-relative index `136 + ((player + 4 - me) & 3)` is added on the fly per seat!
#[derive(Clone, Copy)]
struct CompactHistMove {
    player: u8,
    len: u8,
    nz: [u8; 16],
}

impl CompactHistMove {
    #[inline]
    fn from_move(player: u8, m: &Move) -> Self {
        let mut h = [0u8; 54];
        for &c in &m.cards {
            h[c as usize] += 1;
        }
        let mut nz = [0u8; 16];
        let mut len = 0usize;
        for t in 0..54usize {
            let c = h[t];
            if c >= 1 {
                nz[len] = (2 * t) as u8;
                len += 1;
            }
            if c >= 2 {
                nz[len] = (2 * t + 1) as u8;
                len += 1;
            }
        }
        let ku = m.kind as usize;
        if ku < NUM_KINDS {
            nz[len] = (108 + ku) as u8;
            len += 1;
        }
        let keyu = m.key as usize;
        if keyu < 16 {
            nz[len] = (120 + keyu) as u8;
            len += 1;
        }
        CompactHistMove { player, len: len as u8, nz }
    }
}

/// Reusable per-thread scratch buffers so rollouts and SMC do zero heap allocations per ply.
pub struct FastScratch {
    gates: Vec<f32>,
    s_buf: [u8; STATE_DIM],
    a_buf: [u8; ACTION_DIM],
    mlp_a: Vec<f32>,
    mlp_b: Vec<f32>,
    z_buf: Vec<f32>,
    head_base: Vec<f32>,
    head_a: Vec<f32>,
    head_b: Vec<f32>,
    ora_a: Vec<f32>,
    ora_b: Vec<f32>,
    q_buf: Vec<f32>,
    h_seats: [[f32; 512]; 4], // supports up to L = 512 (xlarge)
    c_seats: [[f32; 512]; 4],
    hist_log: Vec<CompactHistMove>,
}

impl FastScratch {
    pub fn new(net: &FastNet) -> Self {
        let max_mlp = net.state_mlp.iter().map(|l| l.n_out).max().unwrap_or(768);
        let max_head = net.head.iter().map(|l| l.n_out).max().unwrap_or(768);
        let (oa, ob) = match &net.oracle {
            Some(o) => (o.fc0.n_out, o.fc1.n_out),
            None => (0, 0),
        };
        FastScratch {
            gates: vec![0f32; 4 * net.l],
            s_buf: [0u8; STATE_DIM],
            a_buf: [0u8; ACTION_DIM],
            mlp_a: vec![0f32; max_mlp],
            mlp_b: vec![0f32; max_mlp],
            z_buf: vec![0f32; net.z_dim],
            head_base: vec![0f32; net.head[0].n_out],
            head_a: vec![0f32; max_head],
            head_b: vec![0f32; max_head],
            ora_a: vec![0f32; oa],
            ora_b: vec![0f32; ob],
            q_buf: Vec::with_capacity(128),
            h_seats: [[0f32; 512]; 4],
            c_seats: [[0f32; 512]; 4],
            hist_log: Vec::with_capacity(160),
        }
    }
}

#[derive(Clone)]
pub struct FastNet {
    pub l: usize,
    pub z_dim: usize,
    pub hist_len: usize,
    w_ih: FastLin,
    w_hh: FastLin,
    b_lstm: Vec<f32>,
    state_mlp: Vec<FastLin>,
    head: Vec<FastLin>,
    belief: Option<FastLin>,
    oracle: Option<OracleHead>,
}

impl FastNet {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        l: usize,
        hist_len: usize,
        w_ih: Vec<f32>,
        w_hh: Vec<f32>,
        b_lstm: Vec<f32>,
        state_mlp: Vec<(Vec<f32>, Vec<f32>, usize, usize)>,
        head: Vec<(Vec<f32>, Vec<f32>, usize, usize)>,
        belief: Option<(Vec<f32>, Vec<f32>, usize, usize)>,
    ) -> Result<Self, String> {
        if l == 0 || l > 512 {
            return Err(format!("unsupported lstm size {l}"));
        }
        if w_ih.len() != 4 * l * HIST_DIM || w_hh.len() != 4 * l * l || b_lstm.len() != 4 * l {
            return Err("lstm weight shape mismatch".into());
        }
        let s_layers: Vec<FastLin> = state_mlp
            .into_iter()
            .map(|(w, b, no, ni)| FastLin::new(w, b, no, ni))
            .collect();
        let h_layers: Vec<FastLin> = head
            .into_iter()
            .map(|(w, b, no, ni)| FastLin::new(w, b, no, ni))
            .collect();
        if s_layers.is_empty() || h_layers.is_empty() {
            return Err("empty mlp layers".into());
        }
        let z_dim = s_layers.last().unwrap().n_out;
        if s_layers[0].n_in != STATE_DIM + l || h_layers[0].n_in != z_dim + ACTION_DIM || h_layers.last().unwrap().n_out != 1 {
            return Err("mlp input/output dim mismatch".into());
        }
        let b_lin = belief.map(|(w, b, no, ni)| FastLin::new(w, b, no, ni));
        Ok(FastNet {
            l,
            z_dim,
            hist_len: hist_len.clamp(1, HIST_LEN),
            w_ih: FastLin::new(w_ih, vec![0.0; 4 * l], 4 * l, HIST_DIM),
            w_hh: FastLin::new(w_hh, vec![0.0; 4 * l], 4 * l, l),
            b_lstm,
            state_mlp: s_layers,
            head: h_layers,
            belief: b_lin,
            oracle: None,
        })
    }

    /// Attach a 4-hand level-invariant Perfect-Information `OracleHead` (`122 + 408 + z_dim + ACTION_DIM + 1 -> h1 -> h2 -> 1`).
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    pub fn set_oracle(
        &mut self,
        fc0: (Vec<f32>, Vec<f32>, usize, usize),
        ln0: (Vec<f32>, Vec<f32>),
        fc1: (Vec<f32>, Vec<f32>, usize, usize),
        ln1: (Vec<f32>, Vec<f32>),
        fc2: (Vec<f32>, Vec<f32>, usize, usize),
    ) -> Result<(), String> {
        let exp_in = 122 + 408 + self.z_dim + ACTION_DIM + 1;
        if fc0.3 != exp_in || fc1.3 != fc0.2 || fc2.3 != fc1.2 || fc2.2 != 1 {
            return Err(format!("oracle dim mismatch: got ({}, {}, {}, {}), expected in={exp_in}", fc0.3, fc0.2, fc1.2, fc2.2));
        }
        if ln0.0.len() != fc0.2 || ln0.1.len() != fc0.2 || ln1.0.len() != fc1.2 || ln1.1.len() != fc1.2 {
            return Err("oracle layernorm shape mismatch".into());
        }
        self.oracle = Some(OracleHead {
            fc0: FastLin::new(fc0.0, fc0.1, fc0.2, fc0.3),
            ln0_w: ln0.0,
            ln0_b: ln0.1,
            fc1: FastLin::new(fc1.0, fc1.1, fc1.2, fc1.3),
            ln1_w: ln1.0,
            ln1_b: ln1.1,
            fc2: FastLin::new(fc2.0, fc2.1, fc2.2, fc2.3),
        });
        Ok(())
    }

    /// Evaluate 4-hand level-invariant `OracleHead` for `(g, me, chosen_mv, q_lvl)` (in level units).
    /// Requires `sc.s_buf` and `sc.z_buf` to be populated for `(g, me)`.
    #[inline]
    pub fn eval_oracle(&self, g: &Game, me: u8, chosen_mv: &Move, q_lvl: f32, sc: &mut FastScratch) -> f32 {
        let Some(ref ora) = self.oracle else {
            return q_lvl;
        };
        let n0 = ora.fc0.n_out;
        let n1 = ora.fc1.n_out;
        sc.ora_a[..n0].copy_from_slice(&ora.fc0.b);

        // 1. `[0..122]`: own `strength_thermo` (`sc.s_buf[108..230]`)
        ora.fc0.add_bits(&sc.s_buf[108..230], 0, &mut sc.ora_a[..n0]);

        // 2. `[122..530]`: RHO, Partner, LHO level-invariant `strength_thermo` (122) + `summary14` (14)
        let lv = g.level;
        let wt = wild_type(lv) as usize;
        for r in 1..4u8 {
            let p = ((me + r) & 3) as usize;
            let base = 122 + ((r - 1) as usize) * 136;
            let hand = &g.hands[p];
            let mut reg15 = [0u8; 15];
            let mut tot_c = 0u32;
            for (t, &c) in hand.iter().enumerate() {
                tot_c += c as u32;
                if t != wt && c > 0 {
                    reg15[sidx(rank_of(t as u8), lv) as usize] += c;
                }
            }
            let wild = hand[wt];
            for (i, &cnt) in reg15.iter().enumerate() {
                for k in 0..8u8 {
                    if cnt > k {
                        let col = base + i * 8 + (k as usize);
                        add_assign_slice(&mut sc.ora_a[..n0], &ora.fc0.wt[col * n0..(col + 1) * n0]);
                    }
                }
            }
            if wild >= 1 {
                let col = base + 120;
                add_assign_slice(&mut sc.ora_a[..n0], &ora.fc0.wt[col * n0..(col + 1) * n0]);
            }
            if wild >= 2 {
                let col = base + 121;
                add_assign_slice(&mut sc.ora_a[..n0], &ora.fc0.wt[col * n0..(col + 1) * n0]);
            }
            let mut nat_b4 = 0u32;
            let mut nat_b5 = 0u32;
            let mut max_nat = 0u8;
            let mut wild_b5 = 0u32;
            let mut singles = 0u32;
            let mut pairs = 0u32;
            let mut triples = 0u32;
            for &c in &reg15[..13] {
                if c > max_nat {
                    max_nat = c;
                }
                if c >= 4 {
                    nat_b4 += 1;
                }
                if c >= 5 {
                    nat_b5 += 1;
                }
                if c + wild >= 5 {
                    wild_b5 += 1;
                }
                if c == 1 {
                    singles += 1;
                } else if c == 2 {
                    pairs += 1;
                } else if c == 3 {
                    triples += 1;
                }
            }
            let max_rk_cnt = max_nat + wild;
            let sj = reg15[13];
            let bj = reg15[14];
            let sum14 = [
                if sj >= 2 && bj >= 2 { 1.0 } else { 0.0 },
                (nat_b4 as f32) / 3.0,
                (nat_b5 as f32) / 2.0,
                if max_rk_cnt >= 4 { 1.0 } else { 0.0 },
                if max_rk_cnt >= 5 { 1.0 } else { 0.0 },
                if max_rk_cnt >= 6 { 1.0 } else { 0.0 },
                (wild_b5.min(2) as f32) / 2.0,
                (sj as f32) / 2.0,
                (bj as f32) / 2.0,
                (reg15[12] as f32) / 6.0,
                (singles as f32) / 10.0,
                (tot_c as f32) / 27.0,
                (pairs as f32) / 8.0,
                (triples as f32) / 5.0,
            ];
            ora.fc0.add_sparse_f32(&sum14, base + 122, &mut sc.ora_a[..n0]);
        }

        // 3. `[530 .. 530 + z_dim]`: `z`, `[.. + ACTION_DIM]`: `a`, `[last]`: `q_lvl`
        ora.fc0.add_sparse_f32(&sc.z_buf, 530, &mut sc.ora_a[..n0]);
        encode_action(g, me, chosen_mv, &mut sc.a_buf);
        ora.fc0.add_bits(&sc.a_buf, 530 + self.z_dim, &mut sc.ora_a[..n0]);
        ora.fc0.add_sparse_f32(&[q_lvl], 530 + self.z_dim + ACTION_DIM, &mut sc.ora_a[..n0]);

        layernorm_gelu_inplace(&mut sc.ora_a[..n0], &ora.ln0_w, &ora.ln0_b);
        ora.fc1.forward_dense(&sc.ora_a[..n0], &mut sc.ora_b[..n1]);
        layernorm_gelu_inplace(&mut sc.ora_b[..n1], &ora.ln1_w, &ora.ln1_b);
        let delta = ora.fc2.dot1(&sc.ora_b[..n1]);
        (q_lvl + delta).clamp(-3.0, 3.0)
    }

    /// Standalone single-game `OracleHead` evaluation for testing exact parity vs PyTorch.
    pub fn eval_oracle_standalone(&self, g: &Game, move_idx: usize) -> f32 {
        let me = g.current;
        let moves = g.legal_moves();
        if move_idx >= moves.len() {
            return 0.0;
        }
        let mut sc = FastScratch::new(self);
        self.init_root_lstm(&g.history, 1 << me, &mut sc.h_seats, &mut sc.c_seats, &mut sc.gates);
        encode_state(g, me, &mut sc.s_buf);
        let h_me = sc.h_seats[me as usize];
        self.state_to_z(&h_me[..self.l], &mut sc);
        self.score_moves(g, me, &moves, &mut sc);
        let q_lvl = sc.q_buf[move_idx] * 3.0;
        self.eval_oracle(g, me, &moves[move_idx], q_lvl, &mut sc)
    }

    /// Step `(h, c)` one LSTM tick on an all-zero padding row (`x = 0`).
    #[inline]
    fn lstm_step_zero(&self, h: &mut [f32], c: &mut [f32], gates: &mut [f32]) {
        let l = self.l;
        gates.copy_from_slice(&self.b_lstm);
        self.w_hh.add_sparse_f32(&h[..l], 0, gates);
        for k in 0..l {
            let i = fast_sigmoid(gates[k]);
            let f = fast_sigmoid(gates[l + k]);
            let gg = gates[2 * l + k].tanh();
            let o = fast_sigmoid(gates[3 * l + k]);
            let ck = f * c[k] + i * gg;
            c[k] = ck;
            h[k] = o * ck.tanh();
        }
    }

    /// Step `(h, c)` one LSTM tick on `cm` from `seat`'s perspective.
    #[inline]
    fn lstm_step_compact(&self, seat: u8, cm: &CompactHistMove, h: &mut [f32], c: &mut [f32], gates: &mut [f32]) {
        let l = self.l;
        let n_out = 4 * l;
        gates.copy_from_slice(&self.b_lstm);
        for &idx in &cm.nz[..cm.len as usize] {
            let u = idx as usize;
            add_assign_slice(gates, &self.w_ih.wt[u * n_out..(u + 1) * n_out]);
        }
        let rel_idx = 136 + (((cm.player + 4 - seat) & 3) as usize);
        add_assign_slice(gates, &self.w_ih.wt[rel_idx * n_out..(rel_idx + 1) * n_out]);
        self.w_hh.add_sparse_f32(&h[..l], 0, gates);
        for k in 0..l {
            let i = fast_sigmoid(gates[k]);
            let f = fast_sigmoid(gates[l + k]);
            let gg = gates[2 * l + k].tanh();
            let o = fast_sigmoid(gates[3 * l + k]);
            let ck = f * c[k] + i * gg;
            c[k] = ck;
            h[k] = o * ck.tanh();
        }
    }

    /// Compute exact `(h, c)` for `seats_mask` (bit `s` = compute seat `s`) on `g.history`
    /// matching the `hist_len` right-aligned window of `GuandanNet.encode_state`.
    fn init_root_lstm(&self, history: &[(u8, Move)], seats_mask: u8, h_out: &mut [[f32; 512]; 4], c_out: &mut [[f32; 512]; 4], gates: &mut [f32]) {
        let l = self.l;
        let n = history.len();
        let take = n.min(self.hist_len);
        let pad = self.hist_len - take;

        // Zero-padding prefix is IDENTICAL for all seats! Compute once and copy!
        let mut h0 = [0f32; 512];
        let mut c0 = [0f32; 512];
        for _ in 0..pad {
            self.lstm_step_zero(&mut h0[..l], &mut c0[..l], gates);
        }
        for s in 0..4usize {
            if (seats_mask & (1 << s)) != 0 {
                h_out[s][..l].copy_from_slice(&h0[..l]);
                c_out[s][..l].copy_from_slice(&c0[..l]);
            }
        }
        for (p, m) in &history[n - take..] {
            let cm = CompactHistMove::from_move(*p, m);
            for s in 0..4u8 {
                if (seats_mask & (1 << s)) != 0 {
                    self.lstm_step_compact(s, &cm, &mut h_out[s as usize][..l], &mut c_out[s as usize][..l], gates);
                }
            }
        }
    }

    /// Compute `z` from `sc.s_buf` (`[STATE_DIM]`) and `h_lstm` (`[l]`) into `sc.z_buf`.
    #[inline]
    fn state_to_z(&self, h_lstm: &[f32], sc: &mut FastScratch) {
        let first = &self.state_mlp[0];
        sc.mlp_a[..first.n_out].copy_from_slice(&first.b);
        first.add_bits(&sc.s_buf, 0, &mut sc.mlp_a[..first.n_out]);
        first.add_sparse_f32(&h_lstm[..self.l], STATE_DIM, &mut sc.mlp_a[..first.n_out]);
        relu_inplace(&mut sc.mlp_a[..first.n_out]);

        let mut prev_len = first.n_out;
        for (idx, lin) in self.state_mlp[1..].iter().enumerate() {
            if idx % 2 == 0 {
                lin.forward_sparse(&sc.mlp_a[..prev_len], &mut sc.mlp_b[..lin.n_out]);
                relu_inplace(&mut sc.mlp_b[..lin.n_out]);
            } else {
                lin.forward_sparse(&sc.mlp_b[..prev_len], &mut sc.mlp_a[..lin.n_out]);
                relu_inplace(&mut sc.mlp_a[..lin.n_out]);
            }
            prev_len = lin.n_out;
        }
        if (self.state_mlp.len() - 1) % 2 == 0 {
            sc.z_buf.copy_from_slice(&sc.mlp_a[..self.z_dim]);
        } else {
            sc.z_buf.copy_from_slice(&sc.mlp_b[..self.z_dim]);
        }
    }

    /// Score all `moves` at `g` for `me` given `sc.z_buf`, writing Q (in units of 3 levels) into `sc.q_buf`
    /// and returning the argmax move index.
    #[inline]
    fn score_moves(&self, g: &Game, me: u8, moves: &[Move], sc: &mut FastScratch) -> usize {
        let h0 = &self.head[0];
        let n0 = h0.n_out;
        sc.head_base[..n0].copy_from_slice(&h0.b);
        h0.add_sparse_f32(&sc.z_buf, 0, &mut sc.head_base[..n0]);

        sc.q_buf.clear();
        let n_layers = self.head.len();
        let mut best_i = 0usize;
        let mut best_q = f32::NEG_INFINITY;

        for (mi, mv) in moves.iter().enumerate() {
            encode_action(g, me, mv, &mut sc.a_buf);
            if n_layers == 1 {
                // Degenerate 1-layer head
                let mut v = sc.head_base[0];
                for (ai, &bit) in sc.a_buf.iter().enumerate() {
                    if bit != 0 {
                        v += h0.wt[(self.z_dim + ai) * n0];
                    }
                }
                sc.q_buf.push(v);
                if v > best_q {
                    best_q = v;
                    best_i = mi;
                }
                continue;
            }
            sc.head_a[..n0].copy_from_slice(&sc.head_base[..n0]);
            h0.add_bits(&sc.a_buf, self.z_dim, &mut sc.head_a[..n0]);
            relu_inplace(&mut sc.head_a[..n0]);

            let mut cur_len = n0;
            let mid_count = n_layers - 2; // hidden layers before the final scalar linear layer
            for j in 0..mid_count {
                let lin = &self.head[1 + j];
                if j % 2 == 0 {
                    lin.forward_sparse(&sc.head_a[..cur_len], &mut sc.head_b[..lin.n_out]);
                    relu_inplace(&mut sc.head_b[..lin.n_out]);
                } else {
                    lin.forward_sparse(&sc.head_b[..cur_len], &mut sc.head_a[..lin.n_out]);
                    relu_inplace(&mut sc.head_a[..lin.n_out]);
                }
                cur_len = lin.n_out;
            }
            let last = &self.head[n_layers - 1];
            let qv = if mid_count % 2 == 0 {
                last.dot1(&sc.head_a[..cur_len])
            } else {
                last.dot1(&sc.head_b[..cur_len])
            };
            sc.q_buf.push(qv);
            if qv > best_q {
                best_q = qv;
                best_i = mi;
            }
        }
        best_i
    }

    /// Exact single-game forward pass matching `GuandanNet` / `NumpyNet`:
    /// returns `(moves, q_raw, Option<belief_logits>)`.
    pub fn evaluate_game(&self, g: &Game, want_belief: bool) -> (Vec<Move>, Vec<f32>, Option<Vec<f32>>) {
        let me = g.current;
        let moves = g.legal_moves();
        if moves.is_empty() {
            return (moves, vec![], None);
        }
        let mut sc = FastScratch::new(self);
        self.init_root_lstm(&g.history, 1 << me, &mut sc.h_seats, &mut sc.c_seats, &mut sc.gates);
        encode_state(g, me, &mut sc.s_buf);
        let h_me = sc.h_seats[me as usize];
        self.state_to_z(&h_me[..self.l], &mut sc);
        self.score_moves(g, me, &moves, &mut sc);
        let lg = if want_belief {
            self.belief.as_ref().map(|b| {
                let mut out = vec![0f32; b.n_out];
                b.forward_dense(&sc.z_buf, &mut out);
                out
            })
        } else {
            None
        };
        (moves, sc.q_buf, lg)
    }

    /// Simulate one determinized world after playing `first_move`, using lazy per-seat
    /// incremental `T=1` LSTM updates (and optional exact `Solver` leaf cutoff when
    /// `solver_leaf_cards > 0`), returning `round_reward(me_team)`.
    #[allow(clippy::too_many_arguments)]
    fn simulate_branch(
        &self,
        world: &Game,
        first_move: &Move,
        me_team: u8,
        root_h: &[[f32; 512]; 4],
        root_c: &[[f32; 512]; 4],
        solver_leaf_cards: u32,
        ora_plies: u32,
        ora_blend: f32,
        solver: &mut Option<Solver>,
        sc: &mut FastScratch,
    ) -> f32 {
        let mut g = world.clone();
        let l = self.l;
        for s in 0..4 {
            sc.h_seats[s][..l].copy_from_slice(&root_h[s][..l]);
            sc.c_seats[s][..l].copy_from_slice(&root_c[s][..l]);
        }
        sc.hist_log.clear();
        let mut cursor = [0usize; 4];

        let p0 = g.current;
        let record0 = g.phase == Phase::Play;
        if g.apply(first_move).is_err() {
            return 0.0;
        }
        if record0 {
            sc.hist_log.push(CompactHistMove::from_move(p0, first_move));
        }

        let use_ora = ora_plies > 0 && ora_blend > 0.0 && self.oracle.is_some();
        let ob = ora_blend.clamp(0.0, 1.0);
        let mut ora_sum = 0.0f32;
        let mut ora_cnt = 0u32;

        for step_idx in 0..400u32 {
            match g.phase {
                Phase::RoundOver | Phase::MatchOver => {
                    let v_end = g.round_reward(me_team);
                    return if ora_cnt > 0 {
                        ob * (ora_sum / (ora_cnt as f32)) + (1.0 - ob) * v_end
                    } else {
                        v_end
                    };
                }
                Phase::Play if solver_leaf_cards > 0 => {
                    let cl: u32 = g.hands.iter().map(hand_count).sum();
                    if cl <= solver_leaf_cards {
                        if let Some(ref mut sol) = solver {
                            if let Some(v_cur) = sol.leaf_value(&g, 60_000) {
                                let sign = if team(g.current) == me_team { 1.0 } else { -1.0 };
                                let v_end = v_cur * sign;
                                return if ora_cnt > 0 {
                                    ob * (ora_sum / (ora_cnt as f32)) + (1.0 - ob) * v_end
                                } else {
                                    v_end
                                };
                            }
                        }
                    }
                }
                _ => {}
            }
            let me = g.current;
            let moves = g.legal_moves();
            if moves.is_empty() {
                break;
            }
            let need_ora_here = use_ora && (step_idx + 1 >= ora_plies) && ora_cnt < 2 && g.phase == Phase::Play;
            let choice = if moves.len() == 1 && !need_ora_here {
                0usize
            } else {
                // Lazily catch up ONLY `me`'s LSTM state to `sc.hist_log.len()`!
                let mu = me as usize;
                while cursor[mu] < sc.hist_log.len() {
                    let cm = sc.hist_log[cursor[mu]];
                    self.lstm_step_compact(me, &cm, &mut sc.h_seats[mu][..l], &mut sc.c_seats[mu][..l], &mut sc.gates);
                    cursor[mu] += 1;
                }
                encode_state(&g, me, &mut sc.s_buf);
                let h_copy = sc.h_seats[mu];
                self.state_to_z(&h_copy[..l], sc);
                self.score_moves(&g, me, &moves, sc)
            };
            let chosen_mv = &moves[choice];
            if need_ora_here {
                let q_lvl = sc.q_buf[choice] * 3.0;
                let sgn = if team(me) == me_team { 1.0 } else { -1.0 };
                ora_sum += self.eval_oracle(&g, me, chosen_mv, q_lvl, sc) * sgn;
                ora_cnt += 1;
                if ora_cnt == 2 && ob >= 0.999 {
                    return 0.5 * ora_sum;
                }
            }
            let rec = g.phase == Phase::Play;
            if rec {
                sc.hist_log.push(CompactHistMove::from_move(me, chosen_mv));
            }
            if g.apply(chosen_mv).is_err() {
                break;
            }
        }
        let v_end = g.round_reward(me_team);
        if ora_cnt > 0 {
            ob * (ora_sum / (ora_cnt as f32)) + (1.0 - ob) * v_end
        } else {
            v_end
        }
    }

    /// Root-cached SMC action-likelihood log-weights for `worlds` from `me`'s perspective!
    /// Uses public-history invariance: LSTM states at the last `last_k` decisions of the other
    /// 3 seats are computed ONCE on `root` before scoring `worlds` in parallel across `threads`.
    pub fn smc_log_weights(
        &self,
        root: &Game,
        worlds: &[Game],
        me: u8,
        last_k: usize,
        temp_levels: f32,
        floor_prob: f32,
        threads: usize,
    ) -> Vec<f64> {
        let m_cnt = worlds.len();
        if m_cnt == 0 || last_k == 0 || root.history.is_empty() {
            return vec![0.0; m_cnt];
        }
        let Some(mut skel) = root.play_start() else {
            return vec![0.0; m_cnt];
        };
        let n = root.history.len();
        let mut keep = vec![false; n];
        let mut kept = 0usize;
        for i in (0..n).rev() {
            if kept >= last_k {
                break;
            }
            if root.history[i].0 != me {
                keep[i] = true;
                kept += 1;
            }
        }
        if kept == 0 {
            return vec![0.0; m_cnt];
        }

        // Precompute exact root-prefix LSTM trajectories AND public state_mlp[0] base (`460..1550 + h_lstm`) ONCE!
        struct Snap {
            actor: u8,
            chosen_mv: Move,
            skel_game: Game,
            played_since: Hand,
            mlp0_public: Vec<f32>,
        }
        let mut snaps: Vec<Snap> = Vec::with_capacity(kept);
        let l = self.l;
        let first = &self.state_mlp[0];
        let mut gates = vec![0f32; 4 * l];
        let mut s_pub = [0u8; STATE_DIM];

        for i in 0..n {
            let (p, ref mv) = root.history[i];
            if keep[i] {
                let mut h_s = [[0f32; 512]; 4];
                let mut c_s = [[0f32; 512]; 4];
                self.init_root_lstm(&root.history[..i], 1 << p, &mut h_s, &mut c_s, &mut gates);
                let mut ps = [0u8; 54];
                for t in 0..54 {
                    ps[t] = root.played[p as usize][t].saturating_sub(skel.played[p as usize][t]);
                }
                let mut clean_skel = skel.clone();
                clean_skel.history.clear();
                clean_skel.history.shrink_to_fit();
                encode_state(&clean_skel, p, &mut s_pub);
                let mut mlp0_pub = first.b.clone();
                // Bytes 460..1550 (played, counts, levels, finished, trick, tribute) are invariant across worlds!
                first.add_bits(&s_pub[460..STATE_DIM], 460, &mut mlp0_pub);
                first.add_sparse_f32(&h_s[p as usize][..l], STATE_DIM, &mut mlp0_pub);
                snaps.push(Snap {
                    actor: p,
                    chosen_mv: mv.clone(),
                    skel_game: clean_skel,
                    played_since: ps,
                    mlp0_public: mlp0_pub,
                });
            }
            if skel.current != p || skel.apply(mv).is_err() {
                break;
            }
        }
        if snaps.is_empty() {
            return vec![0.0; m_cnt];
        }

        let inv_temp = 3.0 / temp_levels.max(0.05);
        let min_lp = floor_prob.max(1e-4).ln() as f64;
        let n_th = threads.max(1).min(m_cnt);
        let chunk = m_cnt.div_ceil(n_th);

        std::thread::scope(|scope| {
            let handles: Vec<_> = worlds
                .chunks(chunk)
                .map(|ws| {
                    let sn_ref = &snaps;
                    scope.spawn(move || {
                        let mut sc = FastScratch::new(self);
                        let mut skel_local: Vec<Game> = sn_ref.iter().map(|s| s.skel_game.clone()).collect();
                        let mut out = Vec::with_capacity(ws.len());
                        for w in ws {
                            let mut logw = 0.0f64;
                            for (si, sn) in sn_ref.iter().enumerate() {
                                let p = sn.actor;
                                let pu = p as usize;
                                let mut hand_then = w.hands[pu];
                                for t in 0..54 {
                                    hand_then[t] += sn.played_since[t];
                                }
                                let moves = legal_moves(&hand_then, sn.skel_game.level, sn.skel_game.last_move.as_ref());
                                if moves.len() <= 1 {
                                    continue;
                                }
                                let chosen_idx = moves.iter().position(|x| {
                                    x.kind == sn.chosen_mv.kind && x.key == sn.chosen_mv.key && x.cards == sn.chosen_mv.cards
                                });
                                let Some(ci) = chosen_idx else {
                                    logw += min_lp;
                                    continue;
                                };
                                let g_tmp = &mut skel_local[si];
                                g_tmp.hands[pu] = hand_then;
                                encode_state(g_tmp, p, &mut sc.s_buf);
                                // Start from precomputed public + h_lstm base, add ONLY private bytes 0..460!
                                let n0 = first.n_out;
                                sc.mlp_a[..n0].copy_from_slice(&sn.mlp0_public);
                                first.add_bits(&sc.s_buf[..460], 0, &mut sc.mlp_a[..n0]);
                                relu_inplace(&mut sc.mlp_a[..n0]);
                                let mut prev_len = n0;
                                for (idx, lin) in self.state_mlp[1..].iter().enumerate() {
                                    if idx % 2 == 0 {
                                        lin.forward_sparse(&sc.mlp_a[..prev_len], &mut sc.mlp_b[..lin.n_out]);
                                        relu_inplace(&mut sc.mlp_b[..lin.n_out]);
                                    } else {
                                        lin.forward_sparse(&sc.mlp_b[..prev_len], &mut sc.mlp_a[..lin.n_out]);
                                        relu_inplace(&mut sc.mlp_a[..lin.n_out]);
                                    }
                                    prev_len = lin.n_out;
                                }
                                if (self.state_mlp.len() - 1) % 2 == 0 {
                                    sc.z_buf.copy_from_slice(&sc.mlp_a[..self.z_dim]);
                                } else {
                                    sc.z_buf.copy_from_slice(&sc.mlp_b[..self.z_dim]);
                                }
                                self.score_moves(g_tmp, p, &moves, &mut sc);

                                let mut mx = f32::NEG_INFINITY;
                                for &qv in &sc.q_buf {
                                    let sv = qv * inv_temp;
                                    if sv > mx {
                                        mx = sv;
                                    }
                                }
                                let mut sum_exp = 0.0f64;
                                for &qv in &sc.q_buf {
                                    sum_exp += ((qv * inv_temp - mx) as f64).exp();
                                }
                                let lp = (sc.q_buf[ci] * inv_temp - mx) as f64 - sum_exp.ln();
                                logw += lp.max(min_lp);
                            }
                            out.push(logw);
                        }
                        out
                    })
                })
                .collect();
            handles.into_iter().flat_map(|h| h.join().unwrap_or_default()).collect()
        })
    }

    /// Resample `target_n` worlds systematically from `cand_worlds` using `logw`.
    pub fn systematic_resample(cand_worlds: &[Game], logw: &[f64], target_n: usize, rng: &mut Rng) -> Vec<Game> {
        let m = cand_worlds.len();
        if m == 0 || target_n == 0 {
            return vec![];
        }
        let mx = logw.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mut w: Vec<f64> = logw.iter().map(|&l| (l - mx).exp()).collect();
        let sum: f64 = w.iter().sum();
        if sum <= 0.0 || !sum.is_finite() {
            return cand_worlds.iter().take(target_n).cloned().collect();
        }
        let mut cdf = Vec::with_capacity(m);
        let mut acc = 0.0;
        for wi in &mut w {
            acc += *wi / sum;
            cdf.push(acc);
        }
        let u0 = rng.uniform();
        let mut out = Vec::with_capacity(target_n);
        let mut idx = 0usize;
        for k in 0..target_n {
            let u = (u0 + k as f64) / (target_n as f64);
            while idx + 1 < m && cdf[idx] < u {
                idx += 1;
            }
            out.push(cand_worlds[idx].clone());
        }
        out
    }

    /// Multi-threaded belief-calibrated full-round rollout search for Mid-Game & `Phase::Return`!
    /// Returns `Some(best_move_idx)` when a candidate beats `net_choice` on both the blended
    /// objective `(1 - prior) * rollout_mean + prior * q_levels` AND paired LCB (`margin_se * SE`),
    /// or `None` to keep `net_choice`.
    #[allow(clippy::too_many_arguments)]
    pub fn rollout_choose(
        &self,
        g: &Game,
        net_choice: usize,
        q_levels: &[f32],
        belief_logits: Option<&[f32]>,
        cfg: &RolloutCfg,
        seed: u64,
    ) -> Option<usize> {
        let moves = g.legal_moves();
        let n_mv = moves.len();
        if n_mv < 2 || net_choice >= n_mv || q_levels.len() != n_mv {
            return None;
        }
        let is_ret = g.phase == Phase::Return;
        let is_play = g.phase == Phase::Play;
        if !is_ret && !is_play {
            return None;
        }
        let cl: u32 = g.hands.iter().map(hand_count).sum();
        let gap_limit = if is_ret {
            if !cfg.search_return {
                return None;
            }
            cfg.return_gap
        } else {
            if cl < cfg.min_cards || cl > cfg.max_cards {
                return None;
            }
            if cfg.early_cards > 0 && cl > cfg.early_cards {
                cfg.early_close_gap
            } else {
                cfg.close_gap
            }
        };

        // Order moves by descending Q and check if top-2 gap triggers search!
        let mut ord: Vec<usize> = (0..n_mv).collect();
        ord.sort_unstable_by(|&a, &b| q_levels[b].partial_cmp(&q_levels[a]).unwrap_or(std::cmp::Ordering::Equal));
        if q_levels[ord[0]] - q_levels[ord[1]] > gap_limit {
            return None;
        }

        let max_k = if is_ret { cfg.return_top_k.max(2) } else { cfg.top_k.max(2) };
        let top_q = q_levels[net_choice];
        let mut cands: Vec<usize> = Vec::with_capacity(max_k.min(n_mv));
        cands.push(net_choice);
        for &i in &ord {
            if cands.len() >= max_k {
                break;
            }
            if i != net_choice && top_q - q_levels[i] <= cfg.max_cand_gap {
                cands.push(i);
            }
        }
        if cands.len() < 2 {
            return None;
        }

        // 1. Compute exact root LSTM states for ALL 4 seats ONCE!
        let mut root_h = [[0f32; 512]; 4];
        let mut root_c = [[0f32; 512]; 4];
        let mut gates = vec![0f32; 4 * self.l];
        self.init_root_lstm(&g.history, 0b1111, &mut root_h, &mut root_c, &mut gates);

        // 2. Sample belief-calibrated worlds (with optional SMC action-likelihood filtering)
        let me = g.current;
        let me_team = team(me);
        let mut rng = Rng::new(seed);
        let m_draw = cfg.worlds * cfg.smc_oversample.max(1);
        let raw_worlds: Vec<Game> = (0..m_draw)
            .map(|_| {
                if let Some(lg) = belief_logits {
                    g.determinize_belief_clustered(me, lg, cfg.floor, cfg.bomb_gamma, &mut rng)
                } else {
                    g.determinize(me, &mut rng)
                }
            })
            .collect();
        let worlds = if cfg.smc_oversample > 1 && is_play && !g.history.is_empty() {
            let logw = self.smc_log_weights(g, &raw_worlds, me, cfg.smc_last_k, 0.40, 0.04, cfg.threads);
            Self::systematic_resample(&raw_worlds, &logw, cfg.worlds, &mut rng)
        } else {
            raw_worlds
        };
        let nw = worlds.len();
        if nw == 0 {
            return None;
        }
        let leaf_cards = cfg.solver_leaf_cards;
        let pr = cfg.prior.clamp(0.0, 0.95) as f64;
        let (o_plies, o_blend) = if is_play && cfg.early_cards > 0 && cl > cfg.early_cards {
            (cfg.early_oracle_plies, cfg.early_oracle_blend)
        } else {
            (cfg.oracle_plies, cfg.oracle_blend)
        };

        let eval_grid = |ws_slice: &[Game], c_slice: &[usize]| -> Vec<f32> {
            let n_w = ws_slice.len();
            let n_c = c_slice.len();
            let total_jobs = n_w * n_c;
            if total_jobs == 0 {
                return vec![];
            }
            let threads = cfg.threads.max(1).min(total_jobs);
            let job_indices: Vec<usize> = (0..total_jobs).collect();
            let chunk = total_jobs.div_ceil(threads);
            std::thread::scope(|scope| {
                let handles: Vec<_> = job_indices
                    .chunks(chunk)
                    .map(|jobs| {
                        let mvs_ref = &moves;
                        let rh = &root_h;
                        let rc = &root_c;
                        scope.spawn(move || {
                            let mut sc = FastScratch::new(self);
                            let mut sol = if leaf_cards > 0 { Some(Solver::new(60_000)) } else { None };
                            let mut res = Vec::with_capacity(jobs.len());
                            for &job in jobs {
                                let wi = job % n_w;
                                let ci = job / n_w;
                                let mv = &mvs_ref[c_slice[ci]];
                                let v = self.simulate_branch(&ws_slice[wi], mv, me_team, rh, rc, leaf_cards, o_plies, o_blend, &mut sol, &mut sc);
                                res.push(v);
                            }
                            res
                        })
                    })
                    .collect();
                handles.into_iter().flat_map(|h| h.join().unwrap_or_default()).collect()
            })
        };

        // 3. Two-stage Sequential Halving when `cands.len() >= 4` and `nw >= 14`:
        // Stage 1 evaluates all `cands` on the first `w1 = nw / 2` worlds, retains `cands[0]`
        // (`net_choice`) + top 2 alternatives, and Stage 2 evaluates finalists on `worlds[w1..nw]`.
        let (final_cands, flat_vals) = if cands.len() >= 4 && nw >= 14 {
            let w1 = nw / 2;
            let vals1 = eval_grid(&worlds[..w1], &cands);
            let mut alts: Vec<(f64, usize)> = (1..cands.len())
                .map(|ci| {
                    let m1 = vals1[ci * w1..(ci + 1) * w1].iter().map(|&x| x as f64).sum::<f64>() / (w1 as f64);
                    let sc1 = (1.0 - pr) * m1 + pr * (q_levels[cands[ci]] as f64);
                    (sc1, ci)
                })
                .collect();
            alts.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
            let mut keep_ci = Vec::with_capacity(3);
            keep_ci.push(0usize);
            for &(_, ci) in alts.iter().take(2) {
                keep_ci.push(ci);
            }
            let fc: Vec<usize> = keep_ci.iter().map(|&ci| cands[ci]).collect();
            let vals2 = eval_grid(&worlds[w1..], &fc);
            let w2 = nw - w1;
            let mut merged = Vec::with_capacity(fc.len() * nw);
            for (new_i, &old_ci) in keep_ci.iter().enumerate() {
                merged.extend_from_slice(&vals1[old_ci * w1..(old_ci + 1) * w1]);
                merged.extend_from_slice(&vals2[new_i * w2..(new_i + 1) * w2]);
            }
            (fc, merged)
        } else {
            let v = eval_grid(&worlds, &cands);
            (cands, v)
        };
        let nc = final_cands.len();

        // 4. Aggregate paired means, SEs, and blended scores against `final_cands[0] == net_choice`
        let mut means = vec![0.0f64; nc];
        for ci in 0..nc {
            let slice = &flat_vals[ci * nw..(ci + 1) * nw];
            means[ci] = slice.iter().map(|&x| x as f64).sum::<f64>() / (nw as f64);
        }
        let blended: Vec<f64> = (0..nc)
            .map(|ci| (1.0 - pr) * means[ci] + pr * (q_levels[final_cands[ci]] as f64))
            .collect();

        let mut best_ci = 0usize;
        let mut best_sc = blended[0];
        let base_vals = &flat_vals[0..nw];

        let min_diff = if is_play && cfg.early_cards > 0 && cl > cfg.early_cards {
            cfg.early_margin_abs as f64
        } else {
            cfg.margin_abs as f64
        };
        for ci in 1..nc {
            if blended[ci] <= best_sc + 1e-9 {
                continue;
            }
            let cand_vals = &flat_vals[ci * nw..(ci + 1) * nw];
            let diff_mean = means[ci] - means[0];
            if diff_mean <= min_diff {
                continue;
            }
            let se = if nw > 1 {
                let var = cand_vals
                    .iter()
                    .zip(base_vals.iter())
                    .map(|(&a, &b)| {
                        let d = (a - b) as f64 - diff_mean;
                        d * d
                    })
                    .sum::<f64>()
                    / ((nw - 1) as f64);
                (var / (nw as f64)).sqrt()
            } else {
                0.0
            };
            if diff_mean > (cfg.margin_se as f64) * se {
                best_sc = blended[ci];
                best_ci = ci;
            }
        }

        if best_ci != 0 {
            Some(final_cands[best_ci])
        } else {
            None
        }
    }

    /// Anti-Fusion Hybrid Endgame Search (`DD + IIR` Concave Envelope):
    /// 1. Screens up to `max_cands` root moves across `worlds` SMC-filtered belief worlds via exact
    ///    Double-Dummy alpha-beta minimax (`V_DD`, guarding against opponent tactical refutations).
    /// 2. Selects `net_choice` + top 3 DD finalists and evaluates Imperfect-Information `FastNet`
    ///    Rollout down to `iir_leaf_cards` (`V_IIR`, modeling realistic non-peeking partner & self
    ///    play) on the EXACT SAME worlds!
    /// 3. Combines `(V_DD, V_IIR)` per world via asymmetric concave envelope:
    ///    - `V_IIR < V_DD` (Telepathy / Strategy-Fusion illusion): `(1 - iir_w_neg) * V_DD + iir_w_neg * V_IIR`
    ///    - `V_IIR >= V_DD` (Opponent Imperfect-Info advantage):   `(1 - iir_w_pos) * V_DD + iir_w_pos * V_IIR`
    #[allow(clippy::too_many_arguments)]
    pub fn anti_fusion_endgame_choose(
        &self,
        g: &Game,
        net_choice: usize,
        q_levels: &[f32],
        belief_logits: Option<&[f32]>,
        cards: u32,
        worlds: usize,
        floor: f32,
        bomb_gamma: f32,
        max_nodes: u64,
        margin: f32,
        margin_se: f32,
        prior: f32,
        max_cands: usize,
        q_gap: f32,
        iir_w_neg: f32,
        iir_w_pos: f32,
        iir_min_cards: u32,
        iir_leaf_cards: u32,
        smc_oversample: usize,
        smc_last_k: usize,
        threads: usize,
        seed: u64,
    ) -> Option<usize> {
        if g.phase != Phase::Play {
            return None;
        }
        let left: u32 = g.hands.iter().map(hand_count).sum();
        if left > cards {
            return None;
        }
        let moves = g.legal_moves();
        let n_mv = moves.len();
        if n_mv < 2 || net_choice >= n_mv || q_levels.len() != n_mv {
            return None;
        }

        let mut ord: Vec<usize> = (0..n_mv).collect();
        ord.sort_unstable_by(|&a, &b| q_levels[b].partial_cmp(&q_levels[a]).unwrap_or(std::cmp::Ordering::Equal));
        let top_q = q_levels[net_choice];
        let limit = if max_cands > 0 { max_cands.min(n_mv) } else { n_mv };
        let mut cands: Vec<usize> = Vec::with_capacity(limit);
        cands.push(net_choice);
        for &i in &ord {
            if cands.len() >= limit {
                break;
            }
            if i != net_choice && (q_gap <= 0.0 || top_q - q_levels[i] <= q_gap) {
                cands.push(i);
            }
        }
        if cands.len() < 2 {
            return None;
        }

        let me = g.current;
        let me_team = team(me);
        let mut rng = Rng::new(seed);
        let n_draw = worlds * smc_oversample.max(1);
        let raw_worlds: Vec<Game> = (0..n_draw)
            .map(|_| {
                if let Some(lg) = belief_logits {
                    g.determinize_belief_clustered(me, lg, floor, bomb_gamma, &mut rng)
                } else {
                    g.determinize(me, &mut rng)
                }
            })
            .collect();
        let ws_vec: Vec<Game> = if smc_oversample > 1 && smc_last_k > 0 && !g.history.is_empty() {
            let logw = self.smc_log_weights(g, &raw_worlds, me, smc_last_k, 0.40, 0.04, threads);
            Self::systematic_resample(&raw_worlds, &logw, worlds, &mut rng)
        } else {
            raw_worlds
        };
        if ws_vec.is_empty() {
            return None;
        }

        let n_th = threads.max(1).min(ws_vec.len());
        let chunk = ws_vec.len().div_ceil(n_th);
        let nc = cands.len();

        // Stage 1: Exact Double-Dummy solve across all `ws_vec` for `cands`
        let dd_rows: Vec<(usize, Vec<f32>)> = std::thread::scope(|scope| {
            let handles: Vec<_> = ws_vec
                .chunks(chunk)
                .enumerate()
                .map(|(bi, sub_ws)| {
                    let base_wi = bi * chunk;
                    let mvs_ref = &moves;
                    let cs_ref = &cands;
                    scope.spawn(move || {
                        let mut solver_dd = Solver::new(max_nodes);
                        let mut out = Vec::with_capacity(sub_ws.len());
                        for (off, w) in sub_ws.iter().enumerate() {
                            if let Some(dd) = solver_dd.solve_subset(w, mvs_ref, cs_ref, max_nodes) {
                                out.push((base_wi + off, dd));
                            }
                        }
                        out
                    })
                })
                .collect();
            handles.into_iter().flat_map(|h| h.join().unwrap_or_default()).collect()
        });

        let ok = dd_rows.len();
        if ok == 0 {
            return None;
        }

        let pr = prior.clamp(0.0, 0.95) as f64;
        let mut dd_sum = vec![0.0f64; nc];
        for (_, dd) in &dd_rows {
            for ci in 0..nc {
                dd_sum[ci] += dd[ci] as f64;
            }
        }

        // Select finalists: `ci = 0` (`net_choice`) + top 2 DD-blended contenders!
        let dd_sc_0 = (1.0 - pr) * (dd_sum[0] / (ok as f64)) + pr * (q_levels[cands[0]] as f64);
        let mut best_alt_dd_sc = f64::NEG_INFINITY;
        let finalist_ci: Vec<usize> = if nc <= 3 {
            for ci in 1..nc {
                let sc = (1.0 - pr) * (dd_sum[ci] / (ok as f64)) + pr * (q_levels[cands[ci]] as f64);
                if sc > best_alt_dd_sc {
                    best_alt_dd_sc = sc;
                }
            }
            (0..nc).collect()
        } else {
            let mut alts: Vec<(f64, usize)> = (1..nc)
                .map(|ci| {
                    let m_dd = dd_sum[ci] / (ok as f64);
                    let sc = (1.0 - pr) * m_dd + pr * (q_levels[cands[ci]] as f64);
                    (sc, ci)
                })
                .collect();
            alts.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
            best_alt_dd_sc = alts[0].0;
            let mut fc = Vec::with_capacity(3);
            fc.push(0usize);
            for &(sc, ci) in alts.iter().take(2) {
                if sc >= best_alt_dd_sc - 0.65 || fc.len() < 2 {
                    fc.push(ci);
                }
            }
            fc
        };
        let nf = finalist_ci.len();

        // Stage 2: Imperfect-Information `FastNet` Rollout (`V_IIR`) on the EXACT SAME `ok` solved worlds
        // for the `nf <= 3` finalists when `left >= iir_min_cards` AND the decision is contested
        // (either a DD contender challenges `net_choice` within `0.20` lvl, or top-2 Q gap `<= 0.20`)!
        let w_neg = iir_w_neg.clamp(0.0, 1.0);
        let w_pos = iir_w_pos.clamp(0.0, 1.0);
        let q_contested = q_levels[ord[0]] - q_levels[ord[1]] <= 0.20;
        let dd_contested = best_alt_dd_sc >= dd_sc_0 - 0.20;
        let use_iir = (w_neg > 0.0 || w_pos > 0.0) && left >= iir_min_cards && (dd_contested || q_contested);

        // `final_vals[fi * ok + ri]` stores per-world score for finalist `fi` on solved world `ri`
        let final_vals: Vec<f32> = if use_iir {
            let mut root_h = [[0f32; 512]; 4];
            let mut root_c = [[0f32; 512]; 4];
            let mut gates = vec![0f32; 4 * self.l];
            self.init_root_lstm(&g.history, 0b1111, &mut root_h, &mut root_c, &mut gates);

            let total_jobs = ok * nf;
            let th2 = threads.max(1).min(total_jobs);
            let ch2 = total_jobs.div_ceil(th2);
            let job_ids: Vec<usize> = (0..total_jobs).collect();
            std::thread::scope(|scope| {
                let handles: Vec<_> = job_ids
                    .chunks(ch2)
                    .map(|jobs| {
                        let mvs_ref = &moves;
                        let cs_ref = &cands;
                        let fc_ref = &finalist_ci;
                        let ddr_ref = &dd_rows;
                        let ws_ref = &ws_vec;
                        let rh = &root_h;
                        let rc = &root_c;
                        scope.spawn(move || {
                            let mut sc = FastScratch::new(self);
                            let mut sol_leaf = if iir_leaf_cards > 0 {
                                Some(Solver::new(60_000))
                            } else {
                                None
                            };
                            let mut out = Vec::with_capacity(jobs.len());
                            for &job in jobs {
                                let ri = job % ok;
                                let fi = job / ok;
                                let (wi, ref dd_vec) = ddr_ref[ri];
                                let ci = fc_ref[fi];
                                let v_dd = dd_vec[ci];
                                let v_iir = self.simulate_branch(
                                    &ws_ref[wi],
                                    &mvs_ref[cs_ref[ci]],
                                    me_team,
                                    rh,
                                    rc,
                                    iir_leaf_cards,
                                    0,
                                    0.0,
                                    &mut sol_leaf,
                                    &mut sc,
                                );
                                let w = if v_iir < v_dd { w_neg } else { w_pos };
                                out.push((1.0 - w) * v_dd + w * v_iir);
                            }
                            out
                        })
                    })
                    .collect();
                handles.into_iter().flat_map(|h| h.join().unwrap_or_default()).collect()
            })
        } else {
            let mut out = Vec::with_capacity(ok * nf);
            for &ci in &finalist_ci {
                for (_, dd_vec) in &dd_rows {
                    out.push(dd_vec[ci]);
                }
            }
            out
        };

        let mut means = vec![0.0f64; nf];
        for fi in 0..nf {
            let sl = &final_vals[fi * ok..(fi + 1) * ok];
            means[fi] = sl.iter().map(|&x| x as f64).sum::<f64>() / (ok as f64);
        }
        let sc_of = |fi: usize| -> f64 {
            (1.0 - pr) * means[fi] + pr * (q_levels[cands[finalist_ci[fi]]] as f64)
        };
        let base_sc = sc_of(0);
        let base_sl = &final_vals[0..ok];
        let mut best_fi = 0usize;
        let mut best_sc = base_sc;

        for fi in 1..nf {
            let sc = sc_of(fi);
            if sc <= best_sc + 1e-9 {
                continue;
            }
            let diff_mean = means[fi] - means[0];
            if diff_mean <= 1e-9 {
                continue;
            }
            if margin_se > 0.0 && ok > 1 {
                let cand_sl = &final_vals[fi * ok..(fi + 1) * ok];
                let var = cand_sl
                    .iter()
                    .zip(base_sl.iter())
                    .map(|(&a, &b)| {
                        let d = (a - b) as f64 - diff_mean;
                        d * d
                    })
                    .sum::<f64>()
                    / ((ok - 1) as f64);
                let se = (var / (ok as f64)).sqrt();
                if diff_mean <= (margin_se as f64) * se {
                    continue;
                }
            }
            best_sc = sc;
            best_fi = fi;
        }

        if best_fi != 0 && best_sc - base_sc > (margin as f64) + 1e-9 {
            Some(cands[finalist_ci[best_fi]])
        } else {
            None
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct RolloutCfg {
    pub min_cards: u32,
    pub max_cards: u32,
    pub early_cards: u32,
    pub early_close_gap: f32,
    pub worlds: usize,
    pub top_k: usize,
    pub close_gap: f32,
    pub max_cand_gap: f32,
    pub prior: f32,
    pub margin_se: f32,
    pub margin_abs: f32,
    pub early_margin_abs: f32,
    pub floor: f32,
    pub bomb_gamma: f32,
    pub solver_leaf_cards: u32,
    pub oracle_plies: u32,
    pub oracle_blend: f32,
    pub early_oracle_plies: u32,
    pub early_oracle_blend: f32,
    pub search_return: bool,
    pub return_gap: f32,
    pub return_top_k: usize,
    pub smc_oversample: usize,
    pub smc_last_k: usize,
    pub threads: usize,
}

impl Default for RolloutCfg {
    fn default() -> Self {
        RolloutCfg {
            min_cards: 27,
            max_cards: 56,
            early_cards: 56,
            early_close_gap: 0.06,
            worlds: 16,
            top_k: 3,
            close_gap: 0.08,
            max_cand_gap: 0.45,
            prior: 0.35,
            margin_se: 0.75,
            margin_abs: 0.0,
            early_margin_abs: 0.0,
            floor: 0.02,
            bomb_gamma: 0.0,
            solver_leaf_cards: 0,
            oracle_plies: 0,
            oracle_blend: 0.0,
            early_oracle_plies: 0,
            early_oracle_blend: 0.0,
            search_return: true,
            return_gap: 0.20,
            return_top_k: 4,
            smc_oversample: 1,
            smc_last_k: 6,
            threads: 8,
        }
    }
}
