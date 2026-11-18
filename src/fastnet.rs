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
    q_buf: Vec<f32>,
    h_seats: [[f32; 512]; 4], // supports up to L = 512 (xlarge)
    c_seats: [[f32; 512]; 4],
    hist_log: Vec<CompactHistMove>,
}

impl FastScratch {
    pub fn new(net: &FastNet) -> Self {
        let max_mlp = net.state_mlp.iter().map(|l| l.n_out).max().unwrap_or(768);
        let max_head = net.head.iter().map(|l| l.n_out).max().unwrap_or(768);
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
        })
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

        for _ in 0..400 {
            match g.phase {
                Phase::RoundOver | Phase::MatchOver => return g.round_reward(me_team),
                Phase::Play if solver_leaf_cards > 0 => {
                    let cl: u32 = g.hands.iter().map(hand_count).sum();
                    if cl <= solver_leaf_cards {
                        if let Some(ref mut sol) = solver {
                            if let Some(v_cur) = sol.leaf_value(&g, 60_000) {
                                let sign = if team(g.current) == me_team { 1.0 } else { -1.0 };
                                return v_cur * sign;
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
            let choice = if moves.len() == 1 {
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
            let rec = g.phase == Phase::Play;
            if rec {
                sc.hist_log.push(CompactHistMove::from_move(me, chosen_mv));
            }
            if g.apply(chosen_mv).is_err() {
                break;
            }
        }
        g.round_reward(me_team)
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
            cfg.close_gap
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
        let nc = cands.len();

        // 3. Evaluate all `nw * nc` common-random-number branches in parallel across threads!
        let total_jobs = nw * nc;
        let threads = cfg.threads.max(1).min(total_jobs);
        let job_indices: Vec<usize> = (0..total_jobs).collect();
        let chunk = total_jobs.div_ceil(threads);
        let leaf_cards = cfg.solver_leaf_cards;

        let flat_vals: Vec<f32> = std::thread::scope(|scope| {
            let handles: Vec<_> = job_indices
                .chunks(chunk)
                .map(|jobs| {
                    let ws_ref = &worlds;
                    let cs_ref = &cands;
                    let mvs_ref = &moves;
                    let rh = &root_h;
                    let rc = &root_c;
                    scope.spawn(move || {
                        let mut sc = FastScratch::new(self);
                        let mut sol = if leaf_cards > 0 { Some(Solver::new(60_000)) } else { None };
                        let mut res = Vec::with_capacity(jobs.len());
                        for &job in jobs {
                            let wi = job % nw;
                            let ci = job / nw;
                            let mv = &mvs_ref[cs_ref[ci]];
                            let v = self.simulate_branch(&ws_ref[wi], mv, me_team, rh, rc, leaf_cards, &mut sol, &mut sc);
                            res.push(v);
                        }
                        res
                    })
                })
                .collect();
            handles.into_iter().flat_map(|h| h.join().unwrap_or_default()).collect()
        });

        // 4. Aggregate paired means, SEs, and blended scores against `cands[0] == net_choice`
        let mut means = vec![0.0f64; nc];
        for ci in 0..nc {
            let slice = &flat_vals[ci * nw..(ci + 1) * nw];
            means[ci] = slice.iter().map(|&x| x as f64).sum::<f64>() / (nw as f64);
        }
        let pr = cfg.prior.clamp(0.0, 0.95) as f64;
        let blended: Vec<f64> = (0..nc)
            .map(|ci| (1.0 - pr) * means[ci] + pr * (q_levels[cands[ci]] as f64))
            .collect();

        let mut best_ci = 0usize;
        let mut best_sc = blended[0];
        let base_vals = &flat_vals[0..nw];

        for ci in 1..nc {
            if blended[ci] <= best_sc + 1e-9 {
                continue;
            }
            let cand_vals = &flat_vals[ci * nw..(ci + 1) * nw];
            let diff_mean = means[ci] - means[0];
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
            Some(cands[best_ci])
        } else {
            None
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct RolloutCfg {
    pub min_cards: u32,
    pub max_cards: u32,
    pub worlds: usize,
    pub top_k: usize,
    pub close_gap: f32,
    pub max_cand_gap: f32,
    pub prior: f32,
    pub margin_se: f32,
    pub floor: f32,
    pub bomb_gamma: f32,
    pub solver_leaf_cards: u32,
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
            worlds: 16,
            top_k: 3,
            close_gap: 0.08,
            max_cand_gap: 0.45,
            prior: 0.35,
            margin_se: 0.75,
            floor: 0.02,
            bomb_gamma: 0.0,
            solver_leaf_cards: 0,
            search_return: true,
            return_gap: 0.20,
            return_top_k: 4,
            smc_oversample: 1,
            smc_last_k: 6,
            threads: 8,
        }
    }
}
