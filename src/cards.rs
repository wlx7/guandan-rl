//! Card primitives.
//!
//! Card *types* are numbered 0..54:
//!   rank r in 0..13 (0 = "2", 8 = "10", 9 = J, 10 = Q, 11 = K, 12 = A), suit s in 0..4
//!   (0 = spades, 1 = hearts, 2 = clubs, 3 = diamonds) -> type = r * 4 + s
//!   52 = small joker, 53 = big joker.
//! Two decks are used, so a hand stores a count (0..=2) per card type.

pub type Hand = [u8; 54];

pub const SJ: u8 = 52;
pub const BJ: u8 = 53;
pub const HEARTS: u8 = 1;
pub const ACE: u8 = 12;

#[inline]
pub fn rank_of(t: u8) -> u8 {
    if t >= 52 {
        t - 52 + 13
    } else {
        t >> 2
    }
}

#[inline]
pub fn suit_of(t: u8) -> u8 {
    if t >= 52 {
        4
    } else {
        t & 3
    }
}

#[inline]
pub fn ctype(rank: u8, suit: u8) -> u8 {
    rank * 4 + suit
}

/// The wild card ("逢人配"): the heart card of the current level rank.
#[inline]
pub fn wild_type(level: u8) -> u8 {
    level * 4 + HEARTS
}

/// Comparable strength of a rank for singles / pairs / triples / bombs.
/// 2..A -> 0..12, level rank -> 13, small joker -> 14, big joker -> 15.
#[inline]
pub fn strength(rank: u8, level: u8) -> u8 {
    if rank >= 13 {
        rank + 1
    } else if rank == level {
        13
    } else {
        rank
    }
}

/// Dense strength index 0..15 (no gaps): non-level ranks 0..11, level 12, SJ 13, BJ 14.
#[inline]
pub fn sidx(rank: u8, level: u8) -> u8 {
    if rank >= 13 {
        rank
    } else if rank == level {
        12
    } else if rank < level {
        rank
    } else {
        rank - 1
    }
}

/// Natural rank at a sequence position. Position 0 is A (low), 1..=13 are 2..A.
#[inline]
pub fn seq_rank(pos: u8) -> u8 {
    if pos == 0 {
        ACE
    } else {
        pos - 1
    }
}

pub fn hand_count(h: &Hand) -> u32 {
    h.iter().map(|&c| c as u32).sum()
}

pub fn hand_from_cards(cards: &[u8]) -> Hand {
    let mut h = [0u8; 54];
    for &c in cards {
        h[c as usize] += 1;
    }
    h
}

pub fn hand_contains(h: &Hand, cards: &[u8]) -> bool {
    let need = hand_from_cards(cards);
    (0..54).all(|i| h[i] >= need[i])
}

pub fn hand_to_vec(h: &Hand) -> Vec<u8> {
    let mut v = Vec::with_capacity(27);
    for t in 0..54u8 {
        for _ in 0..h[t as usize] {
            v.push(t);
        }
    }
    v
}

pub const RANK_NAMES: [&str; 15] = [
    "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A", "SJ", "BJ",
];
pub const SUIT_NAMES: [&str; 5] = ["♠", "♥", "♣", "♦", ""];

pub fn card_name(t: u8) -> String {
    let r = rank_of(t);
    if r >= 13 {
        if r == 13 { "小王".into() } else { "大王".into() }
    } else {
        format!("{}{}", SUIT_NAMES[suit_of(t) as usize], RANK_NAMES[r as usize])
    }
}

/// xoshiro256** PRNG (small, fast, deterministic across platforms).
#[derive(Clone, Debug)]
pub struct Rng {
    s: [u64; 4],
}

impl Rng {
    pub fn new(seed: u64) -> Rng {
        let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut s = [0u64; 4];
        for x in s.iter_mut() {
            z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut v = z;
            v = (v ^ (v >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            v = (v ^ (v >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            *x = v ^ (v >> 31);
        }
        Rng { s }
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let result = self.s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }

    #[inline]
    pub fn below(&mut self, n: u64) -> u64 {
        if n == 0 {
            return 0;
        }
        self.next_u64() % n
    }

    pub fn shuffle<T>(&mut self, v: &mut [T]) {
        for i in (1..v.len()).rev() {
            let j = self.below(i as u64 + 1) as usize;
            v.swap(i, j);
        }
    }

    pub fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}
