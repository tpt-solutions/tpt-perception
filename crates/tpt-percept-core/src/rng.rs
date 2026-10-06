//! Deterministic pseudo-random number generation.
//!
//! Registration and segmentation algorithms (RANSAC, Hough voting) sample
//! hypotheses randomly; a tiny seeded generator keeps them reproducible —
//! identical inputs and seeds give identical outputs, which property tests
//! and CI rely on. This is *not* cryptographically secure (xorshift64*).

/// xorshift64* (Vigna) — small, fast, allocation-free.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct XorShift64Star {
    state: u64,
}

impl XorShift64Star {
    /// Creates a generator from a seed; a zero seed is replaced by a fixed
    /// non-zero state.
    pub fn new(seed: u64) -> Self {
        XorShift64Star {
            state: if seed == 0 { 0x2545F4914F6CDD1D } else { seed },
        }
    }

    /// Next raw u64.
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }

    /// Uniform index in `0..n`; returns 0 for `n == 0` (callers guard).
    pub fn below(&mut self, n: usize) -> usize {
        if n == 0 {
            return 0;
        }
        (self.next_u64() % n as u64) as usize
    }

    /// Uniform `f64` in `[0, 1)`.
    pub fn next_f64(&mut self) -> f64 {
        // 53 significant bits.
        ((self.next_u64() >> 11) as f64) / (1u64 << 53) as f64
    }

    /// Uniform `f64` in `[min, max)`.
    pub fn range_f64(&mut self, min: f64, max: f64) -> f64 {
        min + self.next_f64() * (max - min)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_for_seed() {
        let mut a = XorShift64Star::new(12345);
        let mut b = XorShift64Star::new(12345);
        for _ in 0..100 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn uniform_statistics() {
        let mut rng = XorShift64Star::new(7);
        let n = 20_000;
        let sum: u64 = (0..n).map(|_| rng.below(10) as u64).sum();
        let mean = sum as f64 / n as f64;
        assert!((mean - 4.5).abs() < 0.1, "mean {mean}");
        for _ in 0..1000 {
            let f = rng.next_f64();
            assert!((0.0..1.0).contains(&f));
        }
    }

    #[test]
    fn zero_seed_handled() {
        let mut a = XorShift64Star::new(0);
        let mut b = XorShift64Star::new(0x2545F4914F6CDD1D);
        assert_eq!(a.next_u64(), b.next_u64());
        assert_eq!(XorShift64Star::new(9).below(0), 0);
    }
}
