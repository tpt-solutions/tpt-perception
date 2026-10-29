#![allow(missing_docs)]

//! k-d tree construction and nearest-neighbour query throughput.
//!
//! Run with: `cargo bench -p tpt-percept-cloud`

use criterion::{black_box, criterion_group, criterion_main, Criterion};

use tpt_percept_cloud::kdtree::KdTree;

/// Deterministic pseudo-random points in the unit cube.
fn cloud(n: usize) -> Vec<[f64; 3]> {
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 11) as f64 / (1u64 << 53) as f64
    };
    (0..n).map(|_| [next(), next(), next()]).collect()
}

fn bench_kdtree(c: &mut Criterion) {
    let points = cloud(100_000);
    let queries = cloud(1_000);

    c.bench_function("kdtree/build_100k", |b| {
        b.iter(|| KdTree::new(black_box(&points)))
    });

    let tree = KdTree::new(&points);
    c.bench_function("kdtree/nearest_1k_queries", |b| {
        b.iter(|| {
            for q in &queries {
                black_box(tree.nearest(q));
            }
        })
    });
}

criterion_group!(benches, bench_kdtree);
criterion_main!(benches);
