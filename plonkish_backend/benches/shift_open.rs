//! Benchmark comparing OLD (Rotation) vs NEW (shift_open) approaches for shift proofs
//!
//! Run with:
//!   cargo bench --package plonkish_backend --bench shift_open --features benchmark
//!
//! This benchmark compares:
//! - OLD: Using Expression with Rotation + batch_open_for_shift (existing mechanism)
//! - NEW: Using piop::shift_open with sum-check protocol (new independent protocol)

use criterion::{
    black_box, criterion_group, criterion_main, measurement::Measurement, BenchmarkGroup,
    BenchmarkId, Criterion,
};
use halo2_curves::bn256::{Bn256, Fr};
use plonkish_backend::{
    backend::{
        hyperplonk::{
            util::{shift_open_test::shift_open_base_circuit, rand_vanilla_plonk_circuit},
            HyperPlonk,
        },
        PlonkishBackend, PlonkishCircuit,
    },
    pcs::{
        multilinear::Zeromorph,
        univariate::UnivariateKzg,
        PolynomialCommitmentScheme,
    },
    piop::shift_open::{prove_shift_open, verify_shift_open},
    poly::multilinear::MultilinearPolynomial,
    util::{
        expression::rotate::Lexical,
        test::std_rng,
        transcript::{InMemoryTranscript, Keccak256Transcript, TranscriptRead, TranscriptWrite},
    },
};
use std::io::Cursor;

type Pcs = Zeromorph<UnivariateKzg<Bn256>>;
type Backend = HyperPlonk<Pcs>;
type Transcript = Keccak256Transcript<Cursor<Vec<u8>>>;

const NUM_VARS_RANGE: std::ops::Range<usize> = 10..16;

/// Benchmark the NEW shift_open protocol (sum-check based)
fn bench_new_shift_open_prove(group: &mut BenchmarkGroup<impl Measurement>) {
    for num_vars in NUM_VARS_RANGE {
        let n = 1 << num_vars;
        let k = n / 4; // Use n/4 as shift amount

        let mut rng = std_rng();

        // Setup
        let param = Pcs::setup(n, 1, &mut rng).unwrap();
        let (pp, _vp) = Pcs::trim(&param, n, 1).unwrap();

        // Create polynomial
        let evals: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();
        let poly = MultilinearPolynomial::new(evals);
        let comm = Pcs::commit(&pp, &poly).unwrap();

        group.bench_with_input(
            BenchmarkId::new("new_shift_open_prove", num_vars),
            &num_vars,
            |b, _| {
                b.iter(|| {
                    let mut transcript = Transcript::new(());
                    prove_shift_open::<Fr, Pcs>(
                        black_box(&pp),
                        black_box(&poly),
                        black_box(&comm),
                        black_box(k),
                        black_box(&mut transcript),
                    )
                    .unwrap()
                })
            },
        );
    }
}

/// Benchmark the NEW shift_open protocol verification
fn bench_new_shift_open_verify(group: &mut BenchmarkGroup<impl Measurement>) {
    for num_vars in NUM_VARS_RANGE {
        let n = 1 << num_vars;
        let k = n / 4;

        let mut rng = std_rng();

        // Setup
        let param = Pcs::setup(n, 1, &mut rng).unwrap();
        let (pp, vp) = Pcs::trim(&param, n, 1).unwrap();

        // Create polynomial and proof
        let evals: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();
        let poly = MultilinearPolynomial::new(evals);
        let comm = Pcs::commit(&pp, &poly).unwrap();

        let proof = {
            let mut transcript = Transcript::new(());
            prove_shift_open::<Fr, Pcs>(&pp, &poly, &comm, k, &mut transcript).unwrap();
            transcript.into_proof()
        };

        group.bench_with_input(
            BenchmarkId::new("new_shift_open_verify", num_vars),
            &num_vars,
            |b, _| {
                b.iter(|| {
                    let mut transcript = Transcript::from_proof((), black_box(proof.as_slice()));
                    verify_shift_open::<Fr, Pcs>(
                        black_box(&vp),
                        black_box(&comm),
                        black_box(k),
                        black_box(num_vars),
                        black_box(&mut transcript),
                    )
                    .unwrap()
                })
            },
        );
    }
}

/// Benchmark the OLD approach: full HyperPlonk circuit with Rotation constraints
fn bench_old_rotation_prove(group: &mut BenchmarkGroup<impl Measurement>) {
    for num_vars in NUM_VARS_RANGE.clone().take(3) {
        // Limit range for full circuit
        let mut rng = std_rng();

        // Create a vanilla plonk circuit (uses Rotation internally)
        let (circuit_info, circuit) =
            rand_vanilla_plonk_circuit::<Fr, Lexical>(num_vars, &mut rng, &mut rng);

        // Setup
        let param = Backend::setup(&circuit_info, &mut rng).unwrap();
        let (pp, _vp) = Backend::preprocess(&param, &circuit_info).unwrap();

        group.bench_with_input(
            BenchmarkId::new("old_rotation_prove", num_vars),
            &num_vars,
            |b, _| {
                b.iter(|| {
                    let mut transcript = Transcript::new(());
                    Backend::prove_with_shift(
                        black_box(&pp),
                        black_box(&circuit),
                        black_box(&mut transcript),
                        &mut std_rng(),
                    )
                    .unwrap()
                })
            },
        );
    }
}

/// Benchmark the OLD approach verification
fn bench_old_rotation_verify(group: &mut BenchmarkGroup<impl Measurement>) {
    for num_vars in NUM_VARS_RANGE.clone().take(3) {
        let mut rng = std_rng();

        let (circuit_info, circuit) =
            rand_vanilla_plonk_circuit::<Fr, Lexical>(num_vars, &mut rng, &mut rng);

        let param = Backend::setup(&circuit_info, &mut rng).unwrap();
        let (pp, vp) = Backend::preprocess(&param, &circuit_info).unwrap();

        // Generate proof
        let proof = {
            let mut transcript = Transcript::new(());
            Backend::prove_with_shift(&pp, &circuit, &mut transcript, &mut rng).unwrap();
            transcript.into_proof()
        };

        let instances = circuit.instances();

        group.bench_with_input(
            BenchmarkId::new("old_rotation_verify", num_vars),
            &num_vars,
            |b, _| {
                b.iter(|| {
                    let mut transcript = Transcript::from_proof((), black_box(proof.as_slice()));
                    Backend::verify_with_shift(
                        black_box(&vp),
                        black_box(instances),
                        black_box(&mut transcript),
                        &mut std_rng(),
                    )
                    .unwrap()
                })
            },
        );
    }
}

fn bench_shift_open(c: &mut Criterion) {
    let mut group = c.benchmark_group("shift_open");
    group.sample_size(10);

    // New approach benchmarks
    bench_new_shift_open_prove(&mut group);
    bench_new_shift_open_verify(&mut group);

    // Old approach benchmarks (full circuit)
    bench_old_rotation_prove(&mut group);
    bench_old_rotation_verify(&mut group);

    group.finish();
}

criterion_group!(benches, bench_shift_open);
criterion_main!(benches);

