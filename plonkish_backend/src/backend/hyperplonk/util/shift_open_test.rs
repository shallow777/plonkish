//! Shift-Open Gadget Integration Tests
//!
//! This module provides E2E tests for the NEW shift_open protocol (piop::shift_open),
//! which is INDEPENDENT from the existing Rotation mechanism.
//!
//! ## Key Differences from Existing Rotation Mechanism:
//!
//! | Feature | Existing (Rotation) | New (shift_open) |
//! |---------|---------------------|------------------|
//! | k value | Static (compile-time) | Dynamic (runtime) |
//! | Method | Expression with Rotation | Sum-check protocol |
//! | Commitment | Needs shifted poly commitment | Only original poly |
//! | Verification | PCS open_shift | Weight function check |
//!
//! ## Integration Pattern:
//!
//! The shift_open gadget can be called AFTER the main circuit proof to add
//! additional shift-relation proofs without modifying existing circuit constraints.
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                     Proving Pipeline                        │
//! ├─────────────────────────────────────────────────────────────┤
//! │  1. Circuit Setup & Preprocessing                           │
//! │  2. Witness Generation                                      │
//! │  3. Main Circuit Proof (HyperPlonk prove)                   │
//! │  4. [NEW] Shift-Open Gadget Proof (for each shift claim)    │
//! │     - k from public input (instance)                        │
//! │     - prove_shift_open(pp, poly, comm, k, transcript)       │
//! └─────────────────────────────────────────────────────────────┘
//! ```

use crate::{
    backend::{
        hyperplonk::util::common::Permutation,
        mock::MockCircuit,
        PlonkishCircuit, PlonkishCircuitInfo,
    },
    piop::shift_open::{prove_shift_open, verify_shift_open},
    pcs::PolynomialCommitmentScheme,
    poly::multilinear::MultilinearPolynomial,
    util::{
        arithmetic::PrimeField,
        expression::{rotate::Rotatable, Expression},
        transcript::{TranscriptRead, TranscriptWrite},
    },
    Error,
};

use rand::RngCore;

/// Information for a shift-open claim
/// 
/// This represents a claim that a polynomial F has a cyclic shift relation:
/// - G(r) = y' where g(x) = f((x + k) mod 2^n)
/// - G is the MLE of g, but is NOT explicitly committed
#[derive(Clone, Debug)]
pub struct ShiftOpenClaim<F> {
    /// The shift amount (public input)
    pub k: usize,
    /// The polynomial being shifted
    pub poly: MultilinearPolynomial<F>,
    /// Expected y' value (optional, for testing)
    pub expected_y_prime: Option<F>,
}

/// Prove a shift-open claim using the NEW sum-check based protocol
///
/// This function demonstrates how to integrate the shift_open gadget
/// into an existing proving pipeline. It should be called AFTER the
/// main circuit proof.
///
/// # Arguments
/// * `pp` - PCS prover parameters
/// * `poly` - The polynomial to prove shift relation for
/// * `comm` - Commitment to the polynomial
/// * `k` - Shift amount (from public input / instance)
/// * `transcript` - Transcript for Fiat-Shamir
///
/// # Returns
/// Ok(()) on success, Error on failure
pub fn prove_shift_claim<F, Pcs>(
    pp: &Pcs::ProverParam,
    poly: &MultilinearPolynomial<F>,
    comm: &Pcs::Commitment,
    k: usize,
    transcript: &mut impl TranscriptWrite<Pcs::CommitmentChunk, F>,
) -> Result<(), Error>
where
    F: PrimeField,
    Pcs: PolynomialCommitmentScheme<F, Polynomial = MultilinearPolynomial<F>>,
{
    prove_shift_open::<F, Pcs>(pp, poly, comm, k, transcript)
}

/// Verify a shift-open claim using the NEW sum-check based protocol
///
/// # Arguments
/// * `vp` - PCS verifier parameters
/// * `comm` - Commitment to the polynomial
/// * `k` - Shift amount (from public input / instance)
/// * `num_vars` - Number of variables in the polynomial
/// * `transcript` - Transcript for Fiat-Shamir
///
/// # Returns
/// (y, y') on success where y = F(r) and y' = G(r)
pub fn verify_shift_claim<F, Pcs>(
    vp: &Pcs::VerifierParam,
    comm: &Pcs::Commitment,
    k: usize,
    num_vars: usize,
    transcript: &mut impl TranscriptRead<Pcs::CommitmentChunk, F>,
) -> Result<(F, F), Error>
where
    F: PrimeField,
    Pcs: PolynomialCommitmentScheme<F, Polynomial = MultilinearPolynomial<F>>,
{
    verify_shift_open::<F, Pcs>(vp, comm, k, num_vars, transcript)
}

/// Creates a simple base circuit for testing shift_open gadget.
///
/// This circuit contains:
/// - instance[0]: k (shift amount as public input)
/// - witness[0]: the polynomial values
///
/// The circuit itself has a trivial constraint, but the shift_open gadget
/// will be invoked separately to prove the shift relation.
///
/// This demonstrates how k flows from public input to the shift_open protocol.
pub fn shift_open_base_circuit<F: PrimeField, R: Rotatable + From<usize>>(
    num_vars: usize,
    k: usize,
    _preprocess_rng: impl RngCore,
    mut witness_rng: impl RngCore,
) -> (PlonkishCircuitInfo<F>, impl PlonkishCircuit<F>, MultilinearPolynomial<F>) {
    let size = 1 << num_vars;
    let rotatable = R::from(num_vars);
    let usable_indices = rotatable.usable_indices();

    // Instance: k as public input
    let instances = vec![F::from(k as u64)];

    // Generate random polynomial values
    let poly_values: Vec<F> = (0..size)
        .map(|_| F::random(&mut witness_rng))
        .collect();
    let poly = MultilinearPolynomial::new(poly_values.clone());

    // Simple selector (all ones for valid rows)
    let q_enable: Vec<F> = (0..size)
        .map(|i| {
            if usable_indices.contains(&i) { F::ONE } else { F::ZERO }
        })
        .collect();

    // Trivial constraint: q_enable * 0 = 0 (always satisfied)
    // The actual shift verification is done by shift_open gadget
    let constraint = Expression::<F>::Constant(F::ZERO);

    let mut permutation = Permutation::default();
    if !usable_indices.is_empty() {
        permutation.copy((1, usable_indices[0]), (1, usable_indices[0]));
    }

    let circuit_info = PlonkishCircuitInfo {
        k: num_vars,
        num_instances: vec![1], // k as public input
        preprocess_polys: vec![q_enable],
        num_witness_polys: vec![1],
        num_challenges: vec![0],
        constraints: vec![constraint],
        lookups: Vec::new(),
        permutations: permutation.into_cycles(),
        max_degree: Some(1),
    };

    (
        circuit_info,
        MockCircuit::new(vec![instances], vec![poly_values]),
        poly,
    )
}

/// Brute-force compute G(r) where g(x) = f((x+k) mod 2^n)
/// Used for testing correctness
pub fn brute_force_shifted_eval<F: PrimeField>(
    poly: &MultilinearPolynomial<F>,
    k: usize,
    r: &[F],
) -> F {
    let num_vars = poly.num_vars();
    let n = 1usize << num_vars;

    // Create shifted polynomial
    let shifted_evals: Vec<F> = (0..n)
        .map(|x| {
            let shifted_idx = (x + (k % n)) % n;
            poly[shifted_idx]
        })
        .collect();
    let shifted_poly = MultilinearPolynomial::new(shifted_evals);

    shifted_poly.evaluate(r)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        pcs::{multilinear::Zeromorph, univariate::UnivariateKzg},
        util::{
            expression::rotate::Lexical,
            test::{rand_vec, seeded_std_rng},
            transcript::{FieldTranscript, InMemoryTranscript, Keccak256Transcript},
        },
    };
    use halo2_curves::bn256::{Bn256, Fr};
    use rand::Rng;
    use std::io::Cursor;

    type Pcs = Zeromorph<UnivariateKzg<Bn256>>;
    type Transcript = Keccak256Transcript<Cursor<Vec<u8>>>;

    /// E2E test: shift_open gadget with k as public input
    ///
    /// This test demonstrates the complete flow:
    /// 1. k is passed as instance[0] (public input)
    /// 2. Polynomial is committed
    /// 3. shift_open proves G(r) = y' where g(x) = f((x+k) mod 2^n)
    /// 4. Verifier checks the proof using k from public input
    ///
    /// NOTE: prove_shift_open internally does domain separation with k and num_vars,
    /// so we don't need to add them separately before calling the function.
    #[test]
    fn test_shift_open_gadget_e2e() {
        let mut rng = seeded_std_rng();

        for num_vars in 3..7 {
            let n = 1usize << num_vars;
            let k: usize = rng.gen_range(1..n);

            println!("Testing shift_open gadget: num_vars={}, k={}", num_vars, k);

            // Setup PCS
            let param = Pcs::setup(n, 1, &mut rng).unwrap();
            let (pp, vp) = Pcs::trim(&param, n, 1).unwrap();

            // Generate random polynomial
            let evals: Vec<Fr> = rand_vec(n, &mut rng);
            let poly = MultilinearPolynomial::new(evals);

            // Commit to polynomial
            let comm = Pcs::commit(&pp, &poly).unwrap();

            // Prove shift relation
            // prove_shift_open internally absorbs k and num_vars for domain separation
            let proof = {
                let mut transcript = Transcript::new(());
                prove_shift_claim::<Fr, Pcs>(&pp, &poly, &comm, k, &mut transcript).unwrap();
                transcript.into_proof()
            };

            // Verify shift relation
            let result = {
                let mut transcript = Transcript::from_proof((), proof.as_slice());
                verify_shift_claim::<Fr, Pcs>(&vp, &comm, k, num_vars, &mut transcript)
            };

            assert!(result.is_ok(), "Verification failed for num_vars={}, k={}", num_vars, k);

            let (y, y_prime) = result.unwrap();
            
            // Verify y' matches brute-force computation
            // Replay the same transcript state to get the same r
            let mut check_transcript = Transcript::new(());
            // Domain separation: k and num_vars (same as prove_shift_open does)
            check_transcript.common_field_element(&Fr::from(k as u64)).unwrap();
            check_transcript.common_field_element(&Fr::from(num_vars as u64)).unwrap();
            let r = check_transcript.squeeze_challenges(num_vars);
            
            let expected_y = poly.evaluate(&r);
            let expected_y_prime = brute_force_shifted_eval(&poly, k, &r);
            
            assert_eq!(y, expected_y, "y mismatch");
            assert_eq!(y_prime, expected_y_prime, "y' mismatch with brute force");

            println!("  ✓ Passed: y={:?}, y'={:?}", y, y_prime);
        }
    }

    /// Test shift_open with k=0 (identity)
    #[test]
    fn test_shift_open_gadget_k_zero() {
        let mut rng = seeded_std_rng();
        let num_vars = 4;
        let n = 1usize << num_vars;
        let k = 0;

        let param = Pcs::setup(n, 1, &mut rng).unwrap();
        let (pp, vp) = Pcs::trim(&param, n, 1).unwrap();

        let evals: Vec<Fr> = rand_vec(n, &mut rng);
        let poly = MultilinearPolynomial::new(evals);
        let comm = Pcs::commit(&pp, &poly).unwrap();

        let proof = {
            let mut transcript = Transcript::new(());
            prove_shift_claim::<Fr, Pcs>(&pp, &poly, &comm, k, &mut transcript).unwrap();
            transcript.into_proof()
        };

        let result = {
            let mut transcript = Transcript::from_proof((), proof.as_slice());
            verify_shift_claim::<Fr, Pcs>(&vp, &comm, k, num_vars, &mut transcript)
        };

        let (y, y_prime) = result.unwrap();
        
        // When k=0, y should equal y'
        assert_eq!(y, y_prime, "When k=0, y should equal y'");
    }

    /// Test shift_open with full rotation (k=n)
    #[test]
    fn test_shift_open_gadget_full_rotation() {
        let mut rng = seeded_std_rng();
        let num_vars = 4;
        let n = 1usize << num_vars;
        let k = n; // Full rotation

        let param = Pcs::setup(n, 1, &mut rng).unwrap();
        let (pp, vp) = Pcs::trim(&param, n, 1).unwrap();

        let evals: Vec<Fr> = rand_vec(n, &mut rng);
        let poly = MultilinearPolynomial::new(evals);
        let comm = Pcs::commit(&pp, &poly).unwrap();

        let proof = {
            let mut transcript = Transcript::new(());
            prove_shift_claim::<Fr, Pcs>(&pp, &poly, &comm, k, &mut transcript).unwrap();
            transcript.into_proof()
        };

        let result = {
            let mut transcript = Transcript::from_proof((), proof.as_slice());
            verify_shift_claim::<Fr, Pcs>(&vp, &comm, k, num_vars, &mut transcript)
        };

        let (y, y_prime) = result.unwrap();
        
        // When k=n, y should equal y' (full rotation = identity)
        assert_eq!(y, y_prime, "When k=n, y should equal y'");
    }

    /// Test that wrong k fails verification
    #[test]
    fn test_shift_open_gadget_wrong_k_fails() {
        let mut rng = seeded_std_rng();
        let num_vars = 4;
        let n = 1usize << num_vars;
        let k: usize = rng.gen_range(1..n/2);
        let wrong_k = k + 1;

        let param = Pcs::setup(n, 1, &mut rng).unwrap();
        let (pp, vp) = Pcs::trim(&param, n, 1).unwrap();

        let evals: Vec<Fr> = rand_vec(n, &mut rng);
        let poly = MultilinearPolynomial::new(evals);
        let comm = Pcs::commit(&pp, &poly).unwrap();

        // Prove with k
        let proof = {
            let mut transcript = Transcript::new(());
            prove_shift_claim::<Fr, Pcs>(&pp, &poly, &comm, k, &mut transcript).unwrap();
            transcript.into_proof()
        };

        // Verify with wrong_k - should fail because:
        // 1. Domain separation with wrong_k changes the random challenge r
        // 2. Weight function computation uses wrong_k
        let result = {
            let mut transcript = Transcript::from_proof((), proof.as_slice());
            verify_shift_claim::<Fr, Pcs>(&vp, &comm, wrong_k, num_vars, &mut transcript)
        };

        assert!(result.is_err(), "Verification should fail with wrong k");
    }

    /// Test base circuit with k as public input
    #[test]
    fn test_shift_open_base_circuit() {
        let num_vars = 4;
        let k = 3;

        let (circuit_info, circuit, poly) = shift_open_base_circuit::<Fr, Lexical>(
            num_vars,
            k,
            seeded_std_rng(),
            seeded_std_rng(),
        );

        // Verify circuit is well-formed
        assert!(circuit_info.is_well_formed());
        
        // Verify k is in instance
        assert_eq!(circuit.instances()[0][0], Fr::from(k as u64));
        
        // Verify polynomial is correct
        assert_eq!(poly.num_vars(), num_vars);
    }

    /// Integration test: base circuit + shift_open gadget
    ///
    /// This demonstrates how shift_open integrates with existing circuits:
    /// 1. Base circuit defines k as public input (instance[0])
    /// 2. Polynomial commitment happens during circuit proving
    /// 3. shift_open gadget is called AFTER main circuit proof
    /// 4. k from instance is used directly in shift_open
    #[test]
    fn test_full_integration() {
        let mut rng = seeded_std_rng();
        let num_vars = 4;
        let n = 1usize << num_vars;
        let k: usize = rng.gen_range(1..n);

        // Create base circuit with k as public input
        let (_circuit_info, circuit, poly) = shift_open_base_circuit::<Fr, Lexical>(
            num_vars,
            k,
            seeded_std_rng(),
            seeded_std_rng(),
        );

        // Extract k from public input - this is how verifier gets k
        let k_from_instance = circuit.instances()[0][0];
        assert_eq!(k_from_instance, Fr::from(k as u64));

        // Setup PCS for shift_open gadget
        let param = Pcs::setup(n, 1, &mut rng).unwrap();
        let (pp, vp) = Pcs::trim(&param, n, 1).unwrap();

        // Commit to polynomial
        let comm = Pcs::commit(&pp, &poly).unwrap();

        // Prove shift relation
        // Note: prove_shift_open internally does domain separation with k
        let proof = {
            let mut transcript = Transcript::new(());
            prove_shift_claim::<Fr, Pcs>(&pp, &poly, &comm, k, &mut transcript).unwrap();
            transcript.into_proof()
        };

        // Verify shift relation
        let result = {
            let mut transcript = Transcript::from_proof((), proof.as_slice());
            verify_shift_claim::<Fr, Pcs>(&vp, &comm, k, num_vars, &mut transcript)
        };

        assert!(result.is_ok(), "Full integration test failed");
        
        let (y, y_prime) = result.unwrap();
        
        // Verify by replaying transcript to get same random point r
        let expected_y_prime = brute_force_shifted_eval(&poly, k, &{
            let mut t = Transcript::new(());
            // Same domain separation as prove_shift_open
            t.common_field_element(&Fr::from(k as u64)).unwrap();
            t.common_field_element(&Fr::from(num_vars as u64)).unwrap();
            t.squeeze_challenges(num_vars)
        });
        
        let expected_y = poly.evaluate(&{
            let mut t = Transcript::new(());
            t.common_field_element(&Fr::from(k as u64)).unwrap();
            t.common_field_element(&Fr::from(num_vars as u64)).unwrap();
            t.squeeze_challenges(num_vars)
        });
        
        assert_eq!(y, expected_y, "Integration test: y mismatch");
        assert_eq!(y_prime, expected_y_prime, "Integration test: y' mismatch");
        
        println!("Integration test passed: k={}, y={:?}, y'={:?}", k, y, y_prime);
    }

    /// Performance comparison test: NEW (shift_open) vs OLD (Rotation) approaches
    ///
    /// This test measures the time taken by each approach for proving and verifying
    /// shift relations. Run with:
    ///   cargo test --package plonkish_backend test_shift_performance_comparison -- --nocapture --ignored
    #[test]
    #[ignore] // Run with --ignored flag to execute
    fn test_shift_performance_comparison() {
        use crate::backend::{
            hyperplonk::{util::rand_vanilla_plonk_circuit, HyperPlonk},
            PlonkishBackend,
        };
        use std::time::Instant;

        println!("\n╔══════════════════════════════════════════════════════════════════════════════╗");
        println!("║           SHIFT PROOF PERFORMANCE COMPARISON: NEW vs OLD                     ║");
        println!("╠══════════════════════════════════════════════════════════════════════════════╣");
        println!("║ NEW: piop::shift_open (sum-check protocol, dynamic k)                        ║");
        println!("║ OLD: Rotation mechanism + batch_open_for_shift (static k, full circuit)      ║");
        println!("╚══════════════════════════════════════════════════════════════════════════════╝\n");

        type Backend = HyperPlonk<Pcs>;

        // Test parameters
        let test_cases = vec![
            (8, 4),    // num_vars=8, k=4
            (10, 8),   // num_vars=10, k=8
            (12, 16),  // num_vars=12, k=16
        ];

        println!("┌────────────┬─────┬───────────────────────────────────────────────────────────┐");
        println!("│  num_vars  │  k  │                         Results                          │");
        println!("├────────────┼─────┼───────────────────────────────────────────────────────────┤");

        for (num_vars, k) in test_cases {
            let n = 1usize << num_vars;
            let mut rng = seeded_std_rng();

            // ========== NEW APPROACH: shift_open ==========
            let param_new = Pcs::setup(n, 1, &mut rng).unwrap();
            let (pp_new, vp_new) = Pcs::trim(&param_new, n, 1).unwrap();

            let evals: Vec<Fr> = rand_vec(n, &mut rng);
            let poly = MultilinearPolynomial::new(evals);
            let comm = Pcs::commit(&pp_new, &poly).unwrap();

            // Warm up
            let _ = {
                let mut t = Transcript::new(());
                prove_shift_claim::<Fr, Pcs>(&pp_new, &poly, &comm, k, &mut t)
            };

            // Measure NEW prove
            let start = Instant::now();
            let proof_new = {
                let mut transcript = Transcript::new(());
                prove_shift_claim::<Fr, Pcs>(&pp_new, &poly, &comm, k, &mut transcript).unwrap();
                transcript.into_proof()
            };
            let new_prove_time = start.elapsed();

            // Measure NEW verify
            let start = Instant::now();
            let _result_new = {
                let mut transcript = Transcript::from_proof((), proof_new.as_slice());
                verify_shift_claim::<Fr, Pcs>(&vp_new, &comm, k, num_vars, &mut transcript).unwrap()
            };
            let new_verify_time = start.elapsed();

            // ========== OLD APPROACH: Full circuit with Rotation ==========
            let (circuit_info, circuit) = rand_vanilla_plonk_circuit::<Fr, Lexical>(
                num_vars, 
                seeded_std_rng(), 
                seeded_std_rng()
            );
            
            let param_old = Backend::setup(&circuit_info, seeded_std_rng()).unwrap();
            let (pp_old, vp_old) = Backend::preprocess(&param_old, &circuit_info).unwrap();

            // Warm up
            let _ = {
                let mut t = Transcript::new(());
                Backend::prove_with_shift(&pp_old, &circuit, &mut t, seeded_std_rng())
            };

            // Measure OLD prove
            let start = Instant::now();
            let proof_old = {
                let mut transcript = Transcript::new(());
                Backend::prove_with_shift(&pp_old, &circuit, &mut transcript, seeded_std_rng()).unwrap();
                transcript.into_proof()
            };
            let old_prove_time = start.elapsed();

            // Measure OLD verify
            let instances = circuit.instances();
            let start = Instant::now();
            let _result_old = {
                let mut transcript = Transcript::from_proof((), proof_old.as_slice());
                Backend::verify_with_shift(&vp_old, instances, &mut transcript, seeded_std_rng()).unwrap()
            };
            let old_verify_time = start.elapsed();

            // Calculate speedup
            let prove_speedup = old_prove_time.as_secs_f64() / new_prove_time.as_secs_f64();
            let verify_speedup = old_verify_time.as_secs_f64() / new_verify_time.as_secs_f64();

            println!("│    {:>4}    │ {:>3} │ NEW prove: {:>8.2?}, verify: {:>8.2?}             │",
                     num_vars, k, new_prove_time, new_verify_time);
            println!("│            │     │ OLD prove: {:>8.2?}, verify: {:>8.2?}             │",
                     old_prove_time, old_verify_time);
            println!("│            │     │ Speedup: prove {:.1}x, verify {:.1}x                     │",
                     prove_speedup, verify_speedup);
            println!("│            │     │ Proof size: NEW={} B, OLD={} B                   │",
                     proof_new.len(), proof_old.len());
            println!("├────────────┼─────┼───────────────────────────────────────────────────────────┤");
        }

        println!("└────────────┴─────┴───────────────────────────────────────────────────────────┘");
        println!("\nNote: The NEW approach is specifically for shift proofs only.");
        println!("      The OLD approach includes full circuit proving (constraints, permutations, etc.).");
        println!("      For fair comparison, consider that NEW is a standalone gadget.\n");
    }

    /// Generic test that works with ANY PCS implementing PolynomialCommitmentScheme
    /// 
    /// This demonstrates that shift_open is PCS-agnostic:
    /// - Only requires standard `open`/`verify` interfaces
    /// - No need for `open_shift` or `verify_shifted_evaluation`
    /// - When Brakedown or other PCS is added, just plug it in!
    fn test_shift_open_with_pcs<P>()
    where
        P: PolynomialCommitmentScheme<Fr, Polynomial = MultilinearPolynomial<Fr>>,
        Keccak256Transcript<Cursor<Vec<u8>>>: TranscriptWrite<P::CommitmentChunk, Fr>
            + TranscriptRead<P::CommitmentChunk, Fr>,
    {
        let mut rng = seeded_std_rng();
        let num_vars = 5;
        let n = 1usize << num_vars;
        let k = 3;

        // Setup with generic PCS
        let param = P::setup(n, 1, &mut rng).unwrap();
        let (pp, vp) = P::trim(&param, n, 1).unwrap();

        let evals: Vec<Fr> = rand_vec(n, &mut rng);
        let poly = MultilinearPolynomial::new(evals);
        let comm = P::commit(&pp, &poly).unwrap();

        // Prove using standard PCS interface
        let proof = {
            let mut transcript = Keccak256Transcript::new(());
            prove_shift_open::<Fr, P>(&pp, &poly, &comm, k, &mut transcript).unwrap();
            transcript.into_proof()
        };

        // Verify using standard PCS interface
        let result = {
            let mut transcript = Keccak256Transcript::from_proof((), proof.as_slice());
            verify_shift_open::<Fr, P>(&vp, &comm, k, num_vars, &mut transcript)
        };

        assert!(result.is_ok(), "shift_open should work with any PCS");
    }

    #[test]
    fn test_shift_open_with_zeromorph() {
        // Currently the only PCS in repo with full PolynomialCommitmentScheme impl
        test_shift_open_with_pcs::<Pcs>();
    }

    // When Brakedown PCS is implemented, add:
    // #[test]
    // fn test_shift_open_with_brakedown() {
    //     test_shift_open_with_pcs::<BrakedownPcs>();
    // }

    /// Quick timing test for shift_open only (no comparison)
    #[test]
    fn test_shift_open_timing() {
        use std::time::Instant;

        let num_vars = 10;
        let n = 1usize << num_vars;
        let k = 8;
        let mut rng = seeded_std_rng();

        // Setup
        let param = Pcs::setup(n, 1, &mut rng).unwrap();
        let (pp, vp) = Pcs::trim(&param, n, 1).unwrap();

        let evals: Vec<Fr> = rand_vec(n, &mut rng);
        let poly = MultilinearPolynomial::new(evals);
        let comm = Pcs::commit(&pp, &poly).unwrap();

        // Prove
        let start = Instant::now();
        let proof = {
            let mut transcript = Transcript::new(());
            prove_shift_claim::<Fr, Pcs>(&pp, &poly, &comm, k, &mut transcript).unwrap();
            transcript.into_proof()
        };
        let prove_time = start.elapsed();

        // Verify
        let start = Instant::now();
        let result = {
            let mut transcript = Transcript::from_proof((), proof.as_slice());
            verify_shift_claim::<Fr, Pcs>(&vp, &comm, k, num_vars, &mut transcript)
        };
        let verify_time = start.elapsed();

        assert!(result.is_ok());
        
        println!("\n[shift_open timing] num_vars={}, k={}", num_vars, k);
        println!("  Prove:  {:?}", prove_time);
        println!("  Verify: {:?}", verify_time);
        println!("  Proof size: {} bytes\n", proof.len());
    }
}
