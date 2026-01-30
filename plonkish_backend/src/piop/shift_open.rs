//! SHIFT-OPEN Protocol Implementation
//!
//! Given:
//! - f: {0,1}^n → F, with MLE polynomial F and commitment C_f
//! - k: shift amount (public input)
//!
//! Prove:
//! - y' = G(r) where g(x) = f((x + k) mod 2^n), G = MLE of g
//!
//! Protocol:
//! 1. Verifier sends random point r
//! 2. Prover sends y = F(r), y' = G(r)
//! 3. Sum-check proves: y' = Σ_i f(i) · w_{k,r}(i) where w_{k,r}(i) = eq((i-k) mod 2^n, r)
//! 4. Sum-check reduces to random point u, prover opens F(u)
//! 5. Verifier computes w(u) and checks consistency

use crate::{
    pcs::PolynomialCommitmentScheme,
    piop::sum_check::{
        classic::{ClassicSumCheck, EvaluationsProver},
        SumCheck, VirtualPolynomial,
    },
    poly::multilinear::MultilinearPolynomial,
    util::{
        arithmetic::{Field, PrimeField},
        expression::{Expression, Query, Rotation},
        transcript::{TranscriptRead, TranscriptWrite},
    },
    Error,
};

/// Convert an index to its bit representation (little-endian)
#[inline]
fn idx_to_bits(idx: usize, num_vars: usize) -> Vec<bool> {
    (0..num_vars).map(|i| (idx >> i) & 1 == 1).collect()
}

/// Evaluate eq(bits, point) where bits is a boolean vector and point is a field vector
/// eq(a, b) = Π_i (a_i · b_i + (1 - a_i)(1 - b_i))
fn eq_eval_bool_field<F: Field>(bits: &[bool], point: &[F]) -> F {
    assert_eq!(bits.len(), point.len());
    bits.iter()
        .zip(point.iter())
        .fold(F::ONE, |acc, (&bit, y)| {
            if bit {
                acc * y
            } else {
                acc * (F::ONE - y)
            }
        })
}

/// Compute weight polynomial evaluations: w(i) = eq((i - k) mod 2^n, r) for i ∈ {0,1}^n
fn compute_weight_evals<F: Field>(num_vars: usize, k: usize, r: &[F]) -> Vec<F> {
    let n = 1usize << num_vars;
    (0..n)
        .map(|i| {
            let shifted_idx = (i + n - (k % n)) % n; // (i - k) mod n, handling negative
            let bits = idx_to_bits(shifted_idx, num_vars);
            eq_eval_bool_field(&bits, r)
        })
        .collect()
}

/// Compute weight evaluation at arbitrary point u:
/// w(u) = Σ_{i∈{0,1}^n} eq(i, u) · eq((i - k) mod 2^n, r)
///
/// This is O(2^n) but correct. Can be made succinct with additional sum-check.
fn compute_weight_at_point<F: PrimeField>(num_vars: usize, k: usize, r: &[F], u: &[F]) -> F {
    let n = 1usize << num_vars;
    (0..n)
        .map(|i| {
            let bits_i = idx_to_bits(i, num_vars);
            let eq_i_u = eq_eval_bool_field(&bits_i, u);

            let shifted_idx = (i + n - (k % n)) % n;
            let bits_shifted = idx_to_bits(shifted_idx, num_vars);
            let eq_shifted_r = eq_eval_bool_field(&bits_shifted, r);

            eq_i_u * eq_shifted_r
        })
        .fold(F::ZERO, |acc, x| acc + x)
}

/// Prove shift-open protocol
///
/// Given polynomial f with commitment C_f, prove that G(r) = y' where:
/// - g(x) = f((x + k) mod 2^n)
/// - G is the MLE of g
/// - r is a random point chosen by verifier
pub fn prove_shift_open<F, Pcs>(
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
    let num_vars = poly.num_vars();
    let n = 1usize << num_vars;

    if num_vars == 0 {
        return Err(Error::InvalidPcsParam(
            "Polynomial must have at least 1 variable".to_string(),
        ));
    }

    // Domain separation: absorb k and num_vars
    transcript.common_field_element(&F::from(k as u64))?;
    transcript.common_field_element(&F::from(num_vars as u64))?;

    // Step 1: Squeeze random point r
    let r = transcript.squeeze_challenges(num_vars);

    // Step 2: Compute and send y = F(r)
    let y = poly.evaluate(&r);
    transcript.write_field_element(&y)?;

    // Step 3: Compute shifted polynomial g(x) = f((x + k) mod 2^n)
    let shifted_evals: Vec<F> = (0..n)
        .map(|x| {
            let shifted_idx = (x + (k % n)) % n;
            poly[shifted_idx]
        })
        .collect();
    let shifted_poly = MultilinearPolynomial::new(shifted_evals);

    // Compute and send y' = G(r)
    let y_prime = shifted_poly.evaluate(&r);
    transcript.write_field_element(&y_prime)?;

    // Step 4: Construct weight polynomial W
    // w(i) = eq((i - k) mod 2^n, r)
    let weight_evals = compute_weight_evals(num_vars, k, &r);
    let weight_poly = MultilinearPolynomial::new(weight_evals);

    // Sanity check: y' should equal Σ_i f(i) · w(i)
    if cfg!(feature = "sanity-check") {
        let sum: F = (0..n)
            .map(|i| poly[i] * weight_poly[i])
            .fold(F::ZERO, |acc, x| acc + x);
        assert_eq!(sum, y_prime, "Sum-check claim mismatch");
    }

    // Step 5: Run sum-check for y' = Σ_i f(i) · w(i)
    // Expression: poly_0 * poly_1
    let f_expr = Expression::<F>::Polynomial(Query::new(0, Rotation::cur()));
    let w_expr = Expression::<F>::Polynomial(Query::new(1, Rotation::cur()));
    let expr = f_expr * w_expr;

    let polys: Vec<&MultilinearPolynomial<F>> = vec![poly, &weight_poly];
    let virtual_poly = VirtualPolynomial::new(&expr, polys.clone(), &[], &[]);

    // Use usize as the trivial Rotatable (no rotation support needed here)
    let (final_eval, u, _evals) = ClassicSumCheck::<EvaluationsProver<F>, usize>::prove(
        &(),
        num_vars,
        virtual_poly,
        y_prime,
        transcript,
    )?;

    // Step 6: Get f(u) and w(u)
    let f_u = poly.evaluate(&u);
    let w_u = weight_poly.evaluate(&u);

    // Sanity check
    if cfg!(feature = "sanity-check") {
        assert_eq!(f_u * w_u, final_eval, "Final evaluation mismatch");
    }

    // Step 7: Send evaluations
    transcript.write_field_element(&f_u)?;
    transcript.write_field_element(&w_u)?;

    // Step 8: Open F at point u
    Pcs::open(pp, poly, comm, &u, &f_u, transcript)?;

    Ok(())
}

/// Verify shift-open protocol
///
/// Returns (y, y') = (F(r), G(r)) on success
pub fn verify_shift_open<F, Pcs>(
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
    if num_vars == 0 {
        return Err(Error::InvalidPcsParam(
            "Polynomial must have at least 1 variable".to_string(),
        ));
    }

    // Domain separation: absorb k and num_vars
    transcript.common_field_element(&F::from(k as u64))?;
    transcript.common_field_element(&F::from(num_vars as u64))?;

    // Step 1: Squeeze random point r (same as prover)
    let r = transcript.squeeze_challenges(num_vars);

    // Step 2: Read y = F(r) and y' = G(r)
    let y = transcript.read_field_element()?;
    let y_prime = transcript.read_field_element()?;

    // Step 3: Verify sum-check for y' = Σ_i f(i) · w(i)
    // Expression degree is 2 (product of two degree-1 polynomials)
    let degree = 2;
    let (final_eval, u) = ClassicSumCheck::<EvaluationsProver<F>, usize>::verify(
        &(),
        num_vars,
        degree,
        y_prime,
        transcript,
    )?;

    // Step 4: Read claimed evaluations
    let f_u = transcript.read_field_element()?;
    let w_u = transcript.read_field_element()?;

    // Step 5: Check sum-check final evaluation: f(u) · w(u) = final_eval
    if f_u * w_u != final_eval {
        return Err(Error::InvalidSumcheck(
            "Final evaluation check failed: f(u) * w(u) != claimed".to_string(),
        ));
    }

    // Step 6: Verify w(u) by direct computation
    // w(u) = Σ_{i∈{0,1}^n} eq(i, u) · eq((i-k) mod 2^n, r)
    let computed_w_u = compute_weight_at_point(num_vars, k, &r, &u);
    if w_u != computed_w_u {
        return Err(Error::InvalidSumcheck(
            "Weight evaluation mismatch: claimed w(u) != computed w(u)".to_string(),
        ));
    }

    // Step 7: Verify PCS opening
    Pcs::verify(vp, comm, &u, &f_u, transcript)?;

    Ok((y, y_prime))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        pcs::{multilinear::Zeromorph, univariate::UnivariateKzg},
        util::{
            test::{rand_vec, seeded_std_rng},
            transcript::{FieldTranscript, InMemoryTranscript, Keccak256Transcript},
        },
    };
    use halo2_curves::bn256::{Bn256, Fr};
    use rand::Rng;
    use std::io::Cursor;

    type Pcs = Zeromorph<UnivariateKzg<Bn256>>;
    type Transcript = Keccak256Transcript<Cursor<Vec<u8>>>;

    /// Brute-force compute G(r) where g(x) = f((x+k) mod 2^n)
    fn brute_force_shifted_eval(poly: &MultilinearPolynomial<Fr>, k: usize, r: &[Fr]) -> Fr {
        let num_vars = poly.num_vars();
        let n = 1usize << num_vars;

        // Create shifted polynomial
        let shifted_evals: Vec<Fr> = (0..n)
            .map(|x| {
                let shifted_idx = (x + (k % n)) % n;
                poly[shifted_idx]
            })
            .collect();
        let shifted_poly = MultilinearPolynomial::new(shifted_evals);

        shifted_poly.evaluate(r)
    }

    #[test]
    fn test_shift_open_basic() {
        let mut rng = seeded_std_rng();

        for num_vars in 2..8 {
            let n = 1usize << num_vars;
            let poly_size = n;

            // Setup PCS
            let param = Pcs::setup(poly_size, 1, &mut rng).unwrap();
            let (pp, vp) = Pcs::trim(&param, poly_size, 1).unwrap();

            // Random polynomial
            let evals: Vec<Fr> = rand_vec(n, &mut rng);
            let poly = MultilinearPolynomial::new(evals);

            // Commit
            let comm = Pcs::commit(&pp, &poly).unwrap();

            // Random shift
            let k: usize = rng.gen_range(0..n);

            // Prove
            let proof = {
                let mut transcript = Transcript::new(());
                prove_shift_open::<Fr, Pcs>(&pp, &poly, &comm, k, &mut transcript).unwrap();
                transcript.into_proof()
            };

            // Verify
            let result = {
                let mut transcript = Transcript::from_proof((), proof.as_slice());
                verify_shift_open::<Fr, Pcs>(&vp, &comm, k, num_vars, &mut transcript)
            };

            assert!(result.is_ok(), "Verification failed for num_vars={num_vars}, k={k}");

            // Check that y' matches brute-force computation
            let (y, y_prime) = result.unwrap();

            // Re-compute r by replaying transcript
            let mut transcript = Transcript::new(());
            transcript.common_field_element(&Fr::from(k as u64)).unwrap();
            transcript
                .common_field_element(&Fr::from(num_vars as u64))
                .unwrap();
            let r = transcript.squeeze_challenges(num_vars);

            // Verify y = F(r)
            assert_eq!(y, poly.evaluate(&r), "y != F(r)");

            // Verify y' = G(r) via brute force
            let expected_y_prime = brute_force_shifted_eval(&poly, k, &r);
            assert_eq!(y_prime, expected_y_prime, "y' != brute-force G(r)");
        }
    }

    #[test]
    fn test_shift_open_k_zero() {
        // When k = 0, shifted polynomial equals original
        let mut rng = seeded_std_rng();
        let num_vars = 4;
        let n = 1usize << num_vars;

        let param = Pcs::setup(n, 1, &mut rng).unwrap();
        let (pp, vp) = Pcs::trim(&param, n, 1).unwrap();

        let evals: Vec<Fr> = rand_vec(n, &mut rng);
        let poly = MultilinearPolynomial::new(evals);
        let comm = Pcs::commit(&pp, &poly).unwrap();

        let k = 0;

        let proof = {
            let mut transcript = Transcript::new(());
            prove_shift_open::<Fr, Pcs>(&pp, &poly, &comm, k, &mut transcript).unwrap();
            transcript.into_proof()
        };

        let result = {
            let mut transcript = Transcript::from_proof((), proof.as_slice());
            verify_shift_open::<Fr, Pcs>(&vp, &comm, k, num_vars, &mut transcript)
        };

        let (y, y_prime) = result.unwrap();
        // When k = 0, y' should equal y
        assert_eq!(y, y_prime, "When k=0, y should equal y'");
    }

    #[test]
    fn test_shift_open_full_rotation() {
        // When k = n, shifted polynomial equals original (full rotation)
        let mut rng = seeded_std_rng();
        let num_vars = 4;
        let n = 1usize << num_vars;

        let param = Pcs::setup(n, 1, &mut rng).unwrap();
        let (pp, vp) = Pcs::trim(&param, n, 1).unwrap();

        let evals: Vec<Fr> = rand_vec(n, &mut rng);
        let poly = MultilinearPolynomial::new(evals);
        let comm = Pcs::commit(&pp, &poly).unwrap();

        let k = n; // Full rotation

        let proof = {
            let mut transcript = Transcript::new(());
            prove_shift_open::<Fr, Pcs>(&pp, &poly, &comm, k, &mut transcript).unwrap();
            transcript.into_proof()
        };

        let result = {
            let mut transcript = Transcript::from_proof((), proof.as_slice());
            verify_shift_open::<Fr, Pcs>(&vp, &comm, k, num_vars, &mut transcript)
        };

        let (y, y_prime) = result.unwrap();
        assert_eq!(y, y_prime, "When k=n, y should equal y'");
    }

    #[test]
    fn test_shift_open_tampered_y_prime_fails() {
        let mut rng = seeded_std_rng();
        let num_vars = 4;
        let n = 1usize << num_vars;

        let param = Pcs::setup(n, 1, &mut rng).unwrap();
        let (pp, vp) = Pcs::trim(&param, n, 1).unwrap();

        let evals: Vec<Fr> = rand_vec(n, &mut rng);
        let poly = MultilinearPolynomial::new(evals);
        let comm = Pcs::commit(&pp, &poly).unwrap();

        let k: usize = rng.gen_range(1..n);

        // Generate valid proof
        let mut proof = {
            let mut transcript = Transcript::new(());
            prove_shift_open::<Fr, Pcs>(&pp, &poly, &comm, k, &mut transcript).unwrap();
            transcript.into_proof()
        };

        // Tamper with y' (second field element after domain separation)
        // Field elements are 32 bytes in bn256
        let field_elem_size = 32;
        let y_prime_offset = field_elem_size; // y is first, y' is second
        if proof.len() > y_prime_offset + field_elem_size {
            // Flip a byte in y'
            proof[y_prime_offset] ^= 0xFF;
        }

        // Verification should fail
        let result = {
            let mut transcript = Transcript::from_proof((), proof.as_slice());
            verify_shift_open::<Fr, Pcs>(&vp, &comm, k, num_vars, &mut transcript)
        };

        assert!(
            result.is_err(),
            "Verification should fail with tampered y'"
        );
    }

    #[test]
    fn test_shift_open_wrong_k_fails() {
        let mut rng = seeded_std_rng();
        let num_vars = 4;
        let n = 1usize << num_vars;

        let param = Pcs::setup(n, 1, &mut rng).unwrap();
        let (pp, vp) = Pcs::trim(&param, n, 1).unwrap();

        let evals: Vec<Fr> = rand_vec(n, &mut rng);
        let poly = MultilinearPolynomial::new(evals);
        let comm = Pcs::commit(&pp, &poly).unwrap();

        let k: usize = rng.gen_range(1..n / 2);
        let wrong_k = k + 1;

        // Generate proof with k
        let proof = {
            let mut transcript = Transcript::new(());
            prove_shift_open::<Fr, Pcs>(&pp, &poly, &comm, k, &mut transcript).unwrap();
            transcript.into_proof()
        };

        // Verify with wrong_k - should fail
        let result = {
            let mut transcript = Transcript::from_proof((), proof.as_slice());
            verify_shift_open::<Fr, Pcs>(&vp, &comm, wrong_k, num_vars, &mut transcript)
        };

        assert!(
            result.is_err(),
            "Verification should fail with wrong k"
        );
    }

    #[test]
    fn test_helper_functions() {
        // Test idx_to_bits
        assert_eq!(idx_to_bits(0, 3), vec![false, false, false]);
        assert_eq!(idx_to_bits(1, 3), vec![true, false, false]);
        assert_eq!(idx_to_bits(5, 3), vec![true, false, true]); // 5 = 101 in binary
        assert_eq!(idx_to_bits(7, 3), vec![true, true, true]);

        // Test eq_eval_bool_field
        let one = Fr::ONE;
        let zero = Fr::ZERO;

        // eq([0,0], [0,0]) = 1
        assert_eq!(
            eq_eval_bool_field(&[false, false], &[zero, zero]),
            Fr::ONE
        );

        // eq([1,1], [1,1]) = 1
        assert_eq!(eq_eval_bool_field(&[true, true], &[one, one]), Fr::ONE);

        // eq([0,1], [1,0]) should be close to 0 when evaluated at boolean points
        assert_eq!(eq_eval_bool_field(&[false, true], &[one, zero]), Fr::ZERO);
    }

    #[test]
    fn test_weight_computation() {
        let num_vars = 3;
        let n = 1usize << num_vars;
        let mut rng = seeded_std_rng();

        let r: Vec<Fr> = rand_vec(num_vars, &mut rng);
        let k = 2;

        // Compute weight at a boolean point
        let u = vec![Fr::ZERO, Fr::ONE, Fr::ZERO]; // u = 2 in little-endian

        let w_u = compute_weight_at_point(num_vars, k, &r, &u);

        // Verify: w(u) = Σ_i eq(i, u) · eq((i-k) mod n, r)
        // At boolean u, only one term is non-zero: when i = u
        let u_idx = 2; // u = [0,1,0] = 2
        let shifted_idx = (u_idx + n - k) % n; // (2 - 2) mod 8 = 0
        let expected = eq_eval_bool_field(&idx_to_bits(shifted_idx, num_vars), &r);

        assert_eq!(w_u, expected, "Weight at boolean point mismatch");
    }
}

