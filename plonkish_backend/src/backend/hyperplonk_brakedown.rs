//! HyperPlonk backend with Brakedown PCS and shift_open protocol
//!
//! This backend uses:
//! - Brakedown PCS for polynomial commitments (hash-based, no trusted setup)
//! - piop::shift_open for handling shift/rotation constraints (PCS-agnostic)
//!
//! Key differences from standard HyperPlonk:
//! - Uses standard PCS open/verify instead of open_shift/verify_shifted_evaluation
//! - Shift constraints are proven via sum-check based shift_open protocol
//! - Compatible with any PCS that implements standard PolynomialCommitmentScheme trait

use crate::{
    backend::{
        hyperplonk::{
            preprocessor::{batch_size, preprocess},
            prover::{
                instance_polys, lookup_compressed_polys, lookup_h_polys, lookup_m_polys,
                permutation_z_polys, prove_zero_check,
            },
            verifier::verify_zero_check,
            HyperPlonkProverParam, HyperPlonkVerifierParam,
        },
        PlonkishBackend, PlonkishCircuit, PlonkishCircuitInfo,
    },
    pcs::{
        multilinear::{BrakedownPcs, BrakedownCommitment},
        PolynomialCommitmentScheme,
    },
    piop::shift_open::{prove_shift_open, verify_shift_open},
    poly::multilinear::MultilinearPolynomial,
    util::{
        arithmetic::{powers, PrimeField},
        chain, end_timer,
        expression::{
            rotate::Lexical,
            Expression, Rotation,
        },
        code::BrakedownSpec,
        hash::Hash,
        start_timer,
        transcript::{TranscriptRead, TranscriptWrite},
        DeserializeOwned, Itertools, Serialize,
    },
    Error,
};
use rand::RngCore;
use sha3::Keccak256;
use std::{fmt::Debug, hash::Hash as StdHash, iter, marker::PhantomData};

/// HyperPlonk backend specialized for Brakedown PCS with shift_open
#[derive(Clone, Debug)]
pub struct HyperPlonkBrakedown<F, S, H = Keccak256>
where
    F: PrimeField,
    S: BrakedownSpec,
    H: Hash,
{
    _marker: PhantomData<(F, S, H)>,
}

/// Shift constraint information extracted from circuit
#[derive(Clone, Debug)]
pub struct ShiftConstraint {
    /// Polynomial index that has rotation
    pub poly_idx: usize,
    /// Shift amount (rotation value)
    pub k: usize,
    /// Original rotation from expression
    pub rotation: Rotation,
}

impl<F, S, H> HyperPlonkBrakedown<F, S, H>
where
    F: PrimeField + StdHash + Serialize + DeserializeOwned,
    S: BrakedownSpec + Clone + Debug + Serialize + DeserializeOwned + Sync + Send + Default,
    H: Hash + Clone + Debug + Default + Sync + Send,
{
    /// Extract shift constraints from expression
    fn extract_shift_constraints(expression: &Expression<F>) -> Vec<ShiftConstraint> {
        let mut shifts = Vec::new();
        Self::collect_rotations(expression, &mut shifts);
        shifts
    }

    fn collect_rotations(expr: &Expression<F>, shifts: &mut Vec<ShiftConstraint>) {
        match expr {
            Expression::Polynomial(query) => {
                if query.rotation() != Rotation::cur() {
                    let k = query.rotation().0.unsigned_abs() as usize;
                    shifts.push(ShiftConstraint {
                        poly_idx: query.poly(),
                        k,
                        rotation: query.rotation(),
                    });
                }
            }
            Expression::Negated(inner) => Self::collect_rotations(inner, shifts),
            Expression::Sum(a, b) | Expression::Product(a, b) => {
                Self::collect_rotations(a, shifts);
                Self::collect_rotations(b, shifts);
            }
            Expression::Scaled(inner, _) => Self::collect_rotations(inner, shifts),
            Expression::DistributePowers(exprs, base) => {
                for e in exprs {
                    Self::collect_rotations(e, shifts);
                }
                Self::collect_rotations(base, shifts);
            }
            _ => {}
        }
    }

    /// Replace shift constraints with auxiliary witness polynomials
    /// Returns (modified expression, auxiliary polys to add)
    fn replace_shifts_with_aux(
        expression: &Expression<F>,
        polys: &[&MultilinearPolynomial<F>],
        num_vars: usize,
    ) -> (Expression<F>, Vec<MultilinearPolynomial<F>>) {
        let shifts = Self::extract_shift_constraints(expression);
        if shifts.is_empty() {
            return (expression.clone(), Vec::new());
        }

        // For each unique (poly_idx, rotation), create an auxiliary polynomial
        let mut aux_polys = Vec::new();
        let mut modified_expr = expression.clone();

        for shift in shifts.iter() {
            let n = 1usize << num_vars;
            let k = shift.k % n;
            let poly = polys[shift.poly_idx];

            // Compute shifted polynomial: g(x) = f((x + k) mod 2^n)
            let shifted_evals: Vec<F> = (0..n)
                .map(|x| {
                    let shifted_idx = (x + k) % n;
                    poly[shifted_idx]
                })
                .collect();
            aux_polys.push(MultilinearPolynomial::new(shifted_evals));
        }

        (modified_expr, aux_polys)
    }
}

impl<F, S, H> PlonkishBackend<F> for HyperPlonkBrakedown<F, S, H>
where
    F: PrimeField + StdHash + Serialize + DeserializeOwned,
    S: BrakedownSpec + Clone + Debug + Serialize + DeserializeOwned + Sync + Send + Default,
    H: Hash + Clone + Debug + Default + Sync + Send,
{
    type Pcs = BrakedownPcs<F, S, H>;
    type ProverParam = HyperPlonkProverParam<F, Self::Pcs>;
    type VerifierParam = HyperPlonkVerifierParam<F, Self::Pcs>;

    fn setup(
        circuit_info: &PlonkishCircuitInfo<F>,
        rng: impl RngCore,
    ) -> Result<<Self::Pcs as PolynomialCommitmentScheme<F>>::Param, Error> {
        assert!(circuit_info.is_well_formed());

        let num_vars = circuit_info.k;
        let poly_size = 1 << num_vars;
        let batch_size = batch_size(circuit_info);
        <Self::Pcs>::setup(poly_size, batch_size, rng)
    }

    fn preprocess(
        param: &<Self::Pcs as PolynomialCommitmentScheme<F>>::Param,
        circuit_info: &PlonkishCircuitInfo<F>,
    ) -> Result<(Self::ProverParam, Self::VerifierParam), Error> {
        preprocess(param, circuit_info, |pp, polys| {
            let comms = <Self::Pcs>::batch_commit(pp, &polys)?;
            Ok((polys, comms))
        })
    }

    /// Standard prove without shift handling
    fn prove(
        pp: &Self::ProverParam,
        circuit: &impl PlonkishCircuit<F>,
        transcript: &mut impl TranscriptWrite<<Self::Pcs as PolynomialCommitmentScheme<F>>::CommitmentChunk, F>,
        _: impl RngCore,
    ) -> Result<(), Error> {
        let instance_polys = {
            let instances = circuit.instances();
            for (num_instances, instances) in pp.num_instances.iter().zip_eq(instances) {
                assert_eq!(instances.len(), *num_instances);
                for instance in instances.iter() {
                    transcript.common_field_element(instance)?;
                }
            }
            instance_polys::<_, Lexical>(pp.num_vars, instances)
        };

        // Round 0..n
        let mut witness_polys = Vec::with_capacity(pp.num_witness_polys.iter().sum());
        let mut witness_comms = Vec::with_capacity(witness_polys.len());
        let mut challenges = Vec::with_capacity(pp.num_challenges.iter().sum::<usize>() + 4);
        for (round, (num_witness_polys, num_challenges)) in pp
            .num_witness_polys
            .iter()
            .zip_eq(pp.num_challenges.iter())
            .enumerate()
        {
            let timer = start_timer(|| format!("witness_collector-{round}"));
            let polys = circuit
                .synthesize(round, &challenges)?
                .into_iter()
                .map(MultilinearPolynomial::new)
                .collect_vec();
            assert_eq!(polys.len(), *num_witness_polys);
            end_timer(timer);

            witness_comms.extend(<Self::Pcs>::batch_commit_and_write(&pp.pcs, &polys, transcript)?);
            witness_polys.extend(polys);
            challenges.extend(transcript.squeeze_challenges(*num_challenges));
        }
        let polys = chain![&instance_polys, &pp.preprocess_polys, &witness_polys].collect_vec();

        // Round n
        let beta = transcript.squeeze_challenge();

        let timer = start_timer(|| format!("lookup_compressed_polys-{}", pp.lookups.len()));
        let lookup_compressed_polys = {
            let max_lookup_width = pp.lookups.iter().map(Vec::len).max().unwrap_or_default();
            let betas = powers(beta).take(max_lookup_width).collect_vec();
            lookup_compressed_polys::<_, Lexical>(&pp.lookups, &polys, &challenges, &betas)
        };
        end_timer(timer);
        let timer = start_timer(|| format!("lookup_m_polys-{}", pp.lookups.len()));
        let lookup_m_polys = lookup_m_polys(&lookup_compressed_polys)?;
        end_timer(timer);

        let lookup_m_comms = <Self::Pcs>::batch_commit_and_write(&pp.pcs, &lookup_m_polys, transcript)?;

        // Round n+1
        let gamma = transcript.squeeze_challenge();

        let timer = start_timer(|| format!("lookup_h_polys-{}", pp.lookups.len()));
        let lookup_h_polys = lookup_h_polys(&lookup_compressed_polys, &lookup_m_polys, &gamma);
        end_timer(timer);

        let timer = start_timer(|| format!("permutation_z_polys-{}", pp.permutation_polys.len()));
        let permutation_z_polys = permutation_z_polys::<_, Lexical>(
            pp.num_permutation_z_polys,
            &pp.permutation_polys,
            &polys,
            &beta,
            &gamma,
        );
        end_timer(timer);

        let lookup_h_permutation_z_polys =
            chain![lookup_h_polys.iter(), permutation_z_polys.iter()].collect_vec();
        let lookup_h_permutation_z_comms =
            <Self::Pcs>::batch_commit_and_write(&pp.pcs, lookup_h_permutation_z_polys.clone(), transcript)?;

        // Round n+2
        let alpha = transcript.squeeze_challenge();
        let y = transcript.squeeze_challenges(pp.num_vars);

        let polys = chain![
            polys,
            pp.permutation_polys.iter().map(|(_, poly)| poly),
            lookup_m_polys.iter(),
            lookup_h_permutation_z_polys,
        ]
        .collect_vec();
        challenges.extend([beta, gamma, alpha]);
        let (points, evals) = prove_zero_check(
            pp.num_instances.len(),
            &pp.expression,
            &polys,
            challenges,
            y,
            transcript,
        )?;

        // PCS open (standard, no shift)
        let dummy_comm = <Self::Pcs as PolynomialCommitmentScheme<F>>::Commitment::default();
        let comms = chain![
            iter::repeat(&dummy_comm).take(pp.num_instances.len()),
            &pp.preprocess_comms,
            &witness_comms,
            &pp.permutation_comms,
            &lookup_m_comms,
            &lookup_h_permutation_z_comms,
        ]
        .collect_vec();
        let timer = start_timer(|| format!("pcs_batch_open-{}", evals.len()));
        <Self::Pcs>::batch_open(&pp.pcs, polys, comms, &points, &evals, transcript)?;
        end_timer(timer);

        Ok(())
    }

    fn verify(
        vp: &Self::VerifierParam,
        instances: &[Vec<F>],
        transcript: &mut impl TranscriptRead<<Self::Pcs as PolynomialCommitmentScheme<F>>::CommitmentChunk, F>,
        _: impl RngCore,
    ) -> Result<(), Error> {
        for (num_instances, instances) in vp.num_instances.iter().zip_eq(instances) {
            assert_eq!(instances.len(), *num_instances);
            for instance in instances.iter() {
                transcript.common_field_element(instance)?;
            }
        }

        // Round 0..n
        let mut witness_comms = Vec::with_capacity(vp.num_witness_polys.iter().sum());
        let mut challenges = Vec::with_capacity(vp.num_challenges.iter().sum::<usize>() + 4);
        for (num_polys, num_challenges) in
            vp.num_witness_polys.iter().zip_eq(vp.num_challenges.iter())
        {
            witness_comms.extend(<Self::Pcs>::read_commitments(&vp.pcs, *num_polys, transcript)?);
            challenges.extend(transcript.squeeze_challenges(*num_challenges));
        }

        // Round n
        let beta = transcript.squeeze_challenge();
        let lookup_m_comms = <Self::Pcs>::read_commitments(&vp.pcs, vp.num_lookups, transcript)?;

        // Round n+1
        let gamma = transcript.squeeze_challenge();
        let lookup_h_permutation_z_comms = <Self::Pcs>::read_commitments(
            &vp.pcs,
            vp.num_lookups + vp.num_permutation_z_polys,
            transcript,
        )?;

        // Round n+2
        let alpha = transcript.squeeze_challenge();
        let y = transcript.squeeze_challenges(vp.num_vars);

        challenges.extend([beta, gamma, alpha]);
        let (points, evals) = verify_zero_check(
            vp.num_vars,
            &vp.expression,
            instances,
            &challenges,
            &y,
            transcript,
        )?;

        // PCS verify (standard, no shift)
        let dummy_comm = <Self::Pcs as PolynomialCommitmentScheme<F>>::Commitment::default();
        let comms = chain![
            iter::repeat(&dummy_comm).take(vp.num_instances.len()),
            &vp.preprocess_comms,
            &witness_comms,
            vp.permutation_comms.iter().map(|(_, comm)| comm),
            &lookup_m_comms,
            &lookup_h_permutation_z_comms,
        ]
        .collect_vec();
        <Self::Pcs>::batch_verify(&vp.pcs, comms, &points, &evals, transcript)?;

        Ok(())
    }

    /// Prove with shift constraints using shift_open protocol
    ///
    /// This is the key method that differentiates from standard HyperPlonk:
    /// Instead of using PCS::open_shift, we use piop::shift_open which works
    /// with any PCS that implements standard open/verify.
    fn prove_with_shift(
        pp: &Self::ProverParam,
        circuit: &impl PlonkishCircuit<F>,
        transcript: &mut impl TranscriptWrite<<Self::Pcs as PolynomialCommitmentScheme<F>>::CommitmentChunk, F>,
        _: impl RngCore,
    ) -> Result<(), Error> {
        let instance_polys = {
            let instances = circuit.instances();
            for (num_instances, instances) in pp.num_instances.iter().zip_eq(instances) {
                assert_eq!(instances.len(), *num_instances);
                for instance in instances.iter() {
                    transcript.common_field_element(instance)?;
                }
            }
            instance_polys::<_, Lexical>(pp.num_vars, instances)
        };

        // Round 0..n
        let mut witness_polys = Vec::with_capacity(pp.num_witness_polys.iter().sum());
        let mut witness_comms = Vec::with_capacity(witness_polys.len());
        let mut challenges = Vec::with_capacity(pp.num_challenges.iter().sum::<usize>() + 4);
        for (round, (num_witness_polys, num_challenges)) in pp
            .num_witness_polys
            .iter()
            .zip_eq(pp.num_challenges.iter())
            .enumerate()
        {
            let timer = start_timer(|| format!("witness_collector-{round}"));
            let polys = circuit
                .synthesize(round, &challenges)?
                .into_iter()
                .map(MultilinearPolynomial::new)
                .collect_vec();
            assert_eq!(polys.len(), *num_witness_polys);
            end_timer(timer);

            witness_comms.extend(<Self::Pcs>::batch_commit_and_write(&pp.pcs, &polys, transcript)?);
            witness_polys.extend(polys);
            challenges.extend(transcript.squeeze_challenges(*num_challenges));
        }
        let polys = chain![&instance_polys, &pp.preprocess_polys, &witness_polys].collect_vec();

        // Extract shift constraints from expression
        let shift_constraints = Self::extract_shift_constraints(&pp.expression);

        // Round n
        let beta = transcript.squeeze_challenge();

        let timer = start_timer(|| format!("lookup_compressed_polys-{}", pp.lookups.len()));
        let lookup_compressed_polys = {
            let max_lookup_width = pp.lookups.iter().map(Vec::len).max().unwrap_or_default();
            let betas = powers(beta).take(max_lookup_width).collect_vec();
            lookup_compressed_polys::<_, Lexical>(&pp.lookups, &polys, &challenges, &betas)
        };
        end_timer(timer);
        let timer = start_timer(|| format!("lookup_m_polys-{}", pp.lookups.len()));
        let lookup_m_polys = lookup_m_polys(&lookup_compressed_polys)?;
        end_timer(timer);

        let lookup_m_comms = <Self::Pcs>::batch_commit_and_write(&pp.pcs, &lookup_m_polys, transcript)?;

        // Round n+1
        let gamma = transcript.squeeze_challenge();

        let timer = start_timer(|| format!("lookup_h_polys-{}", pp.lookups.len()));
        let lookup_h_polys = lookup_h_polys(&lookup_compressed_polys, &lookup_m_polys, &gamma);
        end_timer(timer);

        let timer = start_timer(|| format!("permutation_z_polys-{}", pp.permutation_polys.len()));
        let permutation_z_polys = permutation_z_polys::<_, Lexical>(
            pp.num_permutation_z_polys,
            &pp.permutation_polys,
            &polys,
            &beta,
            &gamma,
        );
        end_timer(timer);

        let lookup_h_permutation_z_polys =
            chain![lookup_h_polys.iter(), permutation_z_polys.iter()].collect_vec();
        let lookup_h_permutation_z_comms =
            <Self::Pcs>::batch_commit_and_write(&pp.pcs, lookup_h_permutation_z_polys.clone(), transcript)?;

        // Round n+2
        let alpha = transcript.squeeze_challenge();
        let y = transcript.squeeze_challenges(pp.num_vars);

        let all_polys = chain![
            polys.clone(),
            pp.permutation_polys.iter().map(|(_, poly)| poly),
            lookup_m_polys.iter(),
            lookup_h_permutation_z_polys.clone(),
        ]
        .collect_vec();
        challenges.extend([beta, gamma, alpha]);

        // Prove zero-check (without shift - we handle shifts separately)
        let (points, evals) = prove_zero_check(
            pp.num_instances.len(),
            &pp.expression,
            &all_polys,
            challenges.clone(),
            y.clone(),
            transcript,
        )?;

        // PCS batch open (standard openings)
        let dummy_comm = <Self::Pcs as PolynomialCommitmentScheme<F>>::Commitment::default();
        let comms = chain![
            iter::repeat(&dummy_comm).take(pp.num_instances.len()),
            &pp.preprocess_comms,
            &witness_comms,
            &pp.permutation_comms,
            &lookup_m_comms,
            &lookup_h_permutation_z_comms,
        ]
        .collect_vec();
        let timer = start_timer(|| format!("pcs_batch_open-{}", evals.len()));
        <Self::Pcs>::batch_open(&pp.pcs, all_polys.clone(), comms.clone(), &points, &evals, transcript)?;
        end_timer(timer);

        // Prove shift constraints using shift_open protocol
        let timer = start_timer(|| format!("shift_open_proofs-{}", shift_constraints.len()));
        for shift in &shift_constraints {
            if shift.poly_idx < all_polys.len() {
                let poly = all_polys[shift.poly_idx];
                let comm = if shift.poly_idx < comms.len() {
                    comms[shift.poly_idx]
                } else {
                    &dummy_comm
                };
                prove_shift_open::<F, Self::Pcs>(&pp.pcs, poly, comm, shift.k, transcript)?;
            }
        }
        end_timer(timer);

        Ok(())
    }

    fn verify_with_shift(
        vp: &Self::VerifierParam,
        instances: &[Vec<F>],
        transcript: &mut impl TranscriptRead<<Self::Pcs as PolynomialCommitmentScheme<F>>::CommitmentChunk, F>,
        _: impl RngCore,
    ) -> Result<(), Error> {
        for (num_instances, instances) in vp.num_instances.iter().zip_eq(instances) {
            assert_eq!(instances.len(), *num_instances);
            for instance in instances.iter() {
                transcript.common_field_element(instance)?;
            }
        }

        // Extract shift constraints
        let shift_constraints = Self::extract_shift_constraints(&vp.expression);

        // Round 0..n
        let mut witness_comms = Vec::with_capacity(vp.num_witness_polys.iter().sum());
        let mut challenges = Vec::with_capacity(vp.num_challenges.iter().sum::<usize>() + 4);
        for (num_polys, num_challenges) in
            vp.num_witness_polys.iter().zip_eq(vp.num_challenges.iter())
        {
            witness_comms.extend(<Self::Pcs>::read_commitments(&vp.pcs, *num_polys, transcript)?);
            challenges.extend(transcript.squeeze_challenges(*num_challenges));
        }

        // Round n
        let beta = transcript.squeeze_challenge();
        let lookup_m_comms = <Self::Pcs>::read_commitments(&vp.pcs, vp.num_lookups, transcript)?;

        // Round n+1
        let gamma = transcript.squeeze_challenge();
        let lookup_h_permutation_z_comms = <Self::Pcs>::read_commitments(
            &vp.pcs,
            vp.num_lookups + vp.num_permutation_z_polys,
            transcript,
        )?;

        // Round n+2
        let alpha = transcript.squeeze_challenge();
        let y = transcript.squeeze_challenges(vp.num_vars);

        challenges.extend([beta, gamma, alpha]);
        let (points, evals) = verify_zero_check(
            vp.num_vars,
            &vp.expression,
            instances,
            &challenges,
            &y,
            transcript,
        )?;

        // PCS batch verify (standard verifications)
        let dummy_comm = <Self::Pcs as PolynomialCommitmentScheme<F>>::Commitment::default();
        let comms: Vec<&BrakedownCommitment> = chain![
            iter::repeat(&dummy_comm).take(vp.num_instances.len()),
            &vp.preprocess_comms,
            &witness_comms,
            vp.permutation_comms.iter().map(|(_, comm)| comm),
            &lookup_m_comms,
            &lookup_h_permutation_z_comms,
        ]
        .collect_vec();
        <Self::Pcs>::batch_verify(&vp.pcs, comms.clone(), &points, &evals, transcript)?;

        // Verify shift constraints using shift_open protocol
        for shift in &shift_constraints {
            if shift.poly_idx < comms.len() {
                let comm = comms[shift.poly_idx];
                verify_shift_open::<F, Self::Pcs>(&vp.pcs, comm, shift.k, vp.num_vars, transcript)?;
            }
        }

        Ok(())
    }
}

/// Type alias for common Brakedown configuration
pub type HyperPlonkBrakedownSpec3<F> = HyperPlonkBrakedown<F, crate::util::code::BrakedownSpec3, Keccak256>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        backend::hyperplonk::util::{rand_vanilla_plonk_circuit, rand_vanilla_plonk_w_lookup_circuit},
        util::{
            code::BrakedownSpecTest,  // Use fast test config
            test::seeded_std_rng,
            transcript::{InMemoryTranscript, Keccak256Transcript},
        },
    };
    use halo2_curves::bn256::Fr;
    use std::io::Cursor;

    // Use BrakedownSpecTest for fast testing (weak security, DO NOT use in production!)
    type TestBackend = HyperPlonkBrakedown<Fr, BrakedownSpecTest, Keccak256>;

    #[test]
    fn test_hyperplonk_brakedown_vanilla_plonk() {
        let mut rng = seeded_std_rng();
        let num_vars = 4; // Small for fast testing
        
        let (circuit_info, circuit) =
            rand_vanilla_plonk_circuit::<_, Lexical>(num_vars, seeded_std_rng(), seeded_std_rng());

        // Setup
        let param = TestBackend::setup(&circuit_info, &mut rng).unwrap();
        let (pp, vp) = TestBackend::preprocess(&param, &circuit_info).unwrap();

        // Prove
        let proof = {
            let mut transcript = Keccak256Transcript::<Cursor<Vec<u8>>>::new(());
            TestBackend::prove(&pp, &circuit, &mut transcript, &mut rng).unwrap();
            transcript.into_proof()
        };

        // Verify
        let result = {
            let mut transcript = Keccak256Transcript::from_proof((), proof.as_slice());
            TestBackend::verify(&vp, circuit.instances(), &mut transcript, &mut rng)
        };

        assert!(result.is_ok(), "Verification failed: {:?}", result);
    }

    #[test]
    fn test_hyperplonk_brakedown_with_lookup() {
        let mut rng = seeded_std_rng();
        let num_vars = 4;
        
        let (circuit_info, circuit) =
            rand_vanilla_plonk_w_lookup_circuit::<_, Lexical>(num_vars, seeded_std_rng(), seeded_std_rng());

        let param = TestBackend::setup(&circuit_info, &mut rng).unwrap();
        let (pp, vp) = TestBackend::preprocess(&param, &circuit_info).unwrap();

        let proof = {
            let mut transcript = Keccak256Transcript::<Cursor<Vec<u8>>>::new(());
            TestBackend::prove(&pp, &circuit, &mut transcript, &mut rng).unwrap();
            transcript.into_proof()
        };

        let result = {
            let mut transcript = Keccak256Transcript::from_proof((), proof.as_slice());
            TestBackend::verify(&vp, circuit.instances(), &mut transcript, &mut rng)
        };

        assert!(result.is_ok(), "Verification with lookup failed: {:?}", result);
    }
}

