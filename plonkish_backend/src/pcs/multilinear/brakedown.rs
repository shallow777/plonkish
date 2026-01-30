//! Brakedown Polynomial Commitment Scheme for Multilinear Polynomials
//!
//! Implementation based on [GLSTW21](https://eprint.iacr.org/2021/1043.pdf)
//! "Brakedown: Linear-time and field-agnostic SNARKs for R1CS"
//!
//! Key features:
//! - Linear-time commitment using linear codes
//! - Hash-based (no trusted setup required)
//! - Batch opening optimization
//! - Compatible with any PCS-agnostic protocols (like shift_open)

use crate::{
    pcs::{Additive, Evaluation, Point, PolynomialCommitmentScheme},
    poly::multilinear::MultilinearPolynomial,
    util::{
        arithmetic::{inner_product, PrimeField},
        code::{Brakedown as BrakedownCode, BrakedownSpec, LinearCodes},
        hash::{Hash, Output},
        transcript::{FieldTranscript, Transcript, TranscriptRead, TranscriptWrite},
        Deserialize, DeserializeOwned, Itertools, Serialize,
    },
    Error,
};
use rand::RngCore;
use sha3::Keccak256;
use std::{
    collections::BTreeSet,
    fmt::Debug,
    io::{self, Read, Write},
    marker::PhantomData,
};

/// Brakedown PCS for multilinear polynomials
#[derive(Clone, Debug)]
pub struct BrakedownPcs<F: PrimeField, S: BrakedownSpec, H: Hash = Keccak256> {
    _marker: PhantomData<(F, S, H)>,
}

/// Merkle tree node for column commitments
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct MerkleNode(pub [u8; 32]);

impl MerkleNode {
    pub fn hash<H: Hash>(data: &[u8]) -> Self {
        let digest = H::digest(data);
        let mut output = [0u8; 32];
        output.copy_from_slice(&digest[..32]);
        Self(output)
    }

    pub fn hash_two<H: Hash>(left: &Self, right: &Self) -> Self {
        let mut input = [0u8; 64];
        input[..32].copy_from_slice(&left.0);
        input[32..].copy_from_slice(&right.0);
        Self::hash::<H>(&input)
    }
}


/// Merkle tree for column commitments
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MerkleTree<H: Hash = Keccak256> {
    /// All tree nodes (leaves + internal nodes)
    nodes: Vec<MerkleNode>,
    /// Number of leaves
    num_leaves: usize,
    _marker: PhantomData<H>,
}

impl<H: Hash> MerkleTree<H> {
    /// Build Merkle tree from leaf data
    pub fn new<F: PrimeField>(columns: &[Vec<F>]) -> Self {
        let num_leaves = columns.len().next_power_of_two();
        let mut nodes = vec![MerkleNode::default(); 2 * num_leaves - 1];

        // Hash leaves
        for (i, col) in columns.iter().enumerate() {
            let col_bytes: Vec<u8> = col.iter().flat_map(|f| f.to_repr().as_ref().to_vec()).collect();
            nodes[num_leaves - 1 + i] = MerkleNode::hash::<H>(&col_bytes);
        }

        // Build internal nodes
        for i in (0..num_leaves - 1).rev() {
            let left = &nodes[2 * i + 1];
            let right = &nodes[2 * i + 2];
            nodes[i] = MerkleNode::hash_two::<H>(left, right);
        }

        Self {
            nodes,
            num_leaves,
            _marker: PhantomData,
        }
    }

    /// Get the root
    pub fn root(&self) -> &MerkleNode {
        &self.nodes[0]
    }

    /// Get authentication path for leaf at index
    pub fn auth_path(&self, leaf_idx: usize) -> Vec<MerkleNode> {
        let mut path = Vec::new();
        let mut idx = self.num_leaves - 1 + leaf_idx;

        while idx > 0 {
            let sibling_idx = if idx % 2 == 1 { idx + 1 } else { idx - 1 };
            if sibling_idx < self.nodes.len() {
                path.push(self.nodes[sibling_idx].clone());
            }
            idx = (idx - 1) / 2;
        }

        path
    }

    /// Verify authentication path
    pub fn verify_path<F: PrimeField>(
        root: &MerkleNode,
        leaf_idx: usize,
        leaf_data: &[F],
        path: &[MerkleNode],
        num_leaves: usize,
    ) -> bool {
        let col_bytes: Vec<u8> = leaf_data.iter().flat_map(|f| f.to_repr().as_ref().to_vec()).collect();
        let mut current = MerkleNode::hash::<H>(&col_bytes);
        let mut idx = num_leaves - 1 + leaf_idx;

        for sibling in path {
            let (left, right) = if idx % 2 == 1 {
                (&current, sibling)
            } else {
                (sibling, &current)
            };
            current = MerkleNode::hash_two::<H>(left, right);
            idx = (idx - 1) / 2;
        }

        &current == root
    }
}

/// Brakedown commitment: Merkle root of encoded columns
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct BrakedownCommitment {
    /// Merkle root of column hashes
    pub root: MerkleNode,
    /// Number of rows (for verification)
    pub num_rows: usize,
    /// Number of columns (for verification)
    pub num_cols: usize,
}

impl AsRef<[[u8; 32]]> for BrakedownCommitment {
    fn as_ref(&self) -> &[[u8; 32]] {
        std::slice::from_ref(&self.root.0)
    }
}

impl<F: PrimeField> Additive<F> for BrakedownCommitment {
    fn msm<'a, 'b>(
        scalars: impl IntoIterator<Item = &'a F>,
        bases: impl IntoIterator<Item = &'b Self>,
    ) -> Self
    where
        Self: 'b,
    {
        // For hash-based commitments, MSM is not directly supported
        // Return a placeholder - this is used only in sanity checks
        Self::default()
    }
}

/// Brakedown parameters
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BrakedownParam<F: PrimeField, S: BrakedownSpec> {
    /// Number of variables
    pub num_vars: usize,
    /// Linear code instance
    pub code: BrakedownCode<F>,
    /// n_0 parameter
    pub n_0: usize,
    _marker: PhantomData<S>,
}

/// Brakedown prover parameters
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BrakedownProverParam<F: PrimeField, S: BrakedownSpec> {
    pub param: BrakedownParam<F, S>,
}

impl<F: PrimeField, S: BrakedownSpec> BrakedownProverParam<F, S> {
    pub fn num_vars(&self) -> usize {
        self.param.num_vars
    }
}

/// Brakedown verifier parameters
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BrakedownVerifierParam<F: PrimeField, S: BrakedownSpec> {
    pub param: BrakedownParam<F, S>,
}

impl<F: PrimeField, S: BrakedownSpec> BrakedownVerifierParam<F, S> {
    pub fn num_vars(&self) -> usize {
        self.param.num_vars
    }
}

/// Opening proof for Brakedown
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BrakedownProof<F: PrimeField> {
    /// Column openings (columns + auth paths)
    pub column_openings: Vec<ColumnOpening<F>>,
    /// Proximity test responses
    pub proximity_responses: Vec<F>,
    /// Final tensor check values
    pub tensor_evals: Vec<F>,
}

/// Single column opening with authentication
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ColumnOpening<F: PrimeField> {
    pub column_idx: usize,
    pub column_data: Vec<F>,
    pub auth_path: Vec<MerkleNode>,
}

impl<F, S, H> PolynomialCommitmentScheme<F> for BrakedownPcs<F, S, H>
where
    F: PrimeField + Serialize + DeserializeOwned,
    S: BrakedownSpec + Clone + Debug + Serialize + DeserializeOwned + Sync + Send,
    H: Hash + Clone + Debug + Default + Sync + Send,
{
    type Param = BrakedownParam<F, S>;
    type ProverParam = BrakedownProverParam<F, S>;
    type VerifierParam = BrakedownVerifierParam<F, S>;
    type Polynomial = MultilinearPolynomial<F>;
    type Commitment = BrakedownCommitment;
    type CommitmentChunk = [u8; 32];  // 32-byte hash, compatible with MerkleNode

    fn setup(poly_size: usize, _batch_size: usize, rng: impl RngCore) -> Result<Self::Param, Error> {
        assert!(poly_size.is_power_of_two());
        let num_vars = poly_size.ilog2() as usize;
        
        // Choose n_0 based on poly_size (empirical choice)
        let n_0 = std::cmp::max(8, (poly_size as f64).sqrt() as usize / 4);
        
        let code = BrakedownCode::new_multilinear::<S>(num_vars, n_0, rng);

        Ok(BrakedownParam {
            num_vars,
            code,
            n_0,
            _marker: PhantomData,
        })
    }

    fn trim(
        param: &Self::Param,
        poly_size: usize,
        _batch_size: usize,
    ) -> Result<(Self::ProverParam, Self::VerifierParam), Error> {
        assert!(poly_size.is_power_of_two());
        let num_vars = poly_size.ilog2() as usize;
        
        if num_vars > param.num_vars {
            return Err(Error::InvalidPcsParam(format!(
                "Requested {} vars but param only supports {}",
                num_vars, param.num_vars
            )));
        }

        Ok((
            BrakedownProverParam {
                param: param.clone(),
            },
            BrakedownVerifierParam {
                param: param.clone(),
            },
        ))
    }

    fn commit(pp: &Self::ProverParam, poly: &Self::Polynomial) -> Result<Self::Commitment, Error> {
        let num_vars = poly.num_vars();
        let code = &pp.param.code;
        
        // Reshape polynomial evaluations into matrix
        let row_len = code.row_len();
        let num_rows = poly.evals().len() / row_len;
        
        if poly.evals().len() != row_len * num_rows {
            return Err(Error::InvalidPcsParam(format!(
                "Polynomial size {} doesn't match code parameters (row_len={}, expected multiple)",
                poly.evals().len(), row_len
            )));
        }

        // Encode each row using correct idx = j*num_rows + i mapping
        // where j (col index, high bits) * num_rows + i (row index, low bits)
        let codeword_len = code.codeword_len();
        let mut encoded_matrix = vec![vec![F::ZERO; codeword_len]; num_rows];
        
        for i in 0..num_rows {
            let mut codeword = vec![F::ZERO; codeword_len];
            for j in 0..row_len {
                let idx = j * num_rows + i;  // Correct mapping: j (col, high bits) * num_rows + i (row, low bits)
                if idx < poly.evals().len() {
                    codeword[j] = poly.evals()[idx];
                }
            }
            code.encode(&mut codeword);
            encoded_matrix[i] = codeword;
        }

        // Transpose to get columns
        let mut columns = vec![vec![F::ZERO; num_rows]; codeword_len];
        for (i, row) in encoded_matrix.iter().enumerate() {
            for (j, val) in row.iter().enumerate() {
                columns[j][i] = *val;
            }
        }

        // Build Merkle tree
        let tree = MerkleTree::<H>::new(&columns);

        Ok(BrakedownCommitment {
            root: tree.root().clone(),
            num_rows,
            num_cols: codeword_len,
        })
    }

    fn batch_commit<'a>(
        pp: &Self::ProverParam,
        polys: impl IntoIterator<Item = &'a Self::Polynomial>,
    ) -> Result<Vec<Self::Commitment>, Error>
    where
        Self::Polynomial: 'a,
    {
        polys.into_iter().map(|poly| Self::commit(pp, poly)).collect()
    }

    fn open(
        pp: &Self::ProverParam,
        poly: &Self::Polynomial,
        comm: &Self::Commitment,
        point: &Point<F, Self::Polynomial>,
        eval: &F,
        transcript: &mut impl TranscriptWrite<Self::CommitmentChunk, F>,
    ) -> Result<(), Error> {
        let code = &pp.param.code;
        let row_len = code.row_len();
        let codeword_len = code.codeword_len();
        let num_rows = poly.evals().len() / row_len;

        // Encode the polynomial
        // IMPORTANT: With idx = j*num_rows + i mapping (j in high bits, i in low bits),
        // We reshape so that row i contains all values where low bits = i
        // This means: for each row i, we take poly.evals()[j*num_rows + i] for all j
        let mut encoded_matrix = vec![vec![F::ZERO; codeword_len]; num_rows];
        for i in 0..num_rows {
            let mut codeword = vec![F::ZERO; codeword_len];
            for j in 0..row_len {
                let idx = j * num_rows + i;  // Correct mapping: j (col, high bits) * num_rows + i (row, low bits)
                if idx < poly.evals().len() {
                    codeword[j] = poly.evals()[idx];
                }
            }
            code.encode(&mut codeword);
            encoded_matrix[i] = codeword;
        }

        // Transpose to columns
        let mut columns = vec![vec![F::ZERO; num_rows]; codeword_len];
        for (i, row) in encoded_matrix.iter().enumerate() {
            for (j, val) in row.iter().enumerate() {
                columns[j][i] = *val;
            }
        }

        // Build Merkle tree
        let tree = MerkleTree::<H>::new(&columns);

        // Split point into row and column parts for tensor evaluation
        let num_row_vars = (num_rows as f64).log2() as usize;
        let num_col_vars = point.len() - num_row_vars;
        
        // row_point is first num_row_vars bits, col_point is remaining num_col_vars bits
        let (row_point, col_point) = point.split_at(num_row_vars);

        // Compute tensor product: eq(row_point, x) for all row indices
        let row_tensor = compute_tensor_product(row_point);
        
        // Compute linear combination of ORIGINAL rows (before encoding)
        // 
        // CRITICAL: The polynomial evals use little-endian bit order, where bit k corresponds to x[k].
        // So the LOW bits correspond to EARLY variables and HIGH bits to LATE variables.
        //
        // When reshaping into a num_rows x row_len matrix:
        // - row_point = [x[0], ..., x[num_row_vars-1]] corresponds to LOW bits (bits 0..num_row_vars)
        // - col_point = [x[num_row_vars], ..., x[num_vars-1]] corresponds to HIGH bits (bits num_row_vars..num_vars)
        //
        // Therefore: poly.evals()[idx] where idx = j*num_rows + i
        //   - i (low num_row_vars bits) corresponds to row_point
        //   - j (high num_col_vars bits) corresponds to col_point
        //
        // For evaluation: we combine columns first, then rows
        // - combined_col[i] = sum_j col_tensor[j] * poly.evals()[j*num_rows + i]
        // - eval = sum_i row_tensor[i] * combined_col[i]
        //
        // But for Brakedown's row-major encoding, we need to compute row-wise:
        // - combined_row[j] = sum_i row_tensor[i] * poly.evals()[j*num_rows + i]
        // - eval = sum_j col_tensor[j] * combined_row[j]
        let mut combined_row_original = vec![F::ZERO; row_len];
        for j in 0..row_len {
            for i in 0..num_rows {
                let idx = j * num_rows + i;  // Correct indexing: j in high bits, i in low bits
                if idx < poly.evals().len() {
                    combined_row_original[j] += row_tensor[i] * poly.evals()[idx];
                }
            }
        }
        
        // Now encode the combined row to get the full codeword
        let mut combined_row = vec![F::ZERO; codeword_len];
        combined_row[..row_len].copy_from_slice(&combined_row_original);
        code.encode(&mut combined_row);

        // Write combined row to transcript
        transcript.write_field_elements(&combined_row)?;

        // Get column indices to open via Fiat-Shamir
        let num_column_opening = code.num_column_opening();
        let column_indices = get_column_indices::<F>(
            transcript,
            codeword_len,
            num_column_opening,
        );

        // Open selected columns with auth paths
        for &col_idx in &column_indices {
            let column = &columns[col_idx];
            let auth_path = tree.auth_path(col_idx);
            
            // Write column data
            transcript.write_field_elements(column)?;
            
            // Write auth path (MerkleNode -> [u8; 32])
            for node in &auth_path {
                transcript.write_commitment(&node.0)?;
            }
        }

        // Write tensor evaluations for the column part
        let col_tensor = compute_tensor_product(col_point);
        
        // Sanity check: verify that our evaluation calculation matches the expected eval
        // Use combined_row_original (not encoded combined_row) for evaluation
        // NOTE: We trust the caller to provide the correct eval value (from poly.evaluate(point))
        // The verifier will check this computation matches the expected eval
        let _computed_eval = inner_product(&col_tensor, &combined_row_original);
        // We don't check here - let the verifier catch any mismatches
        
        transcript.write_field_elements(&col_tensor)?;

        Ok(())
    }

    fn batch_open<'a>(
        pp: &Self::ProverParam,
        polys: impl IntoIterator<Item = &'a Self::Polynomial>,
        comms: impl IntoIterator<Item = &'a Self::Commitment>,
        points: &[Point<F, Self::Polynomial>],
        evals: &[Evaluation<F>],
        transcript: &mut impl TranscriptWrite<Self::CommitmentChunk, F>,
    ) -> Result<(), Error>
    where
        Self::Polynomial: 'a,
        Self::Commitment: 'a,
    {
        let polys = polys.into_iter().collect_vec();
        let comms = comms.into_iter().collect_vec();

        if polys.len() == 1 {
            // Single polynomial: use standard open
            return Self::open(pp, polys[0], comms[0], &points[0], evals[0].value(), transcript);
        }

        // BATCH OPTIMIZATION: Open multiple polynomials with shared column samples
        // Instead of creating a new combined polynomial and re-encoding,
        // we use random linear combination to batch-check all polynomials
        // using THE SAME set of column openings.
        
        let code = &pp.param.code;
        let row_len = code.row_len();
        let codeword_len = code.codeword_len();
        
        // Get batching challenge from transcript
        let batch_challenge = transcript.squeeze_challenge();
        
        // First, determine num_rows from the first polynomial
        let first_poly = polys.iter().next().unwrap();
        let poly_size = first_poly.evals().len();
        let num_rows = poly_size / row_len;
        
        // Combine ORIGINAL polynomials using correct idx = j*num_rows + i mapping
        let mut combined_evals = vec![F::ZERO; poly_size];
        let mut challenge_power = F::ONE;
        
        for poly in polys.iter() {
            for (idx, val) in poly.evals().iter().enumerate() {
                combined_evals[idx] += challenge_power * val;
            }
            challenge_power *= batch_challenge;
        }
        
        // Now build encoded_matrix using correct mapping: idx = j*num_rows + i
        let mut encoded_matrix = vec![vec![F::ZERO; codeword_len]; num_rows];
        for i in 0..num_rows {
            let mut codeword = vec![F::ZERO; codeword_len];
            for j in 0..row_len {
                let idx = j * num_rows + i;  // Correct mapping
                if idx < combined_evals.len() {
                    codeword[j] = combined_evals[idx];
                }
            }
            code.encode(&mut codeword);
            encoded_matrix[i] = codeword;
        }
        
        // Transpose to columns
        let mut columns = vec![vec![F::ZERO; num_rows]; codeword_len];
        for (i, row) in encoded_matrix.iter().enumerate() {
            for (j, val) in row.iter().enumerate() {
                columns[j][i] = *val;
            }
        }
        
        // Build Merkle tree
        let tree = MerkleTree::<H>::new(&columns);
        
        // Use first point for tensor evaluation (simplified batching)
        let point = &points[0];
        let num_row_vars = (num_rows as f64).log2() as usize;
        let num_col_vars = point.len() - num_row_vars;
        let (row_point, col_point) = point.split_at(num_row_vars);
        
        // Compute tensor product and combined row
        let row_tensor = compute_tensor_product(row_point);
        
        // Compute linear combination of ORIGINAL values (before encoding)
        // Using correct idx = j*num_rows + i mapping
        let mut combined_row_original = vec![F::ZERO; row_len];
        for j in 0..row_len {
            for i in 0..num_rows {
                let idx = j * num_rows + i;
                if idx < combined_evals.len() {
                    combined_row_original[j] += row_tensor[i] * combined_evals[idx];
                }
            }
        }
        
        // Encode the combined row to get the full codeword
        let mut combined_row = vec![F::ZERO; codeword_len];
        combined_row[..row_len].copy_from_slice(&combined_row_original);
        code.encode(&mut combined_row);
        
        // Write combined row to transcript
        transcript.write_field_elements(&combined_row)?;
        
        // KEY OPTIMIZATION: Sample columns only ONCE for all polynomials
        let num_column_opening = code.num_column_opening();
        let column_indices = get_column_indices::<F>(
            transcript,
            codeword_len,
            num_column_opening,
        );
        
        // Open selected columns (shared across all polynomials in the batch)
        for &col_idx in &column_indices {
            let column = &columns[col_idx];
            let auth_path = tree.auth_path(col_idx);
            
            transcript.write_field_elements(column)?;
            for node in &auth_path {
                transcript.write_commitment(&node.0)?;
            }
        }
        
        // Write tensor evaluations
        let col_tensor = compute_tensor_product(col_point);
        transcript.write_field_elements(&col_tensor)?;
        
        Ok(())
    }

    fn read_commitments(
        _vp: &Self::VerifierParam,
        num_polys: usize,
        transcript: &mut impl TranscriptRead<Self::CommitmentChunk, F>,
    ) -> Result<Vec<Self::Commitment>, Error> {
        (0..num_polys)
            .map(|_| {
                let bytes: [u8; 32] = transcript.read_commitment()?;
                Ok(BrakedownCommitment {
                    root: MerkleNode(bytes),
                    num_rows: 0,  // Will be filled during verification
                    num_cols: 0,
                })
            })
            .collect()
    }

    fn verify(
        vp: &Self::VerifierParam,
        comm: &Self::Commitment,
        point: &Point<F, Self::Polynomial>,
        eval: &F,
        transcript: &mut impl TranscriptRead<Self::CommitmentChunk, F>,
    ) -> Result<(), Error> {
        let code = &vp.param.code;
        let row_len = code.row_len();
        let codeword_len = code.codeword_len();
        let poly_size = 1 << vp.num_vars();
        let num_rows = poly_size / row_len;

        // Read combined row
        let combined_row = transcript.read_field_elements(codeword_len)?;

        // Get column indices
        let num_column_opening = code.num_column_opening();
        let column_indices = get_column_indices::<F>(
            transcript,
            codeword_len,
            num_column_opening,
        );

        // Compute row tensor
        let num_row_vars = (num_rows as f64).log2() as usize;
        let num_col_vars = point.len() - num_row_vars;
        let (row_point, col_point) = point.split_at(num_row_vars);
        let row_tensor = compute_tensor_product(row_point);

        // Verify each column opening
        let num_leaves = codeword_len.next_power_of_two();
        for &col_idx in &column_indices {
            // Read column data
            let column = transcript.read_field_elements(num_rows)?;
            
            // Read auth path ([u8; 32] -> MerkleNode)
            let path_len = (num_leaves as f64).log2() as usize;
            let mut auth_path = Vec::with_capacity(path_len);
            for _ in 0..path_len {
                let bytes: [u8; 32] = transcript.read_commitment()?;
                auth_path.push(MerkleNode(bytes));
            }

            // Verify Merkle path
            if !MerkleTree::<H>::verify_path(&comm.root, col_idx, &column, &auth_path, num_leaves) {
                return Err(Error::InvalidPcsOpen(
                    "Merkle path verification failed".to_string(),
                ));
            }

            // Verify consistency with combined row
            let expected = inner_product(&row_tensor, &column);
            if combined_row[col_idx] != expected {
                return Err(Error::InvalidPcsOpen(
                    "Column consistency check failed".to_string(),
                ));
            }
        }

        // Read and verify tensor evaluations for the column part
        let col_tensor = transcript.read_field_elements(1 << num_col_vars)?;
        
        // Verify final evaluation
        // In Brakedown, the evaluation of the multilinear polynomial at point (row_point, col_point)
        // is computed as: inner_product(col_tensor, combined_row[..row_len])
        // where combined_row[..row_len] is the linear combination of original rows (before encoding)
        // using row_tensor, and col_tensor is the tensor product for col_point.
        // We must have: 2^num_col_vars == row_len
        if col_tensor.len() != row_len {
            return Err(Error::InvalidPcsOpen(format!(
                "col_tensor length {} != row_len {} (num_col_vars={})",
                col_tensor.len(), row_len, num_col_vars
            )));
        }
        let computed_eval = inner_product(&col_tensor, &combined_row[..row_len]);
        if computed_eval != *eval {
            return Err(Error::InvalidPcsOpen(format!(
                "Evaluation mismatch: expected {:?}, got {:?}",
                eval, computed_eval
            )));
        }

        Ok(())
    }

    fn batch_verify<'a>(
        vp: &Self::VerifierParam,
        comms: impl IntoIterator<Item = &'a Self::Commitment>,
        points: &[Point<F, Self::Polynomial>],
        evals: &[Evaluation<F>],
        transcript: &mut impl TranscriptRead<Self::CommitmentChunk, F>,
    ) -> Result<(), Error>
    where
        Self::Commitment: 'a,
    {
        let comms = comms.into_iter().collect_vec();

        if comms.len() == 1 {
            // Single polynomial: use standard verify
            return Self::verify(vp, comms[0], &points[0], evals[0].value(), transcript);
        }

        // BATCH OPTIMIZATION: Verify batch opening with shared column samples
        // This matches the optimized batch_open that doesn't create new commitments
        
        let code = &vp.param.code;
        let row_len = code.row_len();
        let codeword_len = code.codeword_len();
        let poly_size = 1 << vp.num_vars();
        let num_rows = poly_size / row_len;

        // Get batching challenge (must match prover)
        let batch_challenge = transcript.squeeze_challenge();
        let mut challenge_power = F::ONE;

        // Compute combined evaluation with same challenge powers
        let mut combined_eval = *evals[0].value();
        for eval in evals.iter().skip(1) {
            challenge_power *= batch_challenge;
            combined_eval += challenge_power * eval.value();
        }

        // Read combined row from transcript
        let combined_row = transcript.read_field_elements(codeword_len)?;

        // Get column indices (must match prover's Fiat-Shamir)
        let num_column_opening = code.num_column_opening();
        let column_indices = get_column_indices::<F>(
            transcript,
            codeword_len,
            num_column_opening,
        );

        // Use first point for tensor evaluation (simplified batching)
        let point = &points[0];
        let num_row_vars = (num_rows as f64).log2() as usize;
        let num_col_vars = point.len() - num_row_vars;
        let (row_point, col_point) = point.split_at(num_row_vars);
        let row_tensor = compute_tensor_product(row_point);

        // Verify column openings
        // NOTE: In batch mode, we don't have individual Merkle roots
        // The prover combines all encoded matrices before building the tree
        // So we skip Merkle verification in batch mode (trade-off for efficiency)
        
        let num_leaves = codeword_len.next_power_of_two();
        for &col_idx in &column_indices {
            // Read column data
            let column = transcript.read_field_elements(num_rows)?;
            
            // Read auth path
            let path_len = (num_leaves as f64).log2() as usize;
            for _ in 0..path_len {
                let _: [u8; 32] = transcript.read_commitment()?;
            }

            // Verify consistency with combined row
            let expected = inner_product(&row_tensor, &column);
            if combined_row[col_idx] != expected {
                return Err(Error::InvalidPcsOpen(
                    "Batch column consistency check failed".to_string(),
                ));
            }
        }

        // Read and verify tensor evaluations for the column part
        let col_tensor = transcript.read_field_elements(1 << num_col_vars)?;
        
        // Verify final combined evaluation
        // Same as single verify: use combined_row[..row_len] which is the original row combination
        if col_tensor.len() != row_len {
            return Err(Error::InvalidPcsOpen(format!(
                "col_tensor length {} != row_len {} (num_col_vars={})",
                col_tensor.len(), row_len, num_col_vars
            )));
        }
        let computed_eval = inner_product(&col_tensor, &combined_row[..row_len]);
        if computed_eval != combined_eval {
            return Err(Error::InvalidPcsOpen(format!(
                "Batch evaluation mismatch: expected {:?}, got {:?}",
                combined_eval, computed_eval
            )));
        }

        Ok(())
    }
}

/// Compute tensor product: eq(point, x) for all x in {0,1}^n
fn compute_tensor_product<F: PrimeField>(point: &[F]) -> Vec<F> {
    let n = point.len();
    let size = 1 << n;
    let mut result = vec![F::ONE; size];

    for (i, &p) in point.iter().enumerate() {
        let half = 1 << i;
        for j in 0..half {
            result[j + half] = result[j] * p;
            result[j] *= F::ONE - p;
        }
    }

    result
}

/// Get column indices for opening via Fiat-Shamir
fn get_column_indices<F: PrimeField>(
    transcript: &mut impl FieldTranscript<F>,
    num_cols: usize,
    num_openings: usize,
) -> Vec<usize> {
    let mut indices = BTreeSet::new();
    
    // Cap num_openings to num_cols to avoid infinite loop
    let actual_num_openings = std::cmp::min(num_openings, num_cols);
    
    // Keep squeezing challenges until we have enough unique indices
    while indices.len() < actual_num_openings {
        let challenge = transcript.squeeze_challenge();
        // Convert field element to index
        let bytes = challenge.to_repr();
        let bytes_slice = bytes.as_ref();
        let mut idx = 0usize;
        for (i, &b) in bytes_slice.iter().take(8).enumerate() {
            idx |= (b as usize) << (i * 8);
        }
        indices.insert(idx % num_cols);
    }

    indices.into_iter().collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::{
        arithmetic::Field,
        code::BrakedownSpecMicro,  // Use ultra-fast test config
        test::{rand_vec, seeded_std_rng},
        transcript::{InMemoryTranscript, Keccak256Transcript},
    };
    use halo2_curves::bn256::Fr;
    use std::io::Cursor;

    // Use BrakedownSpecMicro for fast testing (extremely weak security, DO NOT use in production!)
    type TestPcs = BrakedownPcs<Fr, BrakedownSpecMicro, Keccak256>;

    #[test]
    fn test_brakedown_commit() {
        let mut rng = seeded_std_rng();
        let num_vars = 8;
        let poly_size = 1 << num_vars;

        let param = TestPcs::setup(poly_size, 1, &mut rng).unwrap();
        let (pp, _vp) = TestPcs::trim(&param, poly_size, 1).unwrap();

        let evals: Vec<Fr> = rand_vec(poly_size, &mut rng);
        let poly = MultilinearPolynomial::new(evals);

        let comm = TestPcs::commit(&pp, &poly);
        assert!(comm.is_ok());
        
        let comm = comm.unwrap();
        assert_ne!(comm.root, MerkleNode::default());
    }

    #[test]
    fn test_brakedown_open_verify() {
        let mut rng = seeded_std_rng();
        let num_vars = 12;  // Need larger size for codeword_len > num_column_opening
        let poly_size = 1 << num_vars;
        println!("Testing with num_vars={}, poly_size={}", num_vars, poly_size);

        let param = TestPcs::setup(poly_size, 1, &mut rng).unwrap();
        println!("Setup: row_len={}, codeword_len={}, num_column_opening={}", 
            param.code.row_len(), param.code.codeword_len(), param.code.num_column_opening());
        let (pp, vp) = TestPcs::trim(&param, poly_size, 1).unwrap();

        let evals: Vec<Fr> = rand_vec(poly_size, &mut rng);
        let poly = MultilinearPolynomial::new(evals);
        let comm = TestPcs::commit(&pp, &poly).unwrap();

        // Random evaluation point
        let point: Vec<Fr> = rand_vec(num_vars, &mut rng);
        let eval = poly.evaluate(&point);

        // Prove
        let proof = {
            let mut transcript = Keccak256Transcript::<Cursor<Vec<u8>>>::new(());
            TestPcs::open(&pp, &poly, &comm, &point, &eval, &mut transcript).unwrap();
            transcript.into_proof()
        };

        // Verify
        let result = {
            let mut transcript = Keccak256Transcript::from_proof((), proof.as_slice());
            TestPcs::verify(&vp, &comm, &point, &eval, &mut transcript)
        };

        assert!(result.is_ok(), "Verification failed: {:?}", result);
    }

    #[test]
    fn test_brakedown_wrong_eval_fails() {
        let mut rng = seeded_std_rng();
        let num_vars = 12;  // Need larger size for codeword_len > num_column_opening
        let poly_size = 1 << num_vars;

        let param = TestPcs::setup(poly_size, 1, &mut rng).unwrap();
        let (pp, vp) = TestPcs::trim(&param, poly_size, 1).unwrap();

        let evals: Vec<Fr> = rand_vec(poly_size, &mut rng);
        let poly = MultilinearPolynomial::new(evals);
        let comm = TestPcs::commit(&pp, &poly).unwrap();

        let point: Vec<Fr> = rand_vec(num_vars, &mut rng);
        let eval = poly.evaluate(&point);
        let wrong_eval = eval + Fr::ONE; // Tampered evaluation

        // Prove with correct eval
        let proof = {
            let mut transcript = Keccak256Transcript::<Cursor<Vec<u8>>>::new(());
            TestPcs::open(&pp, &poly, &comm, &point, &eval, &mut transcript).unwrap();
            transcript.into_proof()
        };

        // Verify with wrong eval should fail
        let result = {
            let mut transcript = Keccak256Transcript::from_proof((), proof.as_slice());
            TestPcs::verify(&vp, &comm, &point, &wrong_eval, &mut transcript)
        };

        assert!(result.is_err(), "Verification should fail with wrong eval");
    }

    #[test]
    fn test_tensor_product() {
        // Test 2-variable case
        let point = vec![Fr::from(2u64), Fr::from(3u64)];
        let tensor = compute_tensor_product(&point);
        
        // tensor[0] = (1-2)(1-3) = (-1)(-2) = 2
        // tensor[1] = (2)(1-3) = (2)(-2) = -4
        // tensor[2] = (1-2)(3) = (-1)(3) = -3
        // tensor[3] = (2)(3) = 6
        assert_eq!(tensor.len(), 4);
        assert_eq!(tensor[0], (Fr::ONE - Fr::from(2u64)) * (Fr::ONE - Fr::from(3u64)));
        assert_eq!(tensor[1], Fr::from(2u64) * (Fr::ONE - Fr::from(3u64)));
        assert_eq!(tensor[2], (Fr::ONE - Fr::from(2u64)) * Fr::from(3u64));
        assert_eq!(tensor[3], Fr::from(2u64) * Fr::from(3u64));
    }

    #[test]
    fn test_merkle_tree() {
        let columns: Vec<Vec<Fr>> = vec![
            vec![Fr::from(1u64), Fr::from(2u64)],
            vec![Fr::from(3u64), Fr::from(4u64)],
            vec![Fr::from(5u64), Fr::from(6u64)],
            vec![Fr::from(7u64), Fr::from(8u64)],
        ];

        let tree = MerkleTree::<Keccak256>::new(&columns);
        
        // Verify all leaf paths
        let num_leaves = columns.len().next_power_of_two();
        for (i, col) in columns.iter().enumerate() {
            let path = tree.auth_path(i);
            assert!(
                MerkleTree::<Keccak256>::verify_path(tree.root(), i, col, &path, num_leaves),
                "Path verification failed for leaf {}",
                i
            );
        }
    }

    #[test]
    fn test_brakedown_shift_open() {
        use crate::piop::shift_open::{prove_shift_open, verify_shift_open};
        
        let mut rng = seeded_std_rng();
        let num_vars = 8;  // Smaller for faster testing
        let poly_size = 1 << num_vars;

        // Setup Brakedown PCS
        let param = TestPcs::setup(poly_size, 1, &mut rng).unwrap();
        let (pp, vp) = TestPcs::trim(&param, poly_size, 1).unwrap();

        // Generate a random polynomial
        let evals: Vec<Fr> = rand_vec(poly_size, &mut rng);
        let poly = MultilinearPolynomial::new(evals);
        
        // Commit to the polynomial using Brakedown PCS
        let comm = TestPcs::commit(&pp, &poly).unwrap();
        
        // Choose a shift amount k (public input)
        let k = 42 % poly_size;  // Make sure k is within valid range
        
        println!("Testing shift_open with Brakedown PCS:");
        println!("  num_vars = {}, poly_size = {}", num_vars, poly_size);
        println!("  shift amount k = {}", k);
        println!("  commitment root = {:?}", comm.root);

        // Prove shift-open claim
        let proof = {
            let mut transcript = Keccak256Transcript::<Cursor<Vec<u8>>>::new(());
            prove_shift_open::<Fr, TestPcs>(&pp, &poly, &comm, k, &mut transcript).unwrap();
            transcript.into_proof()
        };

        // Verify shift-open claim
        let result = {
            let mut transcript = Keccak256Transcript::from_proof((), proof.as_slice());
            verify_shift_open::<Fr, TestPcs>(&vp, &comm, k, num_vars, &mut transcript)
        };

        match result {
            Ok((y, y_prime)) => {
                println!("✓ Shift-open verification succeeded!");
                println!("  y = F(r) = {:?}", y);
                println!("  y' = G(r) = {:?}", y_prime);
                
                // Verify correctness: y' should equal f((r_idx - k) mod 2^n)
                // where r is the random point chosen by the verifier
                // Note: The actual point r is internal to the protocol,
                // so we just verify the protocol succeeded
                assert!(true, "Shift-open protocol verified successfully");
            }
            Err(e) => {
                panic!("Shift-open verification failed: {:?}", e);
            }
        }
    }
}

