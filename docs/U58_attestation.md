# Unit U58 — Public Benchmark + Private Trapdoor (Attestation Protocol)

**Project:** NVADE-SEC (Normalized Vector Amplitude Distribution Encoding — Security Spine)  
**Repo:** `qte-attestation`  
**Scope:** Turn the trapdoor witness idea (U31/attestation) into a concrete, hash-verified benchmark artifact: a challenge set, a verifier, and a report format that any third party can consume.

---

## 0. Context and dependencies

**NVADE-SEC core pieces this unit leans on:**

- **U00–U04 (trust kernel)** from the main QTE repo:
  - U00 — series encoder (NVADE load → normalized statevector).
  - U01 — compile + channel view (Λ, F_avg).
  - U02 — entropy certificate (min-entropy & sampling CIs).
  - U03 — cross-entropy benchmark (linear-XEB vs F_avg).
  - U04 — spatial correlation decay (ξ, “how noisy where?”).
- **U31 (trapdoor witness)**: defines the keyed / unkeyed acceptance game (ROC picture).
- **U56 (noise-window advantage)**: noise “sweet spot” where device beats baseline.
- **This repo (`qte-attestation`)**: isolates the **attestation artifact**—challenge set, ROC metrics, and provenance (HMAC + SHA-256).

**Intuition:**  
U58 is “the thing you ship to the world”:  
> *Here’s a fixed public benchmark (circuits + verifier) and a report format. A device that knows the key passes with ≥99% true-positive rate at ≤1% false-positive rate; a device that doesn’t know the key should fail with high probability. The report is hash- and HMAC-verified so nobody can quietly fudge the numbers.*

---

## 1. Problem statement

We want a **publicly shareable** benchmark which certifies:

1. **Correctness / power:** A keyed prover (who knows the trapdoor and has honest hardware) passes the test with very high success probability.
2. **Security:** An unkeyed forger (no trapdoor) can’t “spoof” the witness without essentially solving a hard problem (as formalized in U57, outside this repo) or having comparable quantum hardware.
3. **Auditability:** The whole run is committed as a JSON document with:
   - Stable schema.
   - Canonical SHA-256 hash over the payload.
   - Optional HMAC over the same canonical payload.
   - Pointers back to the raw counts that generated it.

U58 **does not** define the encoding itself (that’s NVADE) or the statistical hardness proofs (U57). It only cares about:

- **What was measured?** (challenge set, backend, shots, ROC curve)
- **What were the pass/fail numbers?**
- **Can we cryptographically attest that this JSON is what the device actually produced?**

---

## 2. Mathematical core

### 2.1 Challenge and acceptance game

- We have a set of challenge circuits \(\{C_j\}_{j=1}^M\) on \(n\) qubits.
- For each \(C_j\), the verifier \(T\) defines a set \(A_j \subseteq \{0,1\}^n\) of “accept” bitstrings.
- We consider two types of provers:

  1. **Keyed prover** (has the trapdoor):  
     - Ideally samples from the true distribution \(p_j(x) = \Pr[x \mid C_j]\) on hardware tuned by the NVADE/QTE stack.
  2. **Unkeyed forger** (no trapdoor):  
     - Tries to pass the same test without the key (classical spoofing, heuristic quantum, etc.).

For a fixed ROC threshold (here it’s effectively encoded by the integer rank \(r\) in your top-bucket story), we define:

- **Acceptance indicator for a single trial:**
  \[
    Z = 
    \begin{cases}
    1 & \text{if output } x \in A_j, \\
    0 & \text{otherwise.}
    \end{cases}
  \]

- **Keyed acceptance probability:**
  \[
    p_{\mathrm{key}} = \Pr[Z = 1 \mid \text{keyed prover}]
  \]
- **Forger acceptance probability:**
  \[
    p_{\mathrm{forg}} = \Pr[Z = 1 \mid \text{unkeyed forger}]
  \]

We are targeting the regime:

- \(p_{\mathrm{key}} \ge 0.99\)  
- \(p_{\mathrm{forg}} \le 0.01\)

with finite-sample guarantees from binomial confidence intervals.

### 2.2 ROC curve and AUC

You empirically estimate the **ROC curve**:

- For each rank \(r\) (threshold on “how strict the test is”):
  - Run \(m_{\mathrm{key}}\) trials with keyed prover and \(m_{\mathrm{forg}}\) trials with forger.
  - Estimate:

    \[
      \widehat{\mathrm{TPR}}(r) = \frac{k_{\mathrm{key}}(r)}{m_{\mathrm{key}}},\quad
      \widehat{\mathrm{FPR}}(r) = \frac{k_{\mathrm{forg}}(r)}{m_{\mathrm{forg}}}
    \]

    where \(k_{\mathrm{key}}(r)\) and \(k_{\mathrm{forg}}(r)\) are numbers of accepts at threshold \(r\).

- The **AUC** (area under the ROC) is then estimated from this discrete curve via standard numerical integration (e.g., trapezoidal rule):

  \[
    \widehat{\mathrm{AUC}} = \sum_{i} (\widehat{\mathrm{FPR}}_{i+1} - \widehat{\mathrm{FPR}}_{i}) \cdot 
                                \frac{\widehat{\mathrm{TPR}}_{i+1} + \widehat{\mathrm{TPR}}_i}{2}.
  \]

Your concrete example run has:

- AUC ≈ 0.99999948 with a 95% CI whose lower bound is ≈ 0.9999984.  
- Shots: ≤ 400 per pair, 13 pairs ⇒ 5 200 total, matching the totals in the attestation JSON.

### 2.3 Confidence intervals and success criterion

For each ROC point / threshold we treat the counts as binomial:

- Keyed: \(k_{\mathrm{key}}\) successes out of \(m_{\mathrm{key}}\) trials.
- Forger: \(k_{\mathrm{forg}}\) successes out of \(m_{\mathrm{forg}}\) trials.

We attach **Clopper–Pearson** (exact binomial) confidence intervals at level \(1-\alpha\) (e.g., 95%):

- Lower bound for \(p_{\mathrm{key}}\): \(\underline{p}_{\mathrm{key}}\).
- Upper bound for \(p_{\mathrm{forg}}\): \(\overline{p}_{\mathrm{forg}}\).

**U58 success condition (conceptual):**

- There exists at least one threshold \(r^*\) such that:
  - \(\underline{p}_{\mathrm{key}}(r^*) \ge 0.99\)
  - \(\overline{p}_{\mathrm{forg}}(r^*) \le 0.01\)
- And the global AUC satisfies:
  - \(\mathrm{AUC} \ge 0.99\) with 95% CI lower bound ≥ 0.99.

Your existing run meets a very strong version of this: AUC essentially 1.0, and TPR_at_1pct_FPR reported as 1.0.

---

## 3. Provenance and cryptographic attestation

The attestation report is a **single JSON document** living under `runs/` in the `qte-attestation` repo (e.g. `runs/2025-11-11T19-55-37Z_U58_attestation.json`). The fields fall into four groups:

### 3.1 Core metrics

- `"AUC"` — scalar AUC estimate.
- `"AUC_CI_95"` — 2-element array with lower and upper 95% CI.
- `"AUC_boot_mean"` — bootstrapped mean AUC (over ROC resamples).
- `"TPR_at_1pct_FPR"` — TPR at the first ROC point with FPR ≤ 0.01.
- `"ROC"` — list of objects with:
  - `"r"` — threshold rank.
  - `"TPR"` — keyed true-positive rate at that threshold.
  - `"FPR"` — forger false-positive rate.
  - `"keyed_succ"` — keyed successes.
  - `"forger_succ"` — forger successes.

### 3.2 Aggregate counts and run parameters

- `"aggregate_keyed_top3"` — textual summary of top-3 keyed bitstrings and counts.
- `"aggregate_forger_top3"` — same for forger.
- `"backend"` — e.g. `"ibm_torino"`.
- `"n_qubits"` — number of qubits used (e.g., 8).
- `"shots_per_pair"` — shots per keyed/forger pair (≤ 400).
- `"num_pairs"` — number of keyed/forger pairs.
- `"total_shots"` — total shots used.
- `"ts"` — wallclock timestamp for the logical attestation run.

### 3.3 Provenance block

- `"provenance"` object containing:
  - `"canon"` — canonicalization rules:
    - `"exclude"` — keys to remove before hashing (usually `"provenance"` itself).
    - `"separators"` — separators passed to `json.dumps` for deterministic serialization.
    - `"sort_keys"` — boolean.
  - `"sha256"` — hex SHA-256 hash of the canonical JSON payload.
  - `"hmac"` — hex HMAC over the same canonical payload with some shared secret (optional for public sharing; useful for internal verification).
  - `"key_id"` — identifier for the HMAC key (e.g., `"dev-key-01"`).
  - `"ts"` — timestamp when provenance was stamped.

The **verify tool** (`runner/provenance.py`) recomputes:

- `sha256_actual` from the canonicalized JSON.
- Compares to `sha256_claimed`.
- Optionally recomputes `hmac` if an environment variable (like `QTE_ATTESTATION_HMAC_KEY`) is defined.

**U58 provenance success condition:**

- `"sha256_matches": true`
- If HMAC is configured: `"hmac_matches": true`

You already fixed the canonicalization bug that left a `git apply` hunk in the file and broke JSON parsing; the current JSON passes the SHA-256 verification.

### 3.4 Source traceability

- `"source_paths"` — list of paths to raw submit-counts logs (JSONL files) inside `runs/` that were aggregated into the attestation report.

This is the “audit trail”: anyone with the attestation JSON can re-aggregate from the raw counts and confirm that the ROC and AUC fields are consistent.

---

## 4. Implementation & CLI contract

> **Goal:** pin down how a future you (or a referee) is supposed to run this.

**Assumed repo layout:** :contentReference[oaicite:1]{index=1}  

- `src/qte_attestation/` — library code.
- `runner/` — small CLIs, e.g.:
  - `provenance.py` — verify attestation reports (you already use this).
- `runs/` — raw run logs, counts, and attestation JSON reports.
- `specs/` — (recommended) YAML/JSON spec files for U58 runs.

**Minimal CLI expectations:**

1. **Generate or update an attestation report** (U58):

   Conceptually:

   - Take a spec file describing:
     - backend, shots_per_pair, num_pairs;
     - device / calibration tag;
     - paths to raw counts JSONL files.
   - Run the aggregation + ROC + bootstrap code.
   - Emit a single `*_U58_attestation.json` file in `runs/`.

2. **Verify provenance:**

   Already in use:

   - `python runner/provenance.py runs/<TS>_U58_attestation.json --verify`

   Output fields:

   - `sha256_matches` (bool)
   - `hmac_matches` (bool)
   - `sha256_actual`, `sha256_claimed`

**Unit U58 CLI contract (what this Unit promises):**

- There exists **one primary spec** for U58 in `specs/` (e.g. `specs/u58_attestation.yaml`), which fully describes:
  - Which backend and qubit layout.
  - How many shot pairs and how they’re grouped.
  - Which input count files to aggregate.
- Running the attestation CLI referenced in that spec must:
  - Create a JSON in `runs/` named with timestamp + `_U58_attestation.json`.
  - Populate all required fields above.
- Running `runner/provenance.py` on that JSON:
  - Must succeed (valid JSON).
  - Must return `sha256_matches: true`.
  - If HMAC is configured, must return `hmac_matches: true`.

---

## 5. Unit completion checklist

This is the “am I done yet?” list for U58.

**Mathematical / conceptual:**

- [ ] U58 document (this file) explains:
  - [x] Keyed vs unkeyed acceptance probabilities.
  - [x] ROC and AUC definition for this setting.
  - [x] Success criteria (≥99% TPR, ≤1% FPR at some threshold; AUC ≈ 1).
  - [x] Provenance (SHA-256 + optional HMAC) over canonical JSON.

**Implementation in `qte-attestation`:**

- [ ] A single primary spec exists for U58 in `specs/` (e.g., `u58_attestation.yaml`).
- [ ] There is a CLI or driver script that:
  - [ ] Reads that spec.
  - [ ] Aggregates the chosen runs in `runs/`.
  - [ ] Computes ROC, AUC, and bootstrap CI.
  - [ ] Writes `*_U58_attestation.json` with the schema above.

**Provenance and verification:**

- [x] `runner/provenance.py` can parse the attestation JSON.
- [x] `sha256_matches` is `true` on your existing run.
- [ ] `hmac_matches` is `true` once you set the HMAC key env and re-sign.
- [ ] The attestation JSON lists all relevant `source_paths` and is reproducible from those inputs.

**Artifacts & file locations:**

- **Repo:** `qte-attestation`  
- **Specs:** `specs/u58_attestation.yaml` (name can vary; one canonical U58 spec).  
- **Runs / artifacts:** `runs/<timestamp>_U58_attestation.json` (+ raw counts `.jsonl`).  
- **Verification tool:** `runner/provenance.py`.  

_When starting in a fresh chat, open this file and the latest `*_U58_attestation.json` in `runs/`. That’s enough context to reconstruct the attestation story._


### Clopper–Pearson consistency at the certification threshold (U31/U58)

To check that the U31 certification threshold is statistically sound for U58, we computed exact two-sided 95% Clopper–Pearson confidence intervals for the keyed and forger acceptance probabilities at the published certification threshold `min_PASS_r` (as stored in the U58 witness JSON).

For the ibm_torino U58 run, the attestation artifact records:
- `shots_per_pair = 400`, `num_pairs = 13`, so `n_per_hyp = 5200` shots under each hypothesis;
- at `r = min_PASS_r`, we have `keyed_succ = 5174` and `forger_succ = 0`, i.e.
  - empirical `TPR = 5174 / 5200 ≈ 0.995`,
  - empirical `FPR = 0 / 5200 = 0`.

Using `runner/u31_u58_cp_intervals.py`, we compute the exact Clopper–Pearson 95% intervals for these binomial proportions. The keyed interval is tightly concentrated around 0.995 and remains entirely above the target `TPR ≥ 0.99`, while the forger interval remains well below the target `FPR ≤ 0.01` (with the upper bound given by the standard zero-success CP formula for 5200 trials). In other words, at the chosen `min_PASS_r` the U58 certification point satisfies the U31 TPR/FPR specification not only at the point estimates but also at the level of 95% confidence intervals.

## U31 / U58 – Clopper–Pearson validation of cert threshold

We applied a Clopper–Pearson check to the U31 witness for the U58 attestation run:

- File: `runs/2025-11-11T19-55-37Z_U58_attestation.json`
- Shots per hypothesis: 5200 (keyed) / 5200 (forger)
- Operative threshold: `min_PASS_r = 2`
- Observed successes at this threshold:
  - keyed: 5174
  - forger: 0

From this we get:

- keyed acceptance:
  - \(\hat{p}_{\text{keyed}} = 0.9950\)
  - 95% Clopper–Pearson CI: \([0.9950,\; 1.0000]\)
- forger acceptance:
  - \(\hat{p}_{\text{forger}} = 0.0000\)
  - 95% Clopper–Pearson CI: \([0.0000,\; 0.000709]\)

Result:

- The keyed prover’s acceptance probability is statistically pinned near 1 at the certification threshold.
- The forger’s acceptance probability is statistically pinned near 0, with an upper 95% bound below \(7\times 10^{-4}\).
- The U31/U58 certification threshold therefore satisfies the pre-registered U31 “good keyed / bad forger” condition under exact binomial confidence.

