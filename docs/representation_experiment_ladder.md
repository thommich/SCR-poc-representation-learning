# SCR representation experiment ladder

## Question and scope

Determine, in order, whether (1) SCR month order warrants a temporal encoder, (2) a learned **application representation** adds value beyond conventional features, and (3) movement through monthly representation space adds value beyond a static application position. Each decision has an inexpensive fallback. This is an experiment specification, not a claim of measured uplift.

The supplied `feature_rep.md` and `feature_rep_2.md` discuss transactions and recipient identities. Their **ablation logic** transfers to SCR; their recipient-specific conclusions do not. Here, “entity representation” means a contextual vector for **one application as of its decision cutoff**, built from that person's SCR history. It is not a learned lookup vector for application ID. Modality and institution embeddings are optional internal record-encoder components, not the entity-representation hypothesis being tested.

## Representation hierarchy to materialize

For each application at time `T`, encode eligible SCR records using one shared record encoder. Pool their vectors separately for each SCR reference month `r`, producing `z(person, r | T)`. The `| T` matters: only record versions available by `T` may enter that monthly vector. Feed the up-to-12 monthly vectors, their ages and observed/missing masks into one history aggregator to produce `h(application, T)`, the 16–32D application representation. Materialize the monthly vectors **before** aggregating so one encoder supports both the compressed view and reference-month trajectory features. The monthly and application vectors can have equal dimensions without sharing the same geometric meaning; fit separate reference centroids for each space.

For a stricter trajectory of the person's *as-known* state, run that same frozen history aggregator at successive monthly as-of anchors `t<T` to obtain `h(person, t)`. Every anchor must use only information available by `t`. This differs from a reference-month path formed from `z(person, r | T)`: the latter can include an older SCR month first consulted later, provided it was known by application `T`. Both are valid application-time inputs, but they answer different questions and must be named separately.

## Fixed experimental contract

* Use the frozen ~2M-application spine, its five train and three OOT months, exact product-specific targets, existing baseline features, and incumbent eligibility rules. Preserve one output row per application, including absent or thin SCR histories. Label provenance remains `OBSERVED`, `INFERRED`, or `UNLABELED`; primary OOT conclusions use mature observed outcomes. Evaluate both products separately and pooled only with an explicit weighting rule.
* Use the same SCR source adapter, as-of reconstruction rule, record fields, 12-month lookback candidate, data exclusions, and missing-state handling throughout a comparison. The present `assumed_consultation` adapter is a working hypothesis while Stage 0.2 continues; save its `pit_basis` in every artifact. A later certified adapter requires a fresh, separately identified run.
* Freeze encoder preprocessing and weights before the earliest scored historical application, or use a documented sequence of rolling cutoffs that never trains on later information for an earlier application. Fit cluster centers, reference populations, normalization scales, and any learned risk prototypes to the same admissible cutoff. Do not fit them on all eight application months. Keep OOT out of fit, tuning, feature selection, and threshold selection.
* Use the incumbent's existing validation procedure within the five training months. If it has no fixed temporal validation, designate the last train month before experimentation and apply it to every contender. Reserve all three OOT months for a locked evaluation. Match model tuning budgets and populations.
* Pre-register the smallest economically worthwhile OOT uplift per product and acceptable compute/latency cost before seeing OOT results. Report uncertainty and cost even for a negative result. Do not promote a more complex model on self-supervised loss alone.

## 1. Is temporal representation justified?

**Hypothesis.** Chronological interactions among monthly SCR states add information beyond record content and recency.

Input is the same materialized monthly `z(person, r | T)` sequence for each applicant. Fit the record/month encoder once and **freeze its weights** for the primary T0–T4 ablation so only the history aggregator changes. Every variant uses the same records, history window, month availability and missing-month mask. Use one simple self-supervised task, such as masked historical-state reconstruction; no credit target trains the encoder. Keep the output dimension, training corpus, optimizer budget, early-stopping rule and random seeds aligned. Record parameter count and training/inference time. End-to-end retraining can be a secondary robustness check, clearly separated from the controlled ablation.

| Variant | Change | Main comparison |
|---|---|---|
| T0: set pooling | Pool month vectors without age information | Content baseline |
| T1: time-aware pooling | Add age-to-application and missing-month indicators, then pool without sequential interaction | T1 − T0 estimates recency value |
| T2: ordered GRU | Feed the same month vectors and time attributes in chronological order | T2 − T1 tests sequential interactions, but also changes capacity |
| T3: shuffled GRU | Same GRU, inputs, parameters and budget as T2; reshuffle month order for each training epoch and average several shuffled orders at inference | T2 − T3 is the closest capacity-controlled order test |
| T4: ordered GRU with elapsed-month gap | Add the actual gap between successive observed months | T4 − T2 tests explicit irregular spacing |

The T2–T3 contrast has a limit: age tags reveal chronology even after shuffling. It tests the benefit of **feeding the model in order**, not whether order is absent from the data. Repeated shuffling prevents the control from exploiting one arbitrary fixed order. T1 remains the key baseline for whether an ordered model adds useful interactions beyond recency features.

For each variant, export one frozen application representation and evaluate it with the *same* LightGBM probe and feature budget on internal validation. Also report held-out self-supervised loss, but do not choose solely on that measure. Lock the choice before OOT. Promote ordered modeling only if its incremental OOT gain over T1 and T3 is economically useful, directionally stable across the three months and random seeds, and worth its compute cost. Otherwise use T1 or T0. “No temporal gain” is a valid outcome.

## 2. Is an application representation justified?

**Hypothesis.** A frozen `h(application, T)` and its geometric derivatives carry incremental credit-risk information beyond the incumbent feature book and a strong conventional SCR-history block. The application ID is a join key, never an input to an embedding lookup or encoder. Fix the temporal backbone chosen in step 1 and export one vector per application.

The simple incumbent comparison requested is B0 versus B2 below. It answers whether adding the representation features helps the current LGBM. B1 versus B3 is the stronger test of whether **learning a representation** helps rather than merely adding previously absent SCR information. Compare on identical application rows, outcomes, splits and tuning budgets. If B2 wins but B3 does not, the application vector may be a useful compact package of SCR history without adding information beyond conventional history features.

Optional later ablation: hold the chosen history aggregator fixed and compare learned modality/institution record-token embeddings with one-hot/frequency-coded tokens of matched width. This tests the *internal categorical encoder*, a different question from whether application representations improve LGBM. Do not use `cod_chav_risc_bace` as a persistent entity ID until cross-month stability is verified.

## 3. Incremental LightGBM ladder

Freeze the winning encoder, all geometry references, and the feature lists. Train product-specific LightGBM challengers using the incumbent's population, target, split and tuning budget. Use disjoint feature blocks so each increment has a clear interpretation.

| Model | Added inputs | Question |
|---|---|---|
| B0 | Incumbent selected features | Frozen baseline |
| B1 | B0 + stronger conventional SCR history and coverage features | Can ordinary SCR engineering explain the gain? |
| B2 | B0 + static representation geometry | Value beyond incumbent only |
| B3 | B1 + static representation geometry | Value beyond a stronger SCR baseline |
| B4 | B3 + geometry of the reference-month `z(person, r | T)` path | Value of changes among reported SCR months |
| B5 | B3 + geometry of rolling as-of `h(person, t)` states | Value of how the known customer state evolved before application |
| B6 (diagnostic) | B1 + raw application embedding coordinates | Whether invariant geometry loses useful signal; not the primary deployment set |

Static geometry includes distance to frozen unsupervised centroids, nearest and second-nearest distance, their margin or ratio, distance to a pre-cutoff reference population center, and local-density distance to a pre-cutoff neighbor sample. Include missing-history flags. Avoid treating a cluster number as ordinal. A cluster ID is a category whose labels can permute between fits; use it only with a versioned, frozen vocabulary or a verified cross-version matching step. Prefer sorted distances and label-free summaries for the main set.

Report OOT ROC AUC/Gini, paired uncertainty, KS where useful, calibration, decision-band movement, and training/inference cost. Bootstrap paired *persons* within product and OOT month so repeated applications remain together. Show monthly estimates and performance by customer status, history depth, source/authorization state and observed versus inferred label provenance. Do not score pseudo-labels as ground truth. A B2 gain that disappears in B3 suggests the representation is mostly repackaging conventional SCR information.

## 4. Does trajectory geometry add value?

The notebook workflow now implements the shared record encoder, learned monthly `z` vectors and rolling `h` states described above. The older `scr_features.features` scorer remains a flattened-summary baseline. Version the monthly reference-state table by person, reference month, application cutoff and encoder; version the rolling as-of-state table by person, anchor time and encoder. Recompute every state in a path with the *same frozen encoder, scaler and corresponding reference centroids*. Never subtract vectors from different model versions or from the `z` and `h` spaces.

For states `h_1,...,h_n` at observed anchors `t_1<...<t_n`, define `d(a,b)=||a-b||` and `Δt_j=t_j-t_{j-1}` in months. Use explicit missing values and observation counts when fewer than two or three states exist. Do not interpolate missing months as unchanged behavior.

| Feature family | Examples | Minimum states |
|---|---|---:|
| Step and speed | Latest, mean, maximum and upper-quantile `d(h_j,h_{j-1})/Δt_j`; most recent 1/3/6-month displacement where both anchors exist | 2 |
| Path shape | Total path length `Σd(h_j,h_{j-1})`, net displacement `d(h_n,h_1)`, path efficiency `net/(path+ε)`, distance from current state to earlier personal-state center | 2–3 |
| Change in speed | Difference between consecutive scalar speeds, largest speed jump, latest speed divided by historical median speed | 3 |
| Direction change | Cosine between consecutive **displacement vectors** `(h_j-h_{j-1})` and `(h_{j-1}-h_{j-2})`; omit zero-length steps | 3 |
| Region approach | For each frozen unsupervised centroid `c_k`, signed approach speed `[d(h_{j-1},c_k)-d(h_j,c_k)]/Δt_j`; positive means approaching. Also minimum distance, nearest-region margin and threshold crossings | 2 |
| Region occupancy | Distinct nearest regions visited, dwell fraction in the current region, transition count, concentration/entropy of assignments; these do not use the numerical cluster labels as ordered values | 2 |
| Monthly exposure mix | Change in verified nonnegative exposure shares by modality/institution across reference months; a conventional comparator to latent movement | 2 |

Start B4 with a small preregistered block in monthly `z` space: latest speed, 3-month displacement, path efficiency, latest approach speed to the centroid nearest the **current** monthly state (measure both endpoints against that same centroid), and region-transition count. Test the analogous B5 block separately in rolling `h` space. Add other families only if these blocks show validation value. Divide by the actual observed month gap; do not turn sparse histories into artificial zero speed. A “risk region” centroid derived from default labels requires its own mature, temporally valid training corpus; unsupervised centroids are the default. A post-hoc high-risk label on a centroid is interpretation, not permission to fit or select it using OOT outcomes.

## What is genuinely invariant across encoder retrains?

If *all* embeddings and reference centroids from run B equal those in run A after the same rigid transformation `h_B=Qh_A+b`, where `QᵀQ=I`, Euclidean distances, path lengths, speeds, nearest-centroid distances and angles **between displacement vectors** are identical. Raw coordinates and embedding norm are not translation invariant. Cosine similarity of two vectors measured from the origin is not translation invariant. A global scale change multiplies raw distances and speeds; normalize them by a frozen reference scale (for example the pre-cutoff median pairwise distance) if scale drift is expected.

Independent training runs are **not guaranteed** to differ only by a rigid transform. They may stretch, collapse or rearrange regions. Distances therefore provide geometric invariance under a specific condition, not automatic stability across retraining. Cluster IDs can also permute. Every feature must carry `encoder_version`, `centroid_version`, `reference_population_version`, as-of time, and source basis. Fit and score with one version within an experiment.

Before relying on cross-retrain stability, encode a fixed pre-cutoff anchor panel with at least three seeds and a later valid retrain. Evaluate the `z` and `h` spaces separately. Compare pairwise-distance rank correlation, nearest-neighbor overlap, cluster matching, Procrustes residual, feature Spearman correlation, decile agreement and B3–B5 OOT uplift. Refit centroids and reference scales **inside each run** using that run's admissible historical corpus. If geometry or feature rankings move materially, keep the version frozen for the model lifetime or align models on a stable anchor panel; do not silently mix versions in a feature table.

## Deliverables and decision record

1. Locked experiment manifest: audience/splits/targets, source basis, history window, field list, fit cutoff, seeds, model sizes, training budget, conventional feature list and minimum economic lift.
2. T0–T4 checkpoints with self-supervised and downstream validation scorecards; one selected frozen encoder or an explicit simple-baseline decision. Modality/institution token ablations are optional and separate.
3. Application `h` embeddings, monthly reference-state `z` embeddings, rolling as-of `h` embeddings, versioned centroids/reference scales, coverage flags and LightGBM feature blocks keyed to the frozen application spine.
4. B0–B6 OOT comparison, paired intervals, monthly/segment tables, calibration/decision impact and compute cost.
5. Short decision: keep simple summaries, keep static application geometry, add ordered history, add reference-month trajectories, add as-of trajectories, or stop. State what remains unproven.
