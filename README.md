# SCR representation feature pipeline

The step-by-step temporal, application-representation, LightGBM and trajectory validation plan is in [docs/representation_experiment_ladder.md](docs/representation_experiment_ladder.md). The new notebook workflow implements the shared record encoder, learned monthly vectors, T0–T4 history aggregators, frozen application embeddings, static/trajectory geometry and B0–B6 LightGBM comparisons. The earlier flattened-summary autoencoder remains as a simple reference implementation.

Stage 0 challenger for the full Crediário and Consignado EP application spine. It produces one row per application, including applications without an eligible SCR snapshot under the selected source adapter.

## Evidence and gate

Stage 0.1 found that `risco_bacen_scr_ii` and `risco_bacen_scr` reconcile for not-yet-due balance on one sampled January 2025 batch slice (428,653 groups). The detailed source has a sampled key containing person, partition, acquisition, protocol, consultation time, modality, origin, institution and risk key. `dat_hor_cslt_risc_bace` has multiple string formats. `anomesdia` is a storage partition, not an availability time. Stage 0.2 found mutable DOC3046 RDS partitions and no established historical bank-availability or authorization lineage for the analytical table. The linked shared chat could not be read, so no extra feature definitions are attributed to it.

The default `assumed_consultation` adapter runs with the observed `risco_bacen_scr_ii` physical fields. It assumes a parseable original consultation timestamp is the first bank-usable time for that retained record, that historical consultations are retained without later rewrite, and that `Consulta Lote` numeric and `Consulta Online / Externo` presence implies permitted use for this experiment. It selects the latest such event before each application for each SCR reference month. Only the configured origin code is included. These are **working assumptions**, not Stage 0.2 findings. Applicants without a selected event receive `UNKNOWN_AVAILABILITY`; this adapter cannot distinguish no authorization, no report or no reportable exposure from absence of a row.

The optional `certified_ledger` adapter replaces those assumptions with two independently certified Athena views:

* **Snapshot adapter**: one row per complete person × SCR reference month × consultation event, with the original bank-availability timestamp, replacement timestamp (if any), source join keys, and `REPORT`, `NO_REPORT`, or `NO_EXPOSURE` status. Its time columns must refer to the exact retained version, not the current row's update date. A snapshot must be complete; corrected snapshots need their own event row. No current `S/N` flag or maximum version is used as a historical proxy.
* **Authorization adapter**: an audited interval of authorization and permitted use for each person. Configure `authorization_ledger_complete` only after verifying the ledger covers the audience and period; absent authorization rows otherwise mean `UNKNOWN_AUTHORIZATION`.

The assumed adapter renders without those views. The certified adapter requires their attestation. The `pit_basis` column makes outputs distinguishable. Consultation time alone, `anomesdia`, `dat_vige`, assumed monthly lag, and the current `ind_esto_ssto` value have not been proven sufficient by Stage 0.2.

## Experimental contract

Hypothesis: a frozen 16D self-supervised representation of trailing SCR states adds incremental information beyond incumbent and conventional SCR features. Inherited dependencies: frozen application spine, exact product-specific targets/splits, source field DDL, and a historical pre-cutoff fitting corpus. Start with one early train and one OOT application month per product for the source audit, then scale to all eight months without changing contracts. Success: one row per application; no event with an assumed availability timestamp at or after application; identical incumbent population; coverage accounting by product/vintage/customer/label provenance; and the locked OOT ladder in the experiment plan. The 12-month window, 16 dimensions and 8 clusters are configurable experiment choices, not established SCR facts.

## Run the notebooks

Install Python 3.11+ dependencies with `python -m pip install -e '.[experiments,test]'`; add `.[aws]` if the notebook should submit Athena UNLOAD queries using your configured AWS credentials. Open and run, in order:

1. [`notebooks/00_extract_and_month_vectors.ipynb`](notebooks/00_extract_and_month_vectors.ipynb): render optional partitioned Athena UNLOAD SQL, fit the record-to-month encoder on an unlabeled pre-cutoff SCR sample, and materialize `z` vectors for the fit corpus, the eight application months and earlier as-of anchors.
2. [`notebooks/01_temporal_models.ipynb`](notebooks/01_temporal_models.ipynb): fit T0–T4 on the same frozen monthly vectors, compare train-period validation probes, inspect seed geometry stability and lock one temporal model **before** OOT evaluation.
3. [`notebooks/02_geometry_and_lgbm.ipynb`](notebooks/02_geometry_and_lgbm.ipynb): fit versioned pre-cutoff centroids/scales; export conventional, static and two trajectory feature blocks; evaluate B0–B6 separately for both products on matched rows.

All three notebooks default to a clearly labeled synthetic execution check. For bank data, set `DEMO=False` and the private `REAL` paths/configuration in notebook 00, run its Athena SQL or supply locally exported Parquet, then switch `WORK` to `real` in notebooks 01 and 02. The actual application-spine column mapping, eight month values, product-specific target definitions, source partition list, and incumbent selected feature names are deliberately supplied by that private configuration. The notebooks never infer them. The optional Athena runner reads the usual AWS credential chain; no credentials are stored here.

For a large audience, partition each month UNLOAD into 16 application-hash buckets with `--partitioned-buckets 16`. Athena writes one `bucket=` prefix per bucket while scanning the source once per month. The notebook scores each bucket separately and combines Parquet outputs by streaming. The fit sample size is controlled by `--sample-mod`. Review scan size before running all months.

Run `python -m pytest` for code tests and `python notebooks/validate_notebooks.py` to execute the three notebooks in synthetic mode. No synthetic score is a POC result.

## Earlier flattened-summary baseline

The original single-vector autoencoder remains available as a cheap comparison. It does not produce learned month vectors or trajectories.

1. Copy `config/example.json` to a private config file. Fill the frozen application-spine identifiers, eight actual application months, SCR II partitions covering their lookback, and the single origin code to evaluate. The SCR II field names and table are already wired from Stage 0.1. Keep `sources/` read-only.
2. Render an independent, unlabeled historical fit sample: `scr-features render-fit-sql --config private.json --anchor 2024-12-31T23:59:59 --earliest-application 2025-01-01T00:00:00 --sample-mod 1000 --output-s3 s3://your-bucket/fit/`. Replace these **example** dates with the actual first application timestamp and a strictly earlier anchor. The deterministic hash sample comes from SCR records with consultation timestamps before the anchor, without selecting people because they later applied. Run the printed `UNLOAD` in Athena.
3. `scr-features fit --raw 'fit/*.parquet' --earliest-application 2025-01-01T00:00:00 --out artifacts/encoder` fits the frozen autoencoder and clusters. The command checks that all fit anchors predate the first application. Use `--encoder pca` for the linear ablation.
4. `scr-features render-sql --config private.json --month YYYY-MM --output-s3 s3://your-bucket/unique-month-prefix/` prints the Athena `UNLOAD` for an application month. Use a fresh empty S3 prefix for each month. Run it in Athena; download/export the resulting Parquet files within the bank environment.
5. `scr-features score --raw 'month/*.parquet' --model artifacts/encoder --out features/month.parquet` scores one application month at a time. Repeat for all eight months, then join to the incumbent dataset by `application_id` and `product`. `scr-features audit --features 'features/*.parquet'` checks keys and time/population summaries.

The feature file contains identifiers, split, target value and provenance for joining/evaluation, coverage status, conventional trailing-history features, frozen `scr_emb_00`–`scr_emb_15`, a cluster label, distances to each centroid, embedding norm and cluster margin. The adjacent manifest separates conventional and representation columns for B1/B2/B3 LightGBM tests. Never put identifiers, split, target, or provenance into the feature matrix.

## Limits

The assumed adapter cannot prove source immutability or authorization from the fields currently known. Stage 0.2 must still verify those assumptions before making a point-in-time validity claim. The scorer checks extract-level duplicates, future events, split assignment, key uniqueness and fit cutoff. A single global frozen encoder is valid for the eight months only if its fit data was bank-available before the first application scored. The optional runner can submit Athena SQL only with the user's own configured credentials and workgroup. No bank query was executed here, and no predictive uplift is claimed.

