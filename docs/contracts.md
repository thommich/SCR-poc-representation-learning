# Input contracts and required audits

All timestamps are interpreted in the bank's documented decision timezone. Configure the Athena session timezone consistently. The current prototype assumes naive timestamp values in that timezone and rejects malformed or non-monotone times. A source owner must document the timezone and conversion for every adapter.

## Frozen application spine

One row per `application_id` + `product`, with non-null person ID, application timestamp, product, split, product-specific target and label provenance (`OBSERVED`, `INFERRED`, `UNLABELED`). The first five application months must be `TRAIN`; the last three `OOT`. Do not derive the cutoff from decision time if the model actually scored at a different time. Preserve original population, split, labels, and baseline features. Target may be null for unlabeled rows. Application IDs must not be reused for distinct events.

## Assumed source mode used now

The default mode derives event identity from the observed detailed SCR II fields: person, reference month, `anomesdia`, acquisition, protocol and raw consultation timestamp. It parses the two observed consultation formats, keeps numeric batch and external online consultations, filters to one configured origin, and selects the latest event strictly before each application. `anomesdia` is used only to prune physical partitions. This mode has no independent authorization or version ledger. Its output is tagged `ASSUMED_CONSULTATION`; presence of a selected event yields `ASSUMED_FROM_SCR_EVENT` authorization status, and no event means `UNKNOWN_AVAILABILITY` with unknown authorization. Stage 0.2 continues investigating the assumptions. The pipeline can be exercised now without setting certified-view attestations.

## Certified snapshot view

One row per immutable `event_id`, person and SCR reference month, including `available_at` and optional `replaced_at`, with `available_at < replaced_at`. `available_at` is the later of verified publication, consultation, ingestion and any processing time needed for model use. It must represent the **original** availability of the retained version. `replaced_at` marks when that version ceased to be available, not when a current flag was rewritten. `reference_month` is `YYYYMM` from the economic SCR reference, not `anomesdia`. `status` is one of `REPORT`, `NO_REPORT`, `NO_EXPOSURE`, with complete event coverage. For `REPORT`, the event must map to all and only the detailed rows for that snapshot using person, reference month, source partition, acquisition, protocol and original consultation string. The view must have no ambiguous event ties at the same availability time for a person and reference month. If partial corrections exist, construct complete snapshots upstream first. Replica protocols require certified source-event lineage; otherwise omit them from this view.

Check before setting `snapshot_ledger_complete` and `source_snapshot_join_audited`:

* `event_id` is unique, and a person/reference month has at most one active event at any instant.
* Every `REPORT` event has detailed rows, no `NO_REPORT` event has rows, and source event join keys identify exactly one retained snapshot. Compare counts and sums before and after the join.
* Every event's availability and replacement evidence is independently archived. Current `S/N` flags, `dat_vige`, `dat_atui_rgto`, `anomesdia` and latest version alone do not pass.
* The detailed source's candidate key and balance additivity are checked for each acquisition family actually used. This package defaults to one explicitly configured origin code so market and internal totals are not blindly added.

## Authorization interval view

One row per person and authorization interval with `valid_from`, optional `valid_to`, and Boolean `permitted_use`. Intervals must be non-overlapping for the intended SCR use and historically accurate, including revocation. An absent row means `NO_AUTHORIZATION` only after completeness is certified for all persons and periods; otherwise it means `UNKNOWN_AUTHORIZATION` and extraction is blocked. Authorization is checked at application time, not consultation time alone. If use also requires authorization at consultation, encode that in the certified snapshot view and document it.

## Coverage statuses

`NO_AUTHORIZATION`: certified ledger has no permitted interval at application time. `UNKNOWN_AVAILABILITY`: authorized, but no certified eligible snapshot exists; do not call it no report. `NO_REPORT` and `NO_EXPOSURE`: explicit certified snapshot status. `INSUFFICIENT_HISTORY`: report exists but fewer than the configured number of distinct usable reference months. `USABLE`: enough reference months. The SQL keeps all applications. `NO_EXPOSURE` may be an explicit event with zero balances; do not infer it from a missing detailed row.

## Fit contract

Fit data must use the same source adapter as scoring, with anchor timestamps strictly earlier than the earliest application scored. In assumed mode, `render-fit-sql` samples people from pre-anchor SCR consultations rather than later applicant membership. Only `USABLE` rows enter preprocessing, autoencoder or PCA, and KMeans fit. Fit labels are never read by the training code. Save the fit cutoff and artifact version. Score rejects any application at or before the fit corpus maximum anchor timestamp. The fitted transformer is frozen across all eight model months.

## Outputs and acceptance

At two pilot vintages, compare source spine key count to output key count, status distribution by product, customer status and label provenance, SCR history depth, and amounts against audited source slices. Then run all eight months. Investigate any output duplicate, missing spine key, future availability, authorization breach, or version overlap before feature selection. For B0/B1/B2/B3, preserve incumbent population, split, target, tuning budget, and observed OOT evaluation; pseudo-labels are not ground truth.
