"""Generate the checked-in, readable SCR experiment notebooks."""

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).parent


def write(name, cells):
    notebook = nbf.v4.new_notebook(
        cells=[nbf.v4.new_markdown_cell(body) if kind == "md" else nbf.v4.new_code_cell(body)
               for kind, body in cells],
        metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                  "language_info": {"name": "python"}},
    )
    nbf.validate(notebook)
    nbf.write(notebook, ROOT / name)


write("00_extract_and_month_vectors.ipynb", [
    ("md", """# 00 — Point-in-time extract and monthly SCR vectors

**Purpose.** Freeze the application audience, render/run Athena extraction, fit a record encoder on pre-cutoff unlabeled SCR, and materialize one learned vector per application × SCR reference month. Run this notebook before 01 and 02.

The default `DEMO=True` makes synthetic data and is only an execution check. For bank data, set `DEMO=False`, fill `REAL`, install `python -m pip install -e '.[experiments,aws]'`, and use fresh empty S3 UNLOAD prefixes. Physical spine columns and eight months remain explicit in the private JSON config. Stage 0.2's availability and authorization investigation continues; the source basis is recorded in every output.

**Temporal contract.** The fit anchor and every record, history model, centroid and reference scale must predate the earliest application being scored. Each monthly vector at application `T` uses only records selected as available before `T`. No credit label trains this encoder."""),
    ("code", """from pathlib import Path
import json, sys
import pandas as pd

REPO = Path.cwd()
if not (REPO / 'scr_features').exists():
    REPO = REPO.parent
sys.path.insert(0, str(REPO))

DEMO = True
WORK = REPO / 'notebook_runs' / ('demo' if DEMO else 'real')
WORK.mkdir(parents=True, exist_ok=True)
EXECUTE_ATHENA = False
BUCKET_COUNT = 16
ANCHOR_OFFSETS = (1, 2, 3)
RUN_TAG = 'replace_with_unique_run_id'
FIT_SAMPLE_MOD = 1000  # Increase this divisor if the historical fit extract is too large.

# Fill these for the real run. Each application group is one month × hash bucket.
# Anchor groups use the same keys and can glob several as-of offsets.
REAL = {
    'config': '', 'fit_anchor': '', 'earliest_application': '',
    'fit_raw': '', 'application_groups': {}, 'anchor_groups': {},
    'incumbent': '', 'validation_month': '',
    'baseline_columns': [], 'train_provenances': ['OBSERVED'],
    's3_root': '', 'athena_workgroup': '', 'query_result_s3': '', 'region': None,
}
print('Workspace:', WORK, '| synthetic demo:', DEMO)"""),
    ("md", """## A. Render Athena SQL (real mode)

The fit query samples people from consultations before `fit_anchor` without using future applicant membership. Application queries retain the full audience. Anchor queries re-extract earlier as-of states; they omit anchors at or before the representation fit cutoff. `EXECUTE_ATHENA=True` submits the generated UNLOAD statements and downloads Parquet with the AWS default credential chain. Review generated SQL and expected scan cost before enabling it. Leave it false if your team runs SQL separately; then populate local paths in `REAL`."""),
    ("code", """from scr_features.sql import load_config, render_fit_sql, render_sql, render_anchor_sql
from scr_features.athena import execute_unload, download_parquet_prefix
from scr_features.demo import make_demo_data

if DEMO:
    demo = make_demo_data(str(WORK / 'raw'), applicants_per_product_month=16)
    inputs = {**demo,
              'application_groups': {'demo': demo['application_raw']},
              'anchor_groups': {'demo': demo['anchor_raw']},
              'baseline_columns': ['incumbent_score_a', 'incumbent_score_b'],
              'train_provenances': ['OBSERVED']}
else:
    cfg = load_config(REAL['config'])
    sql_dir = WORK / 'sql'
    sql_dir.mkdir(exist_ok=True)
    root = REAL['s3_root'].rstrip('/')
    if not root.startswith('s3://') or not REAL['fit_anchor']:
        raise ValueError('Set s3_root and fit_anchor in REAL')
    jobs = []
    fit_s3 = f'{root}/{RUN_TAG}/fit/'
    jobs.append(('fit', render_fit_sql(cfg, REAL['fit_anchor'], REAL['earliest_application'],
                                       fit_s3, sample_mod=FIT_SAMPLE_MOD), fit_s3))
    for month in cfg['application_months']:
        prefix = f'{root}/{RUN_TAG}/applications/month={month}/'
        jobs.append((f'application_{month}', render_sql(
            cfg, month, prefix, partitioned_buckets=BUCKET_COUNT), prefix))
        for offset in ANCHOR_OFFSETS:
            anchor_prefix = f'{root}/{RUN_TAG}/anchors/month={month}/offset={offset}/'
            jobs.append((f'anchor_{month}_o{offset}', render_anchor_sql(
                cfg, month, offset, REAL['fit_anchor'], anchor_prefix,
                partitioned_buckets=BUCKET_COUNT), anchor_prefix))
    for name, sql, _ in jobs:
        (sql_dir / f'{name}.sql').write_text(sql, encoding='utf-8')
    print(len(jobs), 'SQL files written to', sql_dir)
    if EXECUTE_ATHENA:
        if RUN_TAG == 'replace_with_unique_run_id':
            raise ValueError('Choose a fresh RUN_TAG before submitting Athena queries')
        for name, sql, prefix in jobs:
            execute_unload(sql, workgroup=REAL['athena_workgroup'],
                           query_result_s3=REAL['query_result_s3'], region=REAL['region'])
            try:
                download_parquet_prefix(prefix, str(WORK / 'downloaded' / name), region=REAL['region'])
            except ValueError as exc:
                if not name.startswith('anchor_') or 'No Parquet files' not in str(exc):
                    raise  # An early anchor may validly have no post-fit-cutoff rows.
    inputs = {**REAL}
    if EXECUTE_ATHENA:
        inputs['fit_raw'] = str(WORK / 'downloaded' / 'fit' / '*.parquet')
        inputs['application_groups'] = {
            f'{month}_b{bucket:02d}': str(WORK / 'downloaded' / f'application_{month}' / f'bucket={bucket}' / '*.parquet')
            for month in cfg['application_months'] for bucket in range(BUCKET_COUNT)
            if list((WORK / 'downloaded' / f'application_{month}' / f'bucket={bucket}').glob('*.parquet'))}
        inputs['anchor_groups'] = {
            f'{month}_b{bucket:02d}': str(WORK / 'downloaded' / f'anchor_{month}_o*' / f'bucket={bucket}' / '*.parquet')
            for month in cfg['application_months'] for bucket in range(BUCKET_COUNT)
            if list((WORK / 'downloaded').glob(f'anchor_{month}_o*/bucket={bucket}/*.parquet'))}
    if not inputs['fit_raw'] or not inputs['application_groups'] or not inputs['incumbent']:
        raise ValueError('Set the local fit, application and incumbent paths in REAL after Athena extraction')
print('Application extract groups:', len(inputs['application_groups']))"""),
    ("md", """## B. Fit the shared record-to-month encoder

The small record MLP consumes current and overdue balances plus verified modality/institution codes. Within-month pooling yields `z` and a masked-record reconstruction task trains it without credit labels. This is a separate model from the later history aggregators. In production, increase epochs only through the pre-registered training budget and inspect reconstruction loss."""),
    ("code", """from scr_features.hierarchy import fit_record_encoder, encode_months

record_dir = WORK / 'models' / 'record_month'
record_manifest = fit_record_encoder(
    inputs['fit_raw'], inputs['earliest_application'], str(record_dir),
    month_dim=16, epochs=2 if DEMO else 25, batch_months=128 if DEMO else 256)
print({k: record_manifest[k] for k in ('version', 'fit_max_anchor', 'monthly_fit_rows', 'final_reconstruction_loss')})"""),
    ("md", """## C. Materialize monthly vectors and preserve all applications

Each extract group is scored separately to bound memory. `apps.parquet` retains covered and uncovered applications; `months.parquet` contains only observed reference months. The same frozen record encoder scores the pre-cutoff fit corpus, the eight application months, and earlier anchor extracts. Anchor extracts are needed only for the separate B5 as-of trajectory."""),
    ("code", """fit_months = WORK / 'vectors' / 'fit_months.parquet'
fit_apps = WORK / 'vectors' / 'fit_apps.parquet'
encode_months(inputs['fit_raw'], str(record_dir), str(fit_months), str(fit_apps), for_fit=True)

groups = {}
for key, raw in inputs['application_groups'].items():
    month_path = WORK / 'vectors' / key / 'months.parquet'
    app_path = WORK / 'vectors' / key / 'apps.parquet'
    encode_months(raw, str(record_dir), str(month_path), str(app_path))
    groups[key] = {'raw': raw, 'months': str(month_path), 'apps': str(app_path)}

anchors = {}
for key, raw in inputs['anchor_groups'].items():
    month_path = WORK / 'vectors' / f'anchor_{key}' / 'months.parquet'
    app_path = WORK / 'vectors' / f'anchor_{key}' / 'apps.parquet'
    encode_months(raw, str(record_dir), str(month_path), str(app_path))
    anchors[key] = {'raw': raw, 'months': str(month_path), 'apps': str(app_path)}

stage00 = {'demo': DEMO, 'fit_months': str(fit_months), 'fit_apps': str(fit_apps),
           'record_model': str(record_dir), 'groups': groups, 'anchors': anchors,
           'incumbent': inputs['incumbent'], 'earliest_application': inputs['earliest_application'],
           'validation_month': inputs['validation_month'],
           'baseline_columns': inputs['baseline_columns'],
           'train_provenances': inputs['train_provenances']}
(WORK / 'stage00.json').write_text(json.dumps(stage00, indent=2), encoding='utf-8')
print('Applications:', sum(len(pd.read_parquet(v['apps'])) for v in groups.values()),
      '| monthly vectors:', sum(len(pd.read_parquet(v['months'])) for v in groups.values()))"""),
    ("md", """**Check before proceeding:** compare the summed application count with the frozen spine, inspect coverage by product/month/customer status, and investigate any missing bucket or duplicate key. Do not interpret synthetic demo numbers as evidence."""),
])


write("01_temporal_models.ipynb", [
    ("md", """# 01 — Does temporal modeling help?

Runs T0 set pooling, T1 time-aware pooling, T2 ordered GRU, T3 shuffled GRU and T4 gap-aware GRU on the **same frozen monthly vectors**. Every history model predicts a withheld recent month vector using only pre-cutoff unlabeled histories. The LightGBM probe uses the training-period validation month only. No OOT score is inspected here.

T2 versus T3 controls GRU capacity but age tags still reveal chronology; treat it as an order-input test, not proof that timestamps contain no order information. Choose a variant using validation, seed stability and compute, then freeze it before notebook 02."""),
    ("code", """from pathlib import Path
import json, sys
import pandas as pd
REPO = Path.cwd()
if not (REPO / 'scr_features').exists(): REPO = REPO.parent
sys.path.insert(0, str(REPO))
WORK = REPO / 'notebook_runs' / 'demo'  # Switch to 'real' after notebook 00 real mode.
stage00 = json.loads((WORK / 'stage00.json').read_text(encoding='utf-8'))
DEMO = stage00['demo']
SEEDS = [17] if DEMO else [17, 29, 43]
EPOCHS = 2 if DEMO else 40
VARIANTS = ['T0', 'T1', 'T2', 'T3', 'T4']
print('Fit cutoff:', json.loads((Path(stage00['record_model'])/'manifest.json').read_text())['fit_max_anchor'])"""),
    ("code", """from scr_features.hierarchy import fit_history, encode_applications
from scr_features.io import combine_parquet
from scr_features.evaluation import temporal_probe

results = []
runs = {}
for variant in VARIANTS:
    for seed in SEEDS:
        key = f'{variant}_seed{seed}'
        model_dir = WORK / 'models' / 'history' / key
        fit_manifest = fit_history(stage00['fit_months'], stage00['fit_apps'],
            stage00['record_model'], stage00['earliest_application'], str(model_dir),
            variant=variant, epochs=EPOCHS, seed=seed)
        fit_emb = WORK / 'embeddings' / key / 'fit.parquet'
        encode_applications(stage00['fit_months'], stage00['fit_apps'], str(model_dir),
                            str(fit_emb), for_fit=True)
        scored = {}
        encoding_seconds = 0.0
        for group, paths in stage00['groups'].items():
            out = WORK / 'embeddings' / key / f'{group}.parquet'
            score_result = encode_applications(paths['months'], paths['apps'], str(model_dir), str(out))
            encoding_seconds += score_result['encoding_seconds']
            scored[group] = str(out)
        combined = WORK / 'embeddings' / key / 'all_applications.parquet'
        combine_parquet(list(scored.values()), str(combined))
        runs[key] = {'model': str(model_dir), 'fit_applications': str(fit_emb),
                     'scored': scored, 'combined': str(combined),
                     'encoding_seconds': encoding_seconds}
        for product in ('CREDIARIO', 'CONSIG_EP'):
            report = temporal_probe(stage00['incumbent'], {key: str(combined)},
                stage00['baseline_columns'], product=product,
                validation_month=stage00['validation_month'],
                train_provenances=stage00['train_provenances'],
                model_params={'n_estimators': 30, 'min_child_samples': 5} if DEMO else None)
            metrics = report['variants'][key]
            results.append({'variant': variant, 'seed': seed, 'product': product,
                            'validation_gini': metrics['gini'],
                            'missing_rate': metrics['missing_embedding_rate'],
                            'ssl_heldout_loss': fit_manifest['heldout_prediction_loss'],
                            'history_parameters': fit_manifest['parameter_count'],
                            'training_seconds': fit_manifest['training_seconds'],
                            'encoding_seconds': encoding_seconds})
table = pd.DataFrame(results)
display(table.sort_values(['product', 'validation_gini'], ascending=[True, False]))"""),
    ("md", """## Unsupervised seed stability on a pre-cutoff anchor panel

This compares the same historical fit applications across seeds. Pairwise distance ranks and neighbor overlap are unchanged by a pure rotation; Procrustes residual measures departures after the best rigid alignment. These checks do not prove identical downstream behavior, but they flag geometry drift before feature promotion."""),
    ("code", """from scr_features.stability import compare_embeddings

stability_results = []
if len(SEEDS) > 1:
    for variant in VARIANTS:
        reference = pd.read_parquet(runs[f'{variant}_seed{SEEDS[0]}']['fit_applications'])
        for seed in SEEDS[1:]:
            candidate = pd.read_parquet(runs[f'{variant}_seed{seed}']['fit_applications'])
            metrics = compare_embeddings(reference, candidate,
                keys=['application_id', 'product'], prefix='h_')
            stability_results.append({'variant': variant, 'seed': seed, **metrics})
    display(pd.DataFrame(stability_results))
else:
    print('One seed in synthetic demo; cross-retrain stability requires multiple seeds.')"""),
    ("md", """## Lock the choice before opening OOT

The code below uses T1 only in synthetic demo mode. For the real run, inspect validation results and cost, set `SELECTED_VARIANT` explicitly, and record the rationale. Seed 17 is the predeclared export seed; other seeds are robustness checks. A model with lower self-supervised loss is not automatically the credit winner."""),
    ("code", """SELECTED_VARIANT = 'T1' if DEMO else None
if SELECTED_VARIANT not in VARIANTS:
    raise ValueError('Set SELECTED_VARIANT from train-period validation before proceeding')
chosen = f'{SELECTED_VARIANT}_seed17'
stage01 = {'chosen': chosen, 'runs': runs, 'validation_results': results,
           'stability_results': stability_results,
           'selection_basis': 'synthetic execution check' if DEMO else 'record preregistered validation/cost rationale'}
(WORK / 'stage01.json').write_text(json.dumps(stage01, indent=2), encoding='utf-8')
print('Frozen choice:', chosen)"""),
])


write("02_geometry_and_lgbm.ipynb", [
    ("md", """# 02 — Application representation, trajectories, and LightGBM

Fit unsupervised reference centroids/scales on pre-cutoff application `h` and monthly `z` embeddings. Score B1 conventional SCR history, B2/B3 static application geometry, B4 reference-month movement and B5 rolling as-of movement. B6 raw embedding coordinates are diagnostic. Each feature table retains every application. Product-specific LightGBM challengers use the same incumbent rows, labels, split and tuning budget. Primary OOT metrics use **observed mature labels only**; inferred labels can be included in training only if `train_provenances` matches the incumbent workflow.

The earlier as-of `h` trajectory requires separately extracted anchor states. The notebook fills missing B5 histories with missing features rather than zero movement. Do not call a trajectory point-in-time valid if its encoder was trained after that anchor."""),
    ("code", """from pathlib import Path
import json, sys, shutil
import pandas as pd
REPO = Path.cwd()
if not (REPO / 'scr_features').exists(): REPO = REPO.parent
sys.path.insert(0, str(REPO))
WORK = REPO / 'notebook_runs' / 'demo'  # Switch to 'real' after real notebooks 00–01.
stage00 = json.loads((WORK / 'stage00.json').read_text(encoding='utf-8'))
stage01 = json.loads((WORK / 'stage01.json').read_text(encoding='utf-8'))
DEMO = stage00['demo']
run = stage01['runs'][stage01['chosen']]
print('Frozen temporal model:', stage01['chosen'])"""),
    ("code", """from scr_features.geometry import fit_geometry, score_geometry, score_rolling_trajectory
from scr_features.hierarchy import encode_applications
from scr_features.features import export_conventional
from scr_features.io import combine_parquet

geometry_dir = WORK / 'models' / 'geometry' / stage01['chosen']
geometry_manifest = fit_geometry(run['fit_applications'], stage00['fit_months'],
                                 stage00['earliest_application'], str(geometry_dir),
                                 clusters=3 if DEMO else 8)
print({k: geometry_manifest[k] for k in ('version', 'fit_max_anchor', 'h_fit_points', 'z_fit_points')})

geometry_files, conventional_files, rolling_files = [], [], []
for group, paths in stage00['groups'].items():
    app_emb = run['scored'][group]
    geom = WORK / 'features' / 'geometry' / f'{group}.parquet'
    conventional = WORK / 'features' / 'conventional' / f'{group}.parquet'
    rolling = WORK / 'features' / 'rolling' / f'{group}.parquet'
    score_geometry(app_emb, paths['months'], str(geometry_dir), str(geom))
    export_conventional(paths['raw'], str(conventional))
    anchor_emb = None
    if group in stage00['anchors']:
        anchor_paths = stage00['anchors'][group]
        anchor_emb = WORK / 'embeddings' / stage01['chosen'] / f'anchor_{group}.parquet'
        encode_applications(anchor_paths['months'], anchor_paths['apps'],
                            run['model'], str(anchor_emb))
    if stage00['anchors']:
        score_rolling_trajectory(str(anchor_emb) if anchor_emb else None,
                                 app_emb, str(geometry_dir), str(rolling))
    geometry_files.append(str(geom))
    conventional_files.append(str(conventional))
    if stage00['anchors']:
        rolling_files.append(str(rolling))

combined_dir = WORK / 'features' / 'combined'
geometry_path = combine_parquet(geometry_files, str(combined_dir / 'geometry.parquet'))
conventional_path = combine_parquet(conventional_files, str(combined_dir / 'conventional.parquet'))
rolling_path = combine_parquet(rolling_files, str(combined_dir / 'rolling.parquet')) if rolling_files else None
geometry_feature_manifest = str(combined_dir / 'geometry.manifest.json')
shutil.copyfile(Path(geometry_files[0]).with_suffix('.manifest.json'), geometry_feature_manifest)
print('Feature files:', geometry_path, conventional_path, rolling_path)"""),
    ("md", """## Locked B0–B6 comparison

`B0` is the incumbent selected feature set. `B1` adds conventional SCR history. `B2` adds static application geometry to B0. `B3` adds it to B1 and is the stronger application-representation test. `B4` and `B5` separately add the two trajectory families to B3. `B6` is a diagnostic raw-coordinate comparison. The three OOT months are evaluated only after the temporal model was selected on train-period validation."""),
    ("code", """from scr_features.evaluation import run_ladder

MIN_DELTA_GINI = 0.0 if DEMO else None  # Set a product-approved economic threshold before real OOT evaluation.
DECISION_THRESHOLD = None  # Optionally set the incumbent policy threshold for crossing rates.
if not DEMO and MIN_DELTA_GINI is None:
    raise ValueError('Pre-register MIN_DELTA_GINI before real OOT evaluation')
scorecards = {}
for product in ('CREDIARIO', 'CONSIG_EP'):
    scorecards[product] = run_ladder(
        stage00['incumbent'], conventional_path, run['combined'], geometry_path,
        geometry_feature_manifest, str(WORK / 'evaluation' / product),
        stage00['baseline_columns'], product=product,
        train_provenances=stage00['train_provenances'],
        validation_month=stage00['validation_month'], rolling_path=rolling_path,
        n_bootstrap=10 if DEMO else 200,
        minimum_delta_gini=MIN_DELTA_GINI, decision_threshold=DECISION_THRESHOLD,
        model_params={'n_estimators': 30, 'min_child_samples': 5} if DEMO else None)
summary = pd.DataFrame([
    {'product': product, 'model': model, 'oot_gini': result['oot_observed']['gini'],
     'oot_auc': result['oot_observed']['auc'], 'oot_brier': result['oot_observed']['brier'],
     'paired_delta_gini_ci': result.get('paired_delta_gini_ci')}
    for product, card in scorecards.items() for model, result in card['models'].items()
])
display(summary)
print('Synthetic execution check only' if DEMO else 'Review monthly/segment metrics and economic threshold before any promotion decision')"""),
])

print("Wrote three notebooks to", ROOT)
