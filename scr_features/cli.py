"""Command line entry points."""

from __future__ import annotations

import argparse
import json

from .features import audit, fit, score
from .sql import load_config, render_anchor_sql, render_fit_sql, render_sql


def main() -> None:
    parser = argparse.ArgumentParser(prog="scr-features")
    sub = parser.add_subparsers(dest="command", required=True)
    render = sub.add_parser("render-sql")
    render.add_argument("--config", required=True)
    render.add_argument("--month", required=True)
    render.add_argument("--output-s3", required=True)
    render.add_argument("--bucket-count", type=int, default=1)
    render.add_argument("--bucket-index", type=int, default=0)
    render.add_argument("--partitioned-buckets", type=int, default=0)
    render_fit = sub.add_parser("render-fit-sql")
    render_fit.add_argument("--config", required=True)
    render_fit.add_argument("--anchor", required=True)
    render_fit.add_argument("--earliest-application", required=True)
    render_fit.add_argument("--sample-mod", type=int, default=1000)
    render_fit.add_argument("--output-s3", required=True)
    render_anchor = sub.add_parser("render-anchor-sql")
    render_anchor.add_argument("--config", required=True)
    render_anchor.add_argument("--month", required=True)
    render_anchor.add_argument("--offset-months", type=int, required=True)
    render_anchor.add_argument("--fit-cutoff", required=True)
    render_anchor.add_argument("--output-s3", required=True)
    render_anchor.add_argument("--bucket-count", type=int, default=1)
    render_anchor.add_argument("--bucket-index", type=int, default=0)
    render_anchor.add_argument("--partitioned-buckets", type=int, default=0)
    fit_cmd = sub.add_parser("fit")
    fit_cmd.add_argument("--raw", required=True)
    fit_cmd.add_argument("--earliest-application", required=True)
    fit_cmd.add_argument("--out", required=True)
    fit_cmd.add_argument("--history-months", type=int, default=12)
    fit_cmd.add_argument("--dimensions", type=int, default=16)
    fit_cmd.add_argument("--clusters", type=int, default=8)
    fit_cmd.add_argument("--encoder", choices=["autoencoder", "pca"], default="autoencoder")
    fit_cmd.add_argument("--max-fit-anchors", type=int, default=100000)
    score_cmd = sub.add_parser("score")
    score_cmd.add_argument("--raw", required=True)
    score_cmd.add_argument("--model", required=True)
    score_cmd.add_argument("--out", required=True)
    audit_cmd = sub.add_parser("audit")
    audit_cmd.add_argument("--features", required=True)
    args = parser.parse_args()
    if args.command == "render-sql":
        print(render_sql(load_config(args.config), args.month, args.output_s3,
                         bucket_count=args.bucket_count, bucket_index=args.bucket_index,
                         partitioned_buckets=args.partitioned_buckets))
    elif args.command == "render-fit-sql":
        print(render_fit_sql(load_config(args.config), args.anchor,
                             args.earliest_application, args.output_s3,
                             sample_mod=args.sample_mod))
    elif args.command == "render-anchor-sql":
        print(render_anchor_sql(load_config(args.config), args.month, args.offset_months,
                                args.fit_cutoff, args.output_s3,
                                bucket_count=args.bucket_count, bucket_index=args.bucket_index,
                                partitioned_buckets=args.partitioned_buckets))
    elif args.command == "fit":
        print(json.dumps(fit(args.raw, args.earliest_application, args.out,
                             history_months=args.history_months, dimensions=args.dimensions,
                             clusters=args.clusters, encoder_type=args.encoder,
                             max_fit_anchors=args.max_fit_anchors), indent=2))
    elif args.command == "score":
        print(json.dumps(score(args.raw, args.model, args.out), indent=2))
    else:
        print(json.dumps(audit(args.features), indent=2, default=str))
