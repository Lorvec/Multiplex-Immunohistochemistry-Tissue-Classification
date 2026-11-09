from __future__ import annotations
import argparse
from pathlib import Path
from types import SimpleNamespace


def build_args(workers: int):
    return SimpleNamespace(
        phase='full',
        seeds=[42, 123, 2026],
        epochs=45,
        patience=9,
        warmup=3,
        batch=64,
        eval_batch=128,
        workers=workers,
        lr_scratch=1e-3,
        lr_pretrained=3e-4,
        weight_decay=1e-4,
        label_smoothing=0.05,
        alpha=0.55,
        temperature=3.0,
        cluster_map=None,
        allow_count_mismatch=False,
        allow_cpu=False,
    )


def main():
    p = argparse.ArgumentParser(
        description='Reproduce the MIHIC study or regenerate reports from a completed run.'
    )
    p.add_argument('--data-root',
                   help='Path containing MIHIC train/, val/, and test/ folders. '
                        'Required for the full workflow.')
    p.add_argument('--output-root', default='MIHIC_reproduction_output',
                   help='Output directory for the full workflow or existing run directory '
                        'for --report-only (default: MIHIC_reproduction_output).')
    p.add_argument('--workers', type=int, default=2,
                   help='PyTorch DataLoader workers (default: 2).')
    p.add_argument('--report-only', action='store_true',
                   help='Regenerate tables and figures, including Figure 5, from saved '
                        'run outputs and test predictions.')
    args_cli = p.parse_args()

    out = Path(args_cli.output_root).expanduser().resolve()
    if args_cli.report_only:
        if not (out / 'validation_summary.csv').is_file():
            p.error(f'--report-only requires validation_summary.csv in {out}')
        if not any(path.is_file() for path in out.glob('test_results/*/seed_*/predictions.csv.gz')):
            p.error(f'--report-only requires saved test predictions under {out / "test_results"}')

        import mihic_pipeline as m

        m.report(out)
        print('\nComplete. Results are in:', out)
        return

    if args_cli.data_root is None:
        p.error('--data-root is required for the full workflow')

    import mihic_pipeline as m

    root = m.find_root(args_cli.data_root)
    out.mkdir(parents=True, exist_ok=True)
    args = build_args(args_cli.workers)

    print('\n[1/5] Architecture and metric checks')
    m.smoke(out)

    print('\n[2/5] Dataset audit + full 57-run training')
    m.run_train(args, root, out)

    print('\n[3/5] Hardware efficiency benchmark')
    m.run_bench(args, out)

    print('\n[4/5] Final test evaluation')
    m.eval_test(args, root, out)

    print('\n[5/5] Tables, figures, calibration, and machine-readable reports')
    m.report(out)

    print('\nComplete. Results are in:', out)


if __name__ == '__main__':
    main()
