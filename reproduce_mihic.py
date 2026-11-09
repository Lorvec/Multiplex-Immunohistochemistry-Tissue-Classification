from __future__ import annotations
import argparse
from pathlib import Path
from types import SimpleNamespace
import mihic_pipeline as m


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
        description='Reproduce the complete MIHIC compact-CNN study from raw MIHIC folders.'
    )
    p.add_argument('--data-root', required=True,
                   help='Path containing MIHIC train/, val/, and test/ folders.')
    p.add_argument('--output-root', default='MIHIC_reproduction_output',
                   help='Directory where checkpoints, metrics, figures, and reports are written.')
    p.add_argument('--workers', type=int, default=2,
                   help='PyTorch DataLoader workers (default: 2).')
    args_cli = p.parse_args()

    root = m.find_root(args_cli.data_root)
    out = Path(args_cli.output_root).expanduser().resolve()
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
