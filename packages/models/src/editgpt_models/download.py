"""Fetch every model in the registry. `python -m editgpt_models.download`."""

from __future__ import annotations

import sys

from editgpt_models.registry import REGISTRY, model_path, models_dir


def main() -> int:
    print(f"models directory: {models_dir()}")
    total = 0.0
    for key, spec in REGISTRY.items():
        path = model_path(key)
        size = path.stat().st_size / 1e6
        total += size
        print(f"  {key:14} {size:7.1f} MB  {spec.role}")
        if spec.note:
            print(f"  {'':14} {'':7}     {spec.note}")
    print(f"\n  {'total':14} {total:7.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
