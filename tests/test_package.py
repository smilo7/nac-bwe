"""
Packaging check: every module ships and imports.

This exists because a `.gitignore` pattern once excluded `nac_bwe/data/` from
the repository, so installing from git produced a package with no
`nac_bwe.data`. Both trainers and the precompute entry point were dead, and the
other tests all passed because none of them import that subpackage.

The manifest below is the point. Walking the installed package with
`pkgutil` only finds what is present, so a missing module produces nothing to
fail on. Listing the expected modules explicitly is what turns an omission into
a failure. Add a line here whenever a module is added.

Run against an INSTALLED package (pip install, not PYTHONPATH) for this to mean
anything:

    python tests/test_package.py
"""

import importlib
import importlib.util
import pkgutil
import sys

EXPECTED = [
    "nac_bwe",
    "nac_bwe.checkpoints",
    "nac_bwe.codec",
    "nac_bwe.data",
    "nac_bwe.data.dataset",
    "nac_bwe.data.precompute",
    "nac_bwe.inference",
    "nac_bwe.inference.listen",
    "nac_bwe.losses",
    "nac_bwe.losses.losses",
    "nac_bwe.models",
    "nac_bwe.models.audio_bwe_net",
    "nac_bwe.models.latent_bwe_net",
    "nac_bwe.training",
    "nac_bwe.training.config",
    "nac_bwe.training.tracking",
    "nac_bwe.training.train_audio_bwe",
    "nac_bwe.training.train_latent_bwe",
]

# Entry points the README documents. A user runs these first.
ENTRY_POINTS = [
    "nac_bwe.checkpoints",
    "nac_bwe.data.precompute",
    "nac_bwe.inference.listen",
    "nac_bwe.training.train_audio_bwe",
    "nac_bwe.training.train_latent_bwe",
]


def main() -> int:
    print("\n=== package completeness ===\n")
    import nac_bwe
    print(f"nac_bwe from {nac_bwe.__file__}")
    if "src" in nac_bwe.__file__.split("/"):
        print("NOTE: imported from the source tree, so this cannot detect a\n"
              "      packaging omission. Install the package and re-run.")

    ok = True

    print(f"\n1. {len(EXPECTED)} expected modules")
    missing = []
    for m in EXPECTED:
        # find_spec raises rather than returning None when a parent package is
        # itself absent, which is exactly the case this test exists to report.
        try:
            found = importlib.util.find_spec(m) is not None
        except ModuleNotFoundError:
            found = False
        if not found:
            print(f"   MISSING {m}")
            missing.append(m)
            continue
        try:
            importlib.import_module(m)
        except Exception as e:
            print(f"   BROKEN  {m}: {type(e).__name__}: {e}")
            missing.append(m)
    ok &= not missing
    print(f"   {'OK' if not missing else 'FAILED'}: "
          f"{len(EXPECTED) - len(missing)}/{len(EXPECTED)} import")

    print("\n2. every shipped submodule imports")
    extra_broken = []
    for info in pkgutil.walk_packages(nac_bwe.__path__, prefix="nac_bwe."):
        try:
            importlib.import_module(info.name)
        except Exception as e:
            extra_broken.append(f"{info.name}: {type(e).__name__}: {e}")
        if info.name not in EXPECTED:
            print(f"   note: {info.name} ships but is not in EXPECTED, add it")
    for b in extra_broken:
        print(f"   BROKEN  {b}")
    ok &= not extra_broken
    print(f"   {'OK' if not extra_broken else 'FAILED'}")

    print("\n3. documented entry points are runnable modules")
    for m in ENTRY_POINTS:
        try:
            spec = importlib.util.find_spec(m)
        except ModuleNotFoundError:
            spec = None
        if spec is None:
            print(f"   MISSING {m}")
            ok = False
        else:
            print(f"   OK      python -m {m}")

    print(f"\n=== {'package is complete' if ok else 'PACKAGE INCOMPLETE'} ===\n")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
