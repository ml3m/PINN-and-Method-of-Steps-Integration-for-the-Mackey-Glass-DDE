SYNASC experiments: Mackey–Glass classical vs PINN
====================================================

This directory is a **self-contained reproducibility bundle** for the SYNASC /
IEEE-style comparison between a **method-of-steps classical DDE solver** and a
**PyTorch physics-informed neural network (PINN)** on the scalar Mackey–Glass
equation with fixed delay.

The implementation is described in the companion conference paper; see the
section **Relation to the paper** below for the path to the ``.tex`` source.
specifications, and the default **results** tree live **under this folder**, so
cloning the parent repository and following the steps below is sufficient to
rerun experiments without hunting paths in ``src/``.

.. contents:: **Table of contents**
   :local:
   :depth: 2


Layout
------

.. code-block:: text

   synasc/
   ├── README.rst                 ← This file
   ├── requirements.txt           ← Core Python dependencies
   ├── requirements-rocm.txt      ← Notes for AMD ROCm / PyTorch ROCm wheels
   ├── run_synasc_comparison.py # Main driver (classical + PINN + figures)
   ├── run_synasc_multi_seed.py # Optional: multi-seed orchestration + plots
   ├── configs/                 # Experiment YAML files (copied for isolation)
   └── results/                 # Default output root (artifacts git-ignored)


Relation to the paper
---------------------

In the parent repository, the conference manuscript is maintained at::

   BACHELOR_THESIS_MAIN/SYNASC/conference.tex

Figures under ``BACHELOR_THESIS_MAIN/SYNASC/`` (or paths referenced by
``\\graphicspath`` in that manuscript) should be regenerated from runs whose
**YAML** and **hyperparameters** match the values stated in the paper.  After
code or config changes, rerun the drivers in this folder and replace the
corresponding assets so the PDF stays consistent with the repository.

If you cite this code publicly, point readers to the **GitHub URL** of the
parent ``pinn`` repository and to this ``synasc/`` subdirectory as the canonical
entry point for SYNASC reproduction.


Environment
-----------

Requirements:

* **Python** 3.10 or newer (3.12 is tested).

* A **virtual environment** is strongly recommended.

* **PyTorch** must be installed for **your** hardware (CPU, NVIDIA CUDA, or
  AMD ROCm).  See https://pytorch.org/get-started/locally/

Install core dependencies from this directory::

   cd synasc
   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   pip install -U pip
   pip install -r requirements.txt

Then install **torch** and optional CUDA/ROCm builds as instructed on the
PyTorch website.  For **AMD ROCm**, follow the comments in
``requirements-rocm.txt`` (second-step ``pip install torch
--index-url https://download.pytorch.org/whl/rocm6.x``).


Quick start (single run)
------------------------

From **inside** ``synasc/`` (recommended):

.. code-block:: bash

   python -u run_synasc_comparison.py \
     --config configs/config_mackey_glass_synasc_t100_windowed.yaml \
     --n-values 10 \
     --output-dir results/t100_windowed_n10

**Defaults** if you omit ``--config`` / ``--output-dir``:

* Config: ``configs/config_mackey_glass_synasc_t100_windowed.yaml``

* Output: ``results/`` (under this ``synasc`` directory)

Relative paths for ``--config`` and ``--output-dir`` are resolved first against
the current working directory, then against this ``synasc`` bundle—so running
from ``synasc/`` with paths like ``configs/...`` is the most predictable.

**Faster figure export:** add ``--no-pdf`` to skip vector PDF generation (PNG
only).


Multi-seed variability
----------------------

Orchestrates ``run_synasc_comparison.py`` once per seed and aggregates metrics /
Plots:

.. code-block:: bash

   cd synasc
   HSA_OVERRIDE_GFX_VERSION=10.3.0   # only if your AMD card needs the override
   python -u run_synasc_multi_seed.py \
     --config configs/config_mackey_glass_synasc_t100_windowed.yaml \
     --base-output results/t100_windowed_multi_seed \
     --n-values 10 \
     --n-seeds 10 --seed-start 1000 \
     --extra-args '--no-pdf'

Use ``--python /path/to/python3`` if the PINN interpreter must differ from the
one running the orchestrator (e.g. a system Python with ROCm).


Configurations shipped in ``configs/``
--------------------------------------

* ``config_mackey_glass_synasc.yaml`` — baseline recipe (see YAML header).

* ``config_mackey_glass_synasc_t20.yaml`` — short horizon.

* ``config_mackey_glass_synasc_t100_windowed.yaml`` — windowed ``[0,100]`` setup
  used in many SYNASC sweeps.

* ``config_mackey_glass_synasc_t200.yaml`` — longer horizon.

* ``config_mackey_glass_synasc_smoke_multi_seed.yaml`` — tiny settings for CI /
  smoke tests.

Older copies may also exist under ``../src/main_programs/configs/`` in the
parent repository; **this** ``synasc/configs/`` tree is what the README assumes
for standalone reproduction.


Outputs
-------

Each run writes to the directory given by ``--output-dir``.  Typical files:

* ``synasc_results.pkl`` — pickled metrics and trajectories for downstream tables.

* ``mackey_glass_n*_comparison.png`` — main comparison figure.

* ``n*_*.png`` / optional ``.pdf`` — heatmaps, snapshots, 3D embeddings, loss
  curves, tables.

* ``checkpoints_n*/*.pt`` — per-window PINN checkpoints (resume on rerun).


**Git:** generated artifacts under ``results/`` are ignored by default via
``results/.gitignore`` so pull requests stay small; keep YAML + code changes
under version control and regenerate plots locally or in CI.


Backward compatibility (repository root)
-----------------------------------------

For existing scripts, the repository root still provides thin wrappers::

   python run_synasc_comparison.py
   python run_synasc_multi_seed.py

They execute the implementations in ``synasc/`` with identical behavior.


License and contact
-------------------

Reuse and attribution follow the **license** and **authorship** of the parent
``pinn`` repository.  For paper-specific queries, refer to the author block in
``BACHELOR_THESIS_MAIN/SYNASC/conference.tex``.

