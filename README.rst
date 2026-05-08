SYNASC experiments: Mackey–Glass classical vs PINN
====================================================

This directory is a **self-contained reproducibility bundle** for the SYNASC /
IEEE-style comparison between a **method-of-steps classical DDE solver** and a
**PyTorch physics-informed neural network (PINN)** on the scalar Mackey–Glass
equation with fixed delay.

The method and experimental protocol are described in the companion **SYNASC / IEEE
submission** associated with this archive. Cite that publication when you use this
code or reproduce its tables and figures.

All configuration YAML files, Python drivers, dependency specifications, and the
default **results** tree live **under this folder**. Clone the public GitHub
repository **PINN-and-Method-of-Steps-Integration-for-the-Mackey-Glass-DDE**, then
``cd`` into the directory that contains this ``README.rst`` (in the authors'
layout that is the ``synasc/`` folder inside the repo).  Run all commands below
from that directory so relative paths resolve correctly.

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

Regenerate figures and metrics with the **YAML files and hyperparameters** that
match the published article (see the paper text and any supplementary
hyperparameter tables). After changing code or configs in this bundle, rerun the
drivers and replace downstream assets so numbers and plots stay aligned with what
you report.

For **citations and links**, use the **repository or artifact URL** printed in
the paper itself (e.g. GitHub, Zenodo, or conference supplemental material)—not an
unpublished private repository.

**GitHub repository name:** ``PINN-and-Method-of-Steps-Integration-for-the-Mackey-Glass-DDE``.

Clone (replace ``<account>`` with the organisation or username that hosts the repo)::

   git clone https://github.com/<account>/PINN-and-Method-of-Steps-Integration-for-the-Mackey-Glass-DDE.git

Experiment scripts and configs for this paper live in the ``synasc/`` subdirectory
after you clone.


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

Older copies of some YAMLs may also exist under ``../src/main_programs/configs/``
when this bundle lives inside a larger checkout; **this** ``synasc/configs/`` tree
is what readers of the public archive should use for standalone reproduction.


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


Backward compatibility (monorepo checkout)
------------------------------------------

If this ``synasc/`` folder sits inside a larger private project, thin wrappers at
that repository's root may still forward to these scripts::

   python run_synasc_comparison.py
   python run_synasc_multi_seed.py

They execute the implementations in ``synasc/`` with identical behavior.


License and contact
-------------------

Reuse and attribution follow the **license** file distributed with this bundle (if
present).  For questions about the experiments, use the **contact or author
information in the published paper** (or its supplement).

