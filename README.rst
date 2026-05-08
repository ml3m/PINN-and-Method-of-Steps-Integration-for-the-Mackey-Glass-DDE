SYNASC experiments: Mackey–Glass classical vs PINN
====================================================

This tree is a **self-contained reproducibility bundle** for the SYNASC / IEEE-style
comparison between a **method-of-steps classical DDE solver** and a **PyTorch
physics-informed neural network (PINN)** on the scalar Mackey–Glass equation with
fixed delay.

The method and experimental protocol are described in the companion **SYNASC / IEEE
submission** associated with this archive. Cite that publication when you use this
code or reproduce its tables and figures.

**Canonical public repository:** https://github.com/ml3m/PINN-and-Method-of-Steps-Integration-for-the-Mackey-Glass-DDE

After cloning, your **working directory** should be the folder that contains this
``README.rst`` and the Python drivers:

* **On GitHub:** repository **root** (scripts and ``configs/`` live next to this file).
* **Inside a larger private monorepo:** typically the ``synasc/`` subdirectory—use
  that path the same way.

Run all commands below from that directory.

.. contents:: **Table of contents**
   :local:
   :depth: 2


Layout
------

.. code-block:: text

   ./
   ├── README.rst                  ← This file
   ├── requirements.txt           ← Core Python dependencies
   ├── requirements-rocm.txt      ← Notes for AMD ROCm / PyTorch ROCm wheels
   ├── run_synasc_comparison.py   ← Main driver (classical + PINN + figures)
   ├── run_synasc_multi_seed.py  ← Optional: multi-seed orchestration + plots
   ├── configs/                   ← Experiment YAML files
   └── results/                   ← Default output root (artifacts git-ignored)


Relation to the paper
---------------------

Regenerate figures and metrics with the **YAML files and hyperparameters** that
match the published article (see the paper text and any supplementary
hyperparameter tables). After changing code or configs in this bundle, rerun the
drivers and replace downstream assets so numbers and plots stay aligned with what
you report.

**Clone**

.. code-block:: bash

   git clone https://github.com/ml3m/PINN-and-Method-of-Steps-Integration-for-the-Mackey-Glass-DDE.git
   cd PINN-and-Method-of-Steps-Integration-for-the-Mackey-Glass-DDE


Environment
-----------

Requirements:

* **Python** 3.10 or newer (3.12 is tested).

* A **virtual environment** is strongly recommended.

* **PyTorch** is installed automatically by ``requirements.txt`` (CPU wheels from
  PyTorch’s index—works on typical reviewer laptops without an NVIDIA stack).
  Training is slower on CPU than on a GPU but reproduction runs end-to-end.

* For **NVIDIA CUDA** or **AMD ROCm**, reinstall **torch** afterward — see
  https://pytorch.org/get-started/locally/ and ``requirements-rocm.txt``.

From the bundle root (directory that contains ``README.rst``)::

   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   pip install -U pip
   pip install -r requirements.txt

No separate PyTorch step is required unless you switch to a GPU-specific wheel.

Quick start (single run)
------------------------

.. code-block:: bash

   python -u run_synasc_comparison.py \
     --config configs/config_mackey_glass_synasc_t100_windowed.yaml \
     --n-values 10 \
     --output-dir results/t100_windowed_n10

**Defaults** if you omit ``--config`` / ``--output-dir``:

* Config: ``configs/config_mackey_glass_synasc_t100_windowed.yaml``

* Output: ``results/`` (under the bundle root)

Relative paths for ``--config`` and ``--output-dir`` are resolved first against
the current working directory, then against the directory that contains
``run_synasc_comparison.py``.

**Faster figure export:** add ``--no-pdf`` to skip vector PDF generation (PNG
only).


Multi-seed variability
----------------------

Orchestrates ``run_synasc_comparison.py`` once per seed and aggregates metrics /
plots:

.. code-block:: bash

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

If you also maintain a larger private checkout, duplicate YAMLs may exist
elsewhere; use the ``configs/`` directory **next to this README** for reproduction.


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

If this bundle lives under ``synasc/`` inside a larger repository, thin wrappers at
that repository's **root** may still forward to these scripts::

   python run_synasc_comparison.py
   python run_synasc_multi_seed.py

They run the copies under ``synasc/`` with the same behavior.


License and contact
-------------------

Reuse and attribution follow the **license** file distributed with this bundle (if
present).  For questions about the experiments, use the **contact or author
information in the published paper** (or its supplement).
