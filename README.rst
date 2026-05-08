Mackey–Glass delay equation: classical integrator vs PINN
=======================================================================

.. raw:: html

   <p align="center">
   <img src="images/mackey_glass_n10_3d_overlay.png"
        alt="Mackey–Glass n=10: 3D delay embedding — RK4 reference (grey), classical MoS (black), PINN (purple)."
        width="85%" style="max-width: 880px;">
   </p>

**Mackey–Glass (n = 10).** Three-dimensional delay embedding (state vs two
lagged values): fine RK4 reference (grey), classical method-of-steps trajectory
(black), PINN prediction (purple). Same style of figure as the long-horizon
windowed experiment in the article.

This repository contains code to reproduce the **Mackey–Glass** benchmark from the
companion **IEEE SYNASC paper** (physics-informed neural network vs method-of-steps classical solver on a scalar DDE with fixed delay).

**Repository:** https://github.com/ml3m/PINN-and-Method-of-Steps-Integration-for-the-Mackey-Glass-DDE

After cloning, use the **repository root** (the directory that contains this
``README.rst``) as your working directory for every command below.

.. contents:: **Table of contents**
   :local:
   :depth: 2


What is included
----------------

**Programs**

* ``run_synasc_comparison.py`` — **Main experiment:** adaptive RK45
  method-of-steps trajectory, high-accuracy RK4 reference, PINN training with
  windowed time domains, metrics, and figures.

* ``run_synasc_multi_seed.py`` — **Optional robustness study:** repeats the main
  experiment for several pseudorandom seeds and aggregates uncertainty in metrics
  and learning curves.

**Supporting files**

* ``requirements.txt`` — Python stack including a **CPU** PyTorch build (suitable
  for reviewers without a CUDA install).

* ``requirements-rocm.txt`` — How to swap in **AMD ROCm** PyTorch after the base
  install; NVIDIA users should follow `PyTorch Get Started`_ instead.

* ``configs/`` — YAML recipes for horizons, windowing, and training hyperparameters
  described in the article.

* ``images/`` — Banner figure for this README (3D overlay for **n = 10**).

* ``results/`` — Default location for run outputs.

.. _PyTorch Get Started: https://pytorch.org/get-started/locally/


Matching the published results
------------------------------

Use the **configuration files and hyperparameters** stated in the paper (including
any supplement). After changing code or YAML, rerun the pipeline and replace
downstream plots or tables so reported numbers stay consistent.

**Clone**

.. code-block:: bash

   git clone https://github.com/ml3m/PINN-and-Method-of-Steps-Integration-for-the-Mackey-Glass-DDE.git
   cd PINN-and-Method-of-Steps-Integration-for-the-Mackey-Glass-DDE


Environment
-----------

* **Python** 3.10+ (3.12 tested). Use a **virtual environment**.

* ``requirements.txt`` installs NumPy, SciPy, Matplotlib, PyYAML, and a **CPU**
  PyTorch wheel (training is slower than on GPU but runs on typical laptops).

* For **CUDA** or **ROCm** PyTorch, reinstall ``torch`` as in the PyTorch site or
  ``requirements-rocm.txt``.

From the repository root::

   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   pip install -U pip
   pip install -r requirements.txt


Single run (main experiment)
-----------------------------

.. code-block:: bash

   python -u run_synasc_comparison.py \
     --config configs/config_mackey_glass_synasc_t100_windowed.yaml \
     --n-values 10 \
     --output-dir results/t100_windowed_n10

**Defaults** if you omit ``--config`` / ``--output-dir``:

* Config: ``configs/config_mackey_glass_synasc_t100_windowed.yaml``

* Output directory: ``results/`` under the repository root

Config and output paths are resolved against the current working directory, then
against the directory that contains ``run_synasc_comparison.py``.


Multi-seed run (optional)
-------------------------

Runs the main program once per seed, then builds aggregate CSV/JSON and overlay
plots.

**AMD ROCm (optional):** some GPUs need a gfx workaround. If required for your
hardware, set this in the shell **before** the Python command; otherwise omit it::

   export HSA_OVERRIDE_GFX_VERSION=10.3.0

**Run:**

.. code-block:: bash

   python -u run_synasc_multi_seed.py \
     --config configs/config_mackey_glass_synasc_t100_windowed.yaml \
     --base-output results/t100_windowed_multi_seed \
     --n-values 10 \
     --n-seeds 10 --seed-start 1000

Use ``--python /path/to/python3`` if the child jobs must use a different
interpreter than the orchestrator.


Configuration files (``configs/``)
---------------------------------

Filenames keep historical tags; each file encodes horizon and training layout:

* ``config_mackey_glass_synasc.yaml`` — Baseline setup (see YAML header).

* ``config_mackey_glass_synasc_t20.yaml`` — Short horizon ``[0,20]``.

* ``config_mackey_glass_synasc_t100_windowed.yaml`` — Windowed ``[0,100]`` as in
  the article’s long-run Mackey–Glass comparison.

* ``config_mackey_glass_synasc_t200.yaml`` — Horizon ``[0,200]``.

* ``config_mackey_glass_synasc_smoke_multi_seed.yaml`` — Minimal settings for
  quick CI or smoke tests.

Use the ``configs/`` directory **in this repository** when reproducing the paper.


Run outputs
-----------

Each run writes under ``--output-dir``:

* ``synasc_results.pkl`` — Serialized trajectories, metrics, and metadata for
  tables or follow-on scripts.

* ``mackey_glass_n*_comparison.png`` — Primary side-by-side comparison figure.

* ``n*_*.png`` / ``.pdf`` — Heatmaps, time slices, embeddings, losses, tables.

* ``checkpoints_n*/*.pt`` — PINN checkpoints per time window (runs resume if
  these exist).

Generated content under ``results/`` is excluded from git by default
(``results/.gitignore``).


License and contact
-------------------

Reuse and attribution follow the **LICENSE** file shipped here, if present.
Questions about the **Mackey–Glass experiments** should go to the authors via the
published paper or its supplement.
