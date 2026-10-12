# CHGNet reproducible inference on AI Makerspace

**Completed H100 run:** job **6199752**, October 11, 2026, 10:55 PM EDT (recorded UTC timestamp is October 12). CHGNet ran on CUDA on an NVIDIA H100 80GB HBM3 in the ICE GPU allocation. All smoke, numerical-reference, and physical-symmetry checks passed. Raw evidence is in `evidence/6199752/`; the original September ICE CPU logs and results are in `evidence/historical/`. Confirm your course's designation of this ICE H100 allocation as AI Makerspace if needed.

The job used 1 GPU, 4 CPU cores, and 16 GB requested RAM. One baseline inference took **1.317 seconds**; loading plus inference and all symmetry checks took **1.811 seconds**, excluding imports and setup. Peak Python RSS was **1,417.6 MiB**; PyTorch peak allocated/reserved GPU memory was **176.0/180.0 MiB**. These measurements exclude the editor and non-PyTorch GPU allocations.

This workflow loads CHGNet checkpoint **0.3.0**, predicts energy, forces, stress, and magnetic moments for the official eight-atom LiMnO2 example, and validates both numerical outputs and physical symmetry. The larger held-out-data evaluation is semester work and is not necessary for this inference assignment.

Model-selection tracking issue: https://github.com/cyber-shuttle/cybershuttle-ai4science-foundry/issues/21

## 1. Required resources

Use the course's authorized AI Makerspace allocation. Confirm the partition, account, QoS, module availability, and GPU type; these are deliberately not guessed. Request **1 GPU, 4 CPU cores, 16 GB RAM, 30 minutes** for inference. The input is small; a GPU with 16 GB VRAM is a conservative request, to be checked against measured use. Allow **10 GB storage** for environment/source/checkpoints, and roughly **1-5 GB transfers** for software. No full MPtrj dataset download is needed. Environment creation may need a longer initial compute session than inference.

Python **3.10.13** and the module `anaconda3/2023.03` were used for the historical ICE baseline. Confirm that module in Makerspace with `module avail`; use an equivalent Python 3.10 installation if necessary and document it. CPU validation is supported, but only the course can confirm whether CPU meets the Makerspace requirement.

## 2. Setup

Clone this project repository to persistent course storage, then change to its root. Commands below assume a Linux compute session, outbound package access, Git, and Python 3.10. Run installations and inference within allocated compute resources.

```bash
module load anaconda3/2023.03
mkdir -p runs
TORCH_BACKEND=cu121 bash scripts/setup.sh 2>&1 | tee runs/setup.log
```

The setup script checks out the exact source revision in `config/workflow.json`, installs PyTorch 2.5.1 from its official CUDA 12.1 wheel index, pins the known NumPy/pymatgen baseline versions, installs CHGNet from the checkout, checks dependencies, and saves artifact hashes and a full environment lock. CUDA wheels include their runtime; the allocated GPU's driver must support it. For a CPU-only diagnostic environment use `TORCH_BACKEND=cpu` in a **separate checkout**.

The first setup used the recovered September dependency lock with CUDA PyTorch; its complete new environment and portable dependency locks are archived with the successful run. For an independent rerun from a fresh checkout, install the archived dependency lock:

```bash
TORCH_BACKEND=cu121 DEPENDENCY_LOCK=evidence/6199752/dependencies-lock.txt bash scripts/setup.sh
```

Use the committed successful run directory `6199752`. This portable lock omits torch and the local-path CHGNet entry; setup installs those explicitly from the official wheel index and pinned source. The dependency lock records exact transitive versions; the setup log records their successful installation in the fresh GPU environment.

## 3. Run inference

In an already allocated GPU session:

```bash
DEVICE=cuda bash scripts/run.sh
```

Or submit a scheduled job from the project root after filling your authorized values:

```bash
export MAKERSPACE_PARTITION='your-course-partition'
export MAKERSPACE_ACCOUNT='your-course-account'
# Optional, if required: export MAKERSPACE_QOS='your-course-qos'
bash scripts/submit.sh
```

`inference.sbatch` supplies the resource requests and explicitly requests CUDA. A missing GPU causes a failure rather than silent CPU fallback. A CPU diagnostic run can be started in a CPU compute session with `DEVICE=cpu bash scripts/run.sh`. Seed is 42; thread count is 4; inference uses the unchanged supplied CIF and explicit checkpoint 0.3.0. Outputs go to `runs/JOB_ID/` (or a UTC-stamped manual ID). No training or fine-tuning is performed.

## 4. Validate

The script exits with code 0 only if all checks pass. `results.json` contains expected/observed outputs, tolerances, maximum errors, hashes, versions, UTC time, hardware, runtime, memory, and job context.

- **Smoke test:** finite outputs with energy scalar, forces 8x3, stress 3x3, and eight magnetic moments.
- **Numerical validation:** compare all four outputs with the official `test_predict_structure` reference. Energy is approximately **-7.36769 eV/atom**; the Mn moments are approximately **3.86942 mu_B**. Relative tolerances are energy 1e-4, forces 1e-3, stress 5e-3, moments 1e-3; absolute tolerance is 1e-4 in each output's units. The comparison uses the maximum of absolute/relative tolerances, matching `pytest.approx` rather than summing them.
- **Physical validation:** rotate coordinates and lattice together by 30, 60, and 120 degrees about [1,2,3]. Energy/moments should be invariant; forces and stress should rotate appropriately. Translation and reordered-site checks test invariance and consistent site mapping. Absolute tolerances are e/m/f 1e-4 and stress 1e-3 in their respective units; they are fixed before execution.

These validate implementation consistency for one supplied input; agreement with the model's reference output is not an independent test of DFT accuracy or generalization. All five transformations passed in the H100 run. Failed checks must be investigated and preserved; do not silently widen tolerances.

```bash
source .venv/bin/activate
python -m unittest discover -s tests -v
python -c 'import json; r=json.load(open("evidence/6199752/results.json")); print(r["passed"], r["reference_checks"], r["symmetry_checks"])'
```

CUDA timing synchronizes the device. GPU memory reports PyTorch peaks, not all driver/editor memory. Process RSS excludes other processes. `inference_seconds` excludes model loading; `load_inference_validation_seconds` includes loading and the complete validation suite, but excludes imports and package installation.

## 5. Preserve execution evidence and submit

After the job finishes:

```bash
sacct -j JOB_ID --format=JobID,State,ExitCode,Elapsed,AllocCPUS,ReqMem,MaxRSS
python scripts/archive_evidence.py RUN_ID
```

Save the `sacct` output and scheduler stdout/stderr beside the archived run. Keep `runs/setup.log` as installation evidence. The archive includes the raw log, results, environment lock, hashes, resource description, and exit status. Keep failed run evidence too. Review logs for unnecessary private paths before publishing; do not commit credentials, environment-variable dumps, GPU weights, `.venv`, or the entire upstream checkout.

The course announcement additionally requires an **Assignment 2 pull request**. Include a hyperlink to [model-tracking issue #21](https://github.com/cyber-shuttle/cybershuttle-ai4science-foundry/issues/21) in the PR description, then update that issue's description with the actual PR hyperlink and relevant run information. Keep the issue open for the semester project; use 'Related to' rather than an automatic closing keyword. Submit the project repository link as requested by the assignment and include the PR link in the submission if the course form allows it. The real logs/results and portable lock are now archived; a second independent checkout rerun is still recommended. Before submission, create and verify the required issue/PR cross-links. See `docs/submission-checklist.md` for the rubric audit.

## 6. Troubleshooting and cleanup

The September ICE CPU run initially used PyPI CHGNet 0.4.0: energy/moments passed but forces/stress failed. The source revision recorded here (reported package 0.4.1) passed all four without changing tolerances. Preserve this distinction between software package version and checkpoint 0.3.0. The recovered original logs are archived in `evidence/historical/`.

If source checkout fails, verify the recorded commit against the old ICE checkout; do not replace it with floating `main`. If compiled graph conversion fails, record the build/compiler errors. If CUDA is missing, check `nvidia-smi`, the GPU request, and torch's installed wheel; do not reinstall inside a running inference job. For numerical failures compare versions, hashes, site order, and device. Contact the TA for unauthorized partition/account errors.

Cancel only your unused job with `scancel JOB_ID`, or stop its OnDemand session. Preserve the tracked evidence. You can archive local generated environments/source/runs by moving `.venv`, `artifacts`, and `runs` to a backup directory; no cleanup script deletes them automatically. Persistent project files remain after compute sessions expire.

## Sources and licenses

- Paper: Deng et al. (2023), https://doi.org/10.1038/s42256-023-00716-3
- Official code, checkpoint, example, and reference: https://github.com/CederGroupHub/chgnet/tree/b9a6c15860237297a5a5169660e83b2e61a78e76
- Artifact source/identity: `config/workflow.json`; downloaded artifact hashes: `artifacts/manifest.json`, copied into execution evidence.
- Code license: upstream `LICENSE` (BSD-style three-clause with enhancements provision). Retain it with any redistributed upstream artifacts. No separate checkpoint license was identified in its README. The tiny example is distributed with the official repository.
- Reference values in `config/reference.json` are transcribed from the official test cited there, using the September reproduction script. No independently downloaded DFT labels are used here. Validate the pinned test file against this transcription during the fresh run; its checksum is archived.
