# Worklog: RTX PRO 4000 Blackwell CUDA MCMC Reconstruction

## 2026-06-15 Initial Branch And Environment Record

### Struggles
- Initial pixi environment resolved to torch 2.2.2 + CUDA 11.8, which imports but warns that NVIDIA RTX PRO 4000 Blackwell sm_120 is not compatible with the installed PyTorch build.
- Applying upstream PR 3653 before upgrading torch made `ns-train` fail under torch 2.2.2 because `torch.amp.custom_bwd/custom_fwd` is unavailable there.
- `pixi.lock` was rewritten by pixi 0.70.2 during install; it is currently treated as environment/build side effect and not included in the worklog setup commit.
- A cu128 pip install was started to replace torch/torchvision in the pixi env with torch 2.8.0+cu128 / torchvision 0.23.0+cu128. It may still be running and must be verified before training.

### Findings
- Repo cloned at `/home/kasm-user/Desktop/nerfstudio` from `git@github.com:yuki-inaho/nerfstudio.git` (remote URL in clone is GitHub HTTPS for origin after clone resolution).
- Base commit before local work: `50e0e3c70c775e89333256213363badbf074f29d`.
- Local compatibility commit exists: `c91dc7fd Apply torch compatibility patches for Blackwell`.
- GPU measured by `nvidia-smi`: NVIDIA RTX PRO 4000 Blackwell, driver 580.159.04, driver CUDA 13.0, 24467 MiB VRAM.
- `nvcc --version` measured CUDA compilation tools 11.8 V11.8.89 from the pixi environment/current PATH.
- Torch before cu128 upgrade: torch 2.2.2, torch CUDA 11.8, CUDA available true, device capability `(12, 0)`, with PyTorch sm_120 incompatibility warning.
- Dataset trajectory evidence so far favors `/home/kasm-user/Desktop/TVA_NYX650_2026_06_04_colmap_0501/gluemap_vggt_result/gluemap_aba` as the straight trajectory candidate: PCA primary variance 0.9999635, RMS lateral/extent 0.0017238, path/axis extent 1.0318, monotonic positive steps 499/500, 501 registered images, 49997 points.
- ALIKED sequential candidate is also mostly one-dimensional but has more frame-order backtracking: primary variance 0.9998414, RMS lateral/extent 0.0036393, path/axis extent 1.6951, positive steps 445/500, marked not monotonic in its analysis JSON.

### Tips
- Keep raw dataset immutable. Use `/home/kasm-user/Desktop/nerfstudio_processed` for processed nerfstudio inputs and `/home/kasm-user/Desktop/nerfstudio_outputs` for training outputs.
- Do not include `.pixi/`, `outputs/`, generated datasets, or large logs in commits.
- If torch cu128 install succeeds, rebuild/reinstall tiny-cuda-nn against the cu128 torch before training.
- If pixi reverts the pip-installed torch later, rerun the cu128 install step or encode it as a pixi task/manifest change in a separate atomic commit.

### Commands / evidence
- `git clone git@github.com:yuki-inaho/nerfstudio.git /home/kasm-user/Desktop/nerfstudio` succeeded.
- `PATH=/home/kasm-user/.pixi/bin:$PATH pixi run post-install` succeeded; log: `/home/kasm-user/Desktop/nerfstudio_task_logs/pixi_post_install_20260615T0329Z.log`.
- `gh pr view 3653 --repo nerfstudio-project/nerfstudio ...` showed PR 3653 open, files: trainer.py, activations.py, download_data.py, eval_utils.py.
- `gh pr view 3711 --repo nerfstudio-project/nerfstudio ...` showed PR 3711 open, files: splatfacto.py, eval_utils.py.
- PR patch logs: `/home/kasm-user/Desktop/nerfstudio_task_logs/pr3653.patch`, `/home/kasm-user/Desktop/nerfstudio_task_logs/pr3711.patch`.
- `git switch -c work/rtx-pro-4000-blackwell-driver-cuda-13-0-nvcc-11-8-mcmc-recon` created the GPU/CUDA-specific work branch.

### Decisions
- Branch name chosen from measured GPU and CUDA values: `work/rtx-pro-4000-blackwell-driver-cuda-13-0-nvcc-11-8-mcmc-recon`.
- PR 3653 was applied because the GPU is Blackwell and the current torch 2.2.2/CUDA 11.8 stack emitted an sm_120 incompatibility warning; however, the PR requires newer torch APIs, so torch cu128 upgrade is required before training.
- PR 3711 was partially overlapping with PR 3653; the duplicate `eval_utils.py` change was already covered, and the unique `splatfacto.get_viewmat` `@torch_compile()` removal was applied.
- Worklog is placed under `temp/` to match the user request. `.gitignore` ignores `temp/*` but explicitly allows `temp/worklog_*.md` so only the worklog is committed.
recorded_at: Mon Jun 15 03:43:03 UTC 2026

## 2026-06-15 cu128 Torch Environment Repair

### Struggles
- Installing `torch==2.8.0+cu128` and `torchvision==0.23.0+cu128` via pip initially upgraded numpy to `2.2.6`, breaking `pyarrow 15.0.2` and `ns-train` imports with numpy ABI errors.
- Existing `tinycudann` was built against torch 2.2/CUDA 11.8 and failed under torch 2.8 with an undefined symbol.
- Rebuilding `tinycudann` with `TCNN_CUDA_ARCHITECTURES=90` failed because PyTorch was compiled with CUDA 12.8 while available `nvcc` is CUDA 11.8.

### Findings
- After repair, environment check passed with `torch 2.8.0+cu128`, torch runtime CUDA `12.8`, CUDA available true, GPU `NVIDIA RTX PRO 4000 Blackwell`, capability `(12, 0)`.
- CUDA tensor matmul succeeded under torch 2.8.0+cu128.
- `numpy==1.26.4` and `tzdata` restored dependency consistency; `pip check` reports no broken requirements.
- `ns-train splatfacto --help` succeeds.
- `tinycudann` is intentionally unavailable after uninstall; this is acceptable for the assigned `splatfacto`/mcmc path but not for nerfacto/tcnn-dependent paths.

### Tips
- Do not rerun plain `pixi run post-install` without reapplying the cu128 pip overlay; pixi manifest still declares conda torch 2.2/CUDA 11.8.
- If tiny-cuda-nn is needed later, install a CUDA 12.8 toolkit/nvcc in the pixi environment first, then rebuild it against torch 2.8.

### Commands / evidence
- `pixi run python -m pip install --upgrade --force-reinstall torch==2.8.0+cu128 torchvision==0.23.0+cu128 --index-url https://download.pytorch.org/whl/cu128`
- `pixi run python -m pip install --force-reinstall numpy==1.26.4 tzdata`
- `TCNN_CUDA_ARCHITECTURES=90 pixi run python -m pip install --force-reinstall --no-cache-dir --no-build-isolation ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch` failed with CUDA mismatch: detected 11.8 vs torch 12.8.
- `pixi run python -m pip uninstall -y tinycudann`
- `pixi run python -m pip check` -> no broken requirements.
- Logs: `/home/kasm-user/Desktop/nerfstudio_task_logs/pip_torch_cu128_20260615T0340Z.log`, `/home/kasm-user/Desktop/nerfstudio_task_logs/pip_fix_numpy_tzdata_20260615T0344Z.log`, `/home/kasm-user/Desktop/nerfstudio_task_logs/pip_rebuild_tinycudann_torch28_20260615T0345Z.log`, `/home/kasm-user/Desktop/nerfstudio_task_logs/pip_uninstall_tinycudann_20260615T0346Z.log`.

### Decisions
- Continue reconstruction with `splatfacto --pipeline.model.strategy mcmc`, because this path imports and its CLI works without tiny-cuda-nn.
- Leave pixi manifest/lock uncommitted for now; the working environment is a pip overlay on top of pixi, and a cleaner manifest-level CUDA 12.8 conversion would be a separate dependency migration.
