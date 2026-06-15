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

## 2026-06-15 Resume Preflight And gsplat Blocker

### Struggles
- `gsplat` JIT compilation fails with the repo/pixi `nvcc` 11.8 when torch detects Blackwell `sm_120`; direct compile uses `compute_120` and fails because CUDA 11.8 does not know that architecture.
- `TORCH_CUDA_ARCH_LIST=9.0+PTX` avoids the unsupported `compute_120` flag, but CUDA 11.8 `nvcc` segfaulted while compiling `fully_fused_projection_packed_2dgs_fwd.cu`.
- A trial `pip install --force-reinstall gsplat==1.5.3` pulled torch `2.12.0+cu130`, numpy `2.2.6`, rich `15.0.0`, and CUDA 13 runtime packages, creating dependency conflicts with nerfstudio, pyarrow, torchvision, and viser.

### Findings
- Resume preflight at `Mon Jun 15 04:01:53 UTC 2026`: repo `/home/kasm-user/Desktop/nerfstudio`, branch `work/rtx-pro-4000-blackwell-driver-cuda-13-0-nvcc-11-8-mcmc-recon`, dataset path exists.
- GPU still idle and available: `NVIDIA RTX PRO 4000 Blackwell`, driver CUDA `13.0`, 24467 MiB VRAM.
- No training process is running after smoke failures.
- After the `gsplat==1.5.3` trial, `pip check` reports conflicts: `pyarrow` requires numpy `<2`, and `torchvision 0.23.0+cu128` requires torch `2.8.0`.

### Tips
- Do not install `gsplat==1.5.3` without constraining dependencies; it tries to upgrade torch and CUDA runtime outside the current nerfstudio compatibility envelope.
- For this repo revision, `gsplat==1.4.0` is the declared dependency. The key blocker is not Python API import but CUDA extension compilation with a Blackwell-capable compiler.

### Commands / evidence
- `TORCH_CUDA_ARCH_LIST=9.0+PTX ... ns-train splatfacto ...` failed with CUDA 11.8 `nvcc` segfault while compiling a gsplat CUDA source. Log: `/home/kasm-user/Desktop/nerfstudio_task_logs/ns_train_smoke_splatfacto_mcmc_sm90ptx_fg_20260615T0402Z.log`.
- `pip install --upgrade --force-reinstall gsplat==1.5.3` completed but broke dependency consistency. Log: `/home/kasm-user/Desktop/nerfstudio_task_logs/pip_install_gsplat_153_20260615T0405Z.log`.

### Decisions
- Revert the Python environment to the known torch `2.8.0+cu128`, numpy `1.26.4`, pillow `10.3.0`, rich `<15`, and repo-pinned `gsplat==1.4.0` baseline before further training attempts.
- Next viable fix should provide a CUDA 12.8+ `nvcc` for gsplat JIT, or use a constrained prebuilt gsplat wheel that matches torch/cu128 and does not alter unrelated dependencies.

## 2026-06-15 Direct Pixi Build Verification

recorded_at: 2026-06-15 04:17:16 UTC

### Struggles
- The prior worker had a functional `pixi run` Python overlay, but `pixi.toml` still declared conda torch 2.2/CUDA 11.8 dependencies.
- `ns-train splatfacto --help` works, but smoke training reaches `gsplat` JIT and then requires a Blackwell-capable CUDA compiler.
- The installed PyPI `nvidia-cuda-nvcc-cu12` package provides CUDA 12.8 headers/PTX tools, but the `nvcc` on `PATH` is still the pixi conda CUDA 11.8 compiler.

### Findings
- Active branch: `work/rtx-pro-4000-blackwell-driver-cuda-13-0-nvcc-11-8-mcmc-recon`.
- Current runtime check passes: `torch 2.8.0+cu128`, torch CUDA `12.8`, CUDA available true, device `NVIDIA RTX PRO 4000 Blackwell`, capability `(12, 0)`, CUDA matmul OK.
- `pixi run python -m pip check` reports no broken requirements.
- `pixi run ns-train splatfacto --help` succeeds.
- No related `pixi`, `ns-train`, `nvcc`, or `cudafe` processes were running after closing the subagent.

### Tips
- Keep the conda CUDA compiler version aligned with the PyTorch CUDA runtime before retrying `gsplat` JIT.
- Avoid unconstrained `gsplat` upgrades; `gsplat==1.5.3` previously pulled torch/CUDA packages outside the repo compatibility envelope.

### Commands / evidence
- `pixi info` showed pixi `0.70.2`, virtual CUDA `13.0`, and the repo default environment at `.pixi/envs/default`.
- `pixi run python - <<'PY' ...` verified torch/cu128, nerfstudio import, gsplat `1.4.0` import, and tinycudann absence.
- `pixi run python -m pip check` -> no broken requirements.
- `pixi run ns-train splatfacto --help` -> generated a 543-line help page successfully.
- `pixi run nvcc --version` -> CUDA compilation tools `11.8`, which is too old for `sm_120`.

### Decisions
- Convert pixi metadata toward the actual working Blackwell stack: PyPI torch/torchvision cu128 plus conda CUDA 12.8 compiler/runtime packages.
- Treat tiny-cuda-nn as optional for this splatfacto/MCMC path because `tinycudann` is not required for the selected method and cannot be rebuilt until the compiler mismatch is resolved.

## 2026-06-15 Pixi CUDA 12.8 Build Completed

recorded_at: 2026-06-15T05:40:02Z

### Struggles
- Moving torch/torchvision into the pixi manifest fixed the pip-overlay drift, but `pixi run post-install` initially failed because newer setuptools removed the `pkg_resources` import path used by the current repo.
- Rebuilding `tinycudann` against torch `2.8.0+cu128` exposed missing CUDA development headers in the conda environment: first `nvrtc.h`/`cusparse.h`, then `cusolverDn.h`.
- `gsplat==1.4.0` JIT for Blackwell `sm_120` failed under the default PyTorch extension parallelism with `ninja -j10` / `nvcc` exit 139.
- After CUDA `.cu` objects compiled, `gsplat` failed on `ext.cpp` because PyTorch's JIT compile command ignored conda `CXXFLAGS` and could not find `cuda_runtime_api.h`.

### Findings
- `pixi.toml` now encodes the Blackwell-compatible stack directly: CUDA 12.8 conda compiler/dev packages plus PyPI `torch==2.8.0+cu128` and `torchvision==0.23.0+cu128`.
- `setuptools==69.5.1`, `numpy==1.26.4`, `pillow==10.3.0`, and `rich==13.7.1` are pinned to avoid known breakages in this repo revision.
- `scripts/pixi-cuda-env.sh` adds the CUDA target include directory to `CPATH`, sets `TORCH_CUDA_ARCH_LIST=12.0+PTX`, sets `TCNN_CUDA_ARCHITECTURES=120`, and defaults `MAX_JOBS=1`.
- With `MAX_JOBS=1`, `gsplat_cuda.so` built successfully for `compute_120` + `sm_120`.
- Final import checks pass: torch `2.8.0+cu128`, torch runtime CUDA `12.8`, GPU capability `(12, 0)`, `gsplat 1.4.0` backend loaded, `tinycudann` import OK.
- Final 2-iteration `splatfacto` MCMC smoke completed successfully on `/home/kasm-user/Desktop/nerfstudio_processed/TVA_NYX650_2026_06_04_gluemap_aba_straight`.

### Tips
- Keep `MAX_JOBS=1` for PyTorch CUDA extension builds on this machine unless a newer compiler/gsplat combination is proven stable; it avoids the observed `nvcc` exit 139.
- `CPATH` is required because PyTorch's generated `build.ninja` does not include conda CUDA target headers for pure C++ extension sources.
- Use `12.0+PTX` rather than plain `12.0` for `TORCH_CUDA_ARCH_LIST` so PyTorch emits both PTX and native Blackwell code.
- Do not remove `scripts/pixi-cuda-env.sh` unless `pixi run python -c "import gsplat.cuda._backend"` and `ns-train splatfacto` are retested from a clean extension cache.

### Commands / evidence
- `pixi install` completed after manifest conversion to CUDA 12.8 / torch cu128.
- `pixi run post-install` succeeded after header and setuptools fixes. Final log: `/home/kasm-user/Desktop/nerfstudio_task_logs/pixi_post_install_final_20260615T043918Z.log`.
- `pixi run python -m pip check` -> no broken requirements.
- `pixi run nvcc --version` -> CUDA compilation tools `12.8`, V12.8.61.
- `pixi run python` import check loaded torch/cu128, `gsplat.cuda._backend`, and `tinycudann`.
- `pixi run --manifest-path /home/kasm-user/Desktop/nerfstudio/pixi.toml ninja -v -j 1` completed `gsplat_cuda.so`. Log: `/home/kasm-user/Desktop/nerfstudio_task_logs/gsplat_cuda_full_j1_cpath_20260615T052731Z.log`.
- `pixi run ns-train splatfacto --pipeline.model.strategy mcmc --max-num-iterations 2 ...` completed successfully. Log: `/home/kasm-user/Desktop/nerfstudio_task_logs/ns_train_smoke_splatfacto_mcmc_cuda128_20260615T053929Z.log`.

### Decisions
- Keep `gsplat==1.4.0` as declared by the repo instead of upgrading to an unconstrained newer package that changes torch/CUDA/numpy dependency surfaces.
- Commit the pixi manifest, lockfile, activation script, and worklog together as the reproducible pixi build fix for this GPU/CUDA branch.

## 2026-06-15 20k MCMC Training And Quality Eval

recorded_at: 2026-06-15T06:10:07Z

### Struggles
- The first full-quality request needed a new push decision because the earlier instruction prohibited push, while the later instruction explicitly asked for commit and push on the Blackwell branch.
- `ns-eval` output capture used the pixi Python environment for JSON formatting because the base shell did not have a `python` command on PATH.
- The pixi manifest still emits a deprecation warning for `[project]`; this is non-blocking for install, train, and eval.

### Findings
- Active branch is `work/rtx-pro-4000-blackwell-driver-cuda-13-0-nvcc-11-8-mcmc-recon`, tracking `origin/work/rtx-pro-4000-blackwell-driver-cuda-13-0-nvcc-11-8-mcmc-recon`.
- 20k `splatfacto` MCMC training completed successfully on `/home/kasm-user/Desktop/nerfstudio_processed/TVA_NYX650_2026_06_04_gluemap_aba_straight`.
- Final checkpoint is `/home/kasm-user/Desktop/nerfstudio_outputs/TVA_NYX650_2026_06_04_gluemap_aba_mcmc_20k/splatfacto/20260615T055229Z/nerfstudio_models/step-000019999.ckpt`.
- `ns-eval` loaded the final checkpoint and saved quality metrics to `/home/kasm-user/Desktop/nerfstudio_task_logs/ns_eval_splatfacto_mcmc_20k_cuda128_20260615T055229Z.json`.
- Eval metrics: PSNR `21.971933364868164`, PSNR std `0.837710440158844`, SSIM `0.7347635626792908`, SSIM std `0.04798934608697891`, LPIPS `0.22709527611732483`, LPIPS std `0.03481098264455795`, throughput `23850562.0` rays/sec, FPS `49.68867111206055`.

### Tips
- Use the saved `config.yml` for evaluation instead of manually selecting checkpoints; `ns-eval` loads the latest checkpoint from the run directory.
- Keep logs outside the repo tree under `/home/kasm-user/Desktop/nerfstudio_task_logs`; only the worklog under `temp/worklog_*.md` is intentionally tracked.
- For comparable metrics, keep the same processed straight-trajectory dataset and the same `splatfacto --pipeline.model.strategy mcmc` configuration.

### Commands / evidence
- Training command: `pixi run ns-train splatfacto --output-dir /home/kasm-user/Desktop/nerfstudio_outputs --experiment-name TVA_NYX650_2026_06_04_gluemap_aba_mcmc_20k --timestamp 20260615T055229Z --vis tensorboard --max-num-iterations 20000 --steps-per-save 5000 --steps-per-eval-image 5000 --steps-per-eval-all-images 10000 --pipeline.datamanager.cache-images cpu --pipeline.model.strategy mcmc --data /home/kasm-user/Desktop/nerfstudio_processed/TVA_NYX650_2026_06_04_gluemap_aba_straight`
- Training log: `/home/kasm-user/Desktop/nerfstudio_task_logs/ns_train_splatfacto_mcmc_20k_cuda128_20260615T055229Z.log`.
- Training reached `19999 (100.00%)` and printed `Training Finished`.
- Eval command: `pixi run ns-eval --load-config /home/kasm-user/Desktop/nerfstudio_outputs/TVA_NYX650_2026_06_04_gluemap_aba_mcmc_20k/splatfacto/20260615T055229Z/config.yml --output-path /home/kasm-user/Desktop/nerfstudio_task_logs/ns_eval_splatfacto_mcmc_20k_cuda128_20260615T055229Z.json`.
- Eval log: `/home/kasm-user/Desktop/nerfstudio_task_logs/ns_eval_splatfacto_mcmc_20k_cuda128_20260615T055229Z.log`.
- `pixi run python -m json.tool /home/kasm-user/Desktop/nerfstudio_task_logs/ns_eval_splatfacto_mcmc_20k_cuda128_20260615T055229Z.json` confirmed the saved metrics.

### Decisions
- Do not commit the model checkpoint, outputs, or raw logs; they are large runtime artifacts and are referenced from the tracked worklog.
- Commit and push this worklog update on the existing Blackwell-specific branch after the 20k training and eval evidence are recorded.
