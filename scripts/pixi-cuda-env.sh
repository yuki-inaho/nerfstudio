#!/usr/bin/env bash

if [ -n "${CONDA_PREFIX:-}" ]; then
    cuda_target_dir="${CONDA_PREFIX}/targets/x86_64-linux"
    if [ -d "${cuda_target_dir}/include" ]; then
        export CPATH="${cuda_target_dir}/include${CPATH:+:${CPATH}}"
    fi
fi

export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0+PTX}"
export TCNN_CUDA_ARCHITECTURES="${TCNN_CUDA_ARCHITECTURES:-120}"
export MAX_JOBS="${MAX_JOBS:-1}"
