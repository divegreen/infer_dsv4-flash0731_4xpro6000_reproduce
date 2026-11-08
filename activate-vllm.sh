# Run with: source the activate-vllm.sh

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    printf 'Use: source %s\n' "$0" >&2
    exit 1
fi

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/.venv-vllm/bin/activate" || return
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1,2,3
export UV_TORCH_BACKEND=cu130
