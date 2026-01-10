#!/bin/bash
# entrypoint.sh
# CUDA可用性をチェックし、利用できない場合はCPUにフォールバックするエントリーポイントスクリプト

set -e

# CUDA可用性のチェック
if ! python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo "Warning: CUDA is not available. Falling back to CPU."
    export TRANSFORMER_GLOBAL_DEVICE=cpu
    export TRANSFORMER_DEVICE=cpu
else
    # CUDAが利用可能な場合でも、実際にデバイスにアクセスできるか確認
    if ! python -c "import torch; torch.cuda.get_device_properties(0)" 2>/dev/null; then
        echo "Warning: CUDA device access failed. Falling back to CPU."
        export TRANSFORMER_GLOBAL_DEVICE=cpu
        export TRANSFORMER_DEVICE=cpu
    else
        echo "CUDA is available. Using GPU."
    fi
fi

# アプリケーションを実行
exec python -m core.main "$@"
