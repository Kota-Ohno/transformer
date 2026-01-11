#!/bin/bash
# entrypoint.sh
# CUDA可用性をチェックし、利用できない場合はCPUにフォールバックするエントリーポイントスクリプト

set -e

# Pythonとtorchの可用性を確認
if ! python -c "import sys; import torch"; then
    echo "Error: Failed to import Python or torch. Please check your Python installation and dependencies." >&2
    exit 1
fi

# CUDA可用性のチェック
# 環境変数が既に設定されている場合はスキップ（ランタイムオーバーライドを許可）
if [ -z "$TRANSFORMER_GLOBAL_DEVICE" ] && [ -z "$TRANSFORMER_DEVICE" ]; then
    if ! python -c "import torch; exit(0 if torch.cuda.is_available() else 1)"; then
        echo "Warning: CUDA is not available. Falling back to CPU." >&2
        export TRANSFORMER_GLOBAL_DEVICE=cpu
        export TRANSFORMER_DEVICE=cpu
    else
        # CUDAが利用可能な場合でも、実際にデバイスにアクセスできるか確認
        # まずデバイス数が0より大きいことを確認
        if ! python -c "import torch; assert torch.cuda.device_count() > 0"; then
            echo "Warning: No CUDA devices found. Falling back to CPU." >&2
            export TRANSFORMER_GLOBAL_DEVICE=cpu
            export TRANSFORMER_DEVICE=cpu
        else
            # 最初の利用可能なデバイスをプローブ
            device_index=0
            if ! python -c "import torch; torch.cuda.get_device_properties($device_index)"; then
                echo "Warning: CUDA device access failed for device $device_index. Falling back to CPU." >&2
                export TRANSFORMER_GLOBAL_DEVICE=cpu
                export TRANSFORMER_DEVICE=cpu
            else
                echo "CUDA is available. Using GPU." >&2
                export TRANSFORMER_GLOBAL_DEVICE=cuda
                export TRANSFORMER_DEVICE=cuda
            fi
        fi
    fi
else
    echo "Device settings detected from environment variables: TRANSFORMER_GLOBAL_DEVICE=$TRANSFORMER_GLOBAL_DEVICE, TRANSFORMER_DEVICE=$TRANSFORMER_DEVICE" >&2
    # 一方だけが設定されている場合は、もう一方も同じ値に設定（後方互換性）
    if [ -n "$TRANSFORMER_GLOBAL_DEVICE" ] && [ -z "$TRANSFORMER_DEVICE" ]; then
        export TRANSFORMER_DEVICE="$TRANSFORMER_GLOBAL_DEVICE"
    elif [ -z "$TRANSFORMER_GLOBAL_DEVICE" ] && [ -n "$TRANSFORMER_DEVICE" ]; then
        export TRANSFORMER_GLOBAL_DEVICE="$TRANSFORMER_DEVICE"
    fi
fi

# アプリケーションを実行
exec python -m core.main "$@"
