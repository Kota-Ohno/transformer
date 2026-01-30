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
    # 設定されている環境変数のみを動的に構築して表示
    log_message="Device settings detected from environment variables:"
    if [ -n "$TRANSFORMER_GLOBAL_DEVICE" ]; then
        log_message="$log_message TRANSFORMER_GLOBAL_DEVICE=$TRANSFORMER_GLOBAL_DEVICE"
    fi
    if [ -n "$TRANSFORMER_DEVICE" ]; then
        log_message="$log_message TRANSFORMER_DEVICE=$TRANSFORMER_DEVICE"
    fi
    echo "$log_message" >&2
    # 両方が設定されていて異なる値の場合はエラー
    if [ -n "$TRANSFORMER_GLOBAL_DEVICE" ] && [ -n "$TRANSFORMER_DEVICE" ]; then
        if [ "$TRANSFORMER_GLOBAL_DEVICE" != "$TRANSFORMER_DEVICE" ]; then
            echo "ERROR: conflicting values for TRANSFORMER_GLOBAL_DEVICE and TRANSFORMER_DEVICE: $TRANSFORMER_GLOBAL_DEVICE vs $TRANSFORMER_DEVICE" >&2
            exit 1
        fi
    fi
    # 一方だけが設定されている場合は、もう一方も同じ値に設定（後方互換性）
    if [ -n "$TRANSFORMER_GLOBAL_DEVICE" ] && [ -z "$TRANSFORMER_DEVICE" ]; then
        export TRANSFORMER_DEVICE="$TRANSFORMER_GLOBAL_DEVICE"
    elif [ -z "$TRANSFORMER_GLOBAL_DEVICE" ] && [ -n "$TRANSFORMER_DEVICE" ]; then
        export TRANSFORMER_GLOBAL_DEVICE="$TRANSFORMER_DEVICE"
    fi

    # デバイス値の検証（後方互換性のコピー後に行う）
    validate_device_value() {
        local value="$1"
        local var_name="$2"
        if [ -n "$value" ]; then
            # "cpu"、"cuda"、または"cuda:N"（Nは非負の整数）の形式をチェック
            if ! echo "$value" | grep -qE '^(cpu|cuda(:[0-9]+)?)$'; then
                echo "ERROR: Invalid value for $var_name: '$value'. Allowed values are 'cpu', 'cuda', or 'cuda:N' (where N is a non-negative integer)." >&2
                exit 1
            fi
        fi
    }

    validate_device_value "$TRANSFORMER_GLOBAL_DEVICE" "TRANSFORMER_GLOBAL_DEVICE"
    validate_device_value "$TRANSFORMER_DEVICE" "TRANSFORMER_DEVICE"

    # CUDA可用性のランタイムチェック（構文検証後、後方互換性のコピー後に行う）
    check_cuda_availability() {
        local value="$1"
        local var_name="$2"
        if [ -n "$value" ] && echo "$value" | grep -qE '^cuda'; then
            # CUDAが利用可能かチェック
            if ! python -c 'import torch; import sys; sys.exit(0 if torch.cuda.is_available() else 1)'; then
                echo "ERROR: CUDA is not available, but $var_name is set to '$value'. Please set it to 'cpu' or ensure CUDA is properly configured." >&2
                exit 1
            fi

            # "cuda:N"の形式の場合、Nが利用可能なGPU数より小さいか確認
            if echo "$value" | grep -qE '^cuda:[0-9]+$'; then
                device_index=$(echo "$value" | sed 's/^cuda://')
                # python呼び出しのエラーハンドリング
                python_output=$(python -c 'import torch; print(torch.cuda.device_count())' 2>&1)
                python_exit_code=$?
                if [ $python_exit_code -ne 0 ]; then
                    echo "ERROR: Failed to get GPU count for $var_name. Python error: $python_output" >&2
                    echo "ERROR: var_name=$var_name, value=$value, device_index=$device_index" >&2
                    exit 1
                fi
                # python_outputが数値のみかどうかを検証
                if ! echo "$python_output" | grep -qE '^[0-9]+$'; then
                    echo "ERROR: Invalid GPU count output for $var_name. Expected numeric value, got: '$python_output'" >&2
                    echo "ERROR: var_name=$var_name, value=$value, device_index=$device_index" >&2
                    exit 1
                fi
                # device_indexが数値かどうかを検証
                if ! echo "$device_index" | grep -qE '^[0-9]+$'; then
                    echo "ERROR: Invalid device index for $var_name. Expected numeric value, got: '$device_index'" >&2
                    echo "ERROR: var_name=$var_name, value=$value" >&2
                    exit 1
                fi
                available_gpus=$python_output
                if [ "$available_gpus" -eq 0 ]; then
                    echo "ERROR: No GPUs available (available_gpus=0), but $var_name is set to '$value'. Please set it to 'cpu' or ensure GPUs are available." >&2
                    exit 1
                elif [ "$device_index" -ge "$available_gpus" ]; then
                    echo "ERROR: Invalid GPU index $device_index for $var_name. Only $available_gpus GPU(s) available (indices 0-$((available_gpus - 1)))." >&2
                    exit 1
                fi
            fi
        fi
    }

    check_cuda_availability "$TRANSFORMER_GLOBAL_DEVICE" "TRANSFORMER_GLOBAL_DEVICE"
    check_cuda_availability "$TRANSFORMER_DEVICE" "TRANSFORMER_DEVICE"
fi

# 引数に基づいて実行
if [ $# -eq 0 ]; then
    # 引数がない場合は対話型シェルを開始
    exec /bin/bash
else
    # 引数がある場合はそれを実行
    exec "$@"
fi
