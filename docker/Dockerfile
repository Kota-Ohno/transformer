FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# 作業ディレクトリの設定
WORKDIR /src

# システムパッケージのインストール（SentencePieceビルドに必要）
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libprotobuf-dev \
    protobuf-compiler \
    && rm -rf /var/lib/apt/lists/*

# requirements.txtをコピーしてライブラリをインストール
COPY app/requirements.txt .
COPY app/requirements-gpu.txt .

# 依存ライブラリのインストール
# 注意: M1/M2/M4 MacではGPUが使用できないため、CPU版PyTorchが必要な場合がある
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -r requirements-gpu.txt || echo "GPU requirements skipped (optional)"

# spaCyモデルのダウンロード（オプション）
RUN python -m spacy download en_core_web_md || echo "en_core_web_md download skipped"
RUN python -m spacy download ja_core_news_md || echo "ja_core_news_md download skipped"

# アプリケーションコードをコピー
COPY app/ .

# エントリーポイントスクリプトをコピー
COPY entrypoint.sh /src/entrypoint.sh
RUN chmod +x /src/entrypoint.sh

# 非rootユーザーの作成
RUN groupadd -r transformer && \
    useradd -r -g transformer -u 1000 transformer && \
    chown -R transformer:transformer /src && \
    chown -R transformer:transformer /home/transformer

# 環境変数の設定
ENV PYTHONUNBUFFERED=1
ENV TRANSFORMER_GLOBAL_DEVICE=cpu

# 非rootユーザーに切り替え
USER transformer

# アプリケーションがリッスンするポートを公開
EXPOSE 6000

# エントリーポイント
ENTRYPOINT ["/src/entrypoint.sh"]
