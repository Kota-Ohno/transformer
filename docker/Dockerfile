FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# 作業ディレクトリの設定
WORKDIR /src

# requirements.txtをコピーしてライブラリをインストール
COPY app/requirements.txt .
COPY app/requirements-gpu.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -r requirements-gpu.txt && \
    python -m spacy download en_core_web_md && \
    python -m spacy download ja_core_news_md

# エントリーポイントスクリプトをコピー
COPY entrypoint.sh /src/entrypoint.sh

# アプリケーションコードをコピー
COPY app/ .

# 非rootユーザーの作成
RUN groupadd -r transformer && \
    useradd -r -g transformer -u 1000 transformer && \
    chown -R transformer:transformer /src && \
    chmod +x /src/entrypoint.sh

# 環境変数の設定
ENV PYTHONUNBUFFERED=1
# デバイス設定（デフォルトはcuda、実行時に上書き可能）
# ランタイムでオーバーライドする場合: docker run -e TRANSFORMER_GLOBAL_DEVICE=cpu
ENV TRANSFORMER_GLOBAL_DEVICE cuda
# 後方互換性のため、TRANSFORMER_DEVICEも設定
# ランタイムでオーバーライドする場合: docker run -e TRANSFORMER_DEVICE=cpu
ENV TRANSFORMER_DEVICE cuda

# 非rootユーザーに切り替え
USER transformer

# アプリケーションがリッスンするポートを公開
EXPOSE 6000

# アプリケーションのエントリーポイント（CUDA可用性を自動検出）
ENTRYPOINT ["/src/entrypoint.sh"]
