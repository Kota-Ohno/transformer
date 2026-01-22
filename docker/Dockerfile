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

# アプリケーションコードをコピー
COPY app/ .

# エントリーポイントスクリプトをコピー（app/の後に実行して上書きを防ぐ）
COPY entrypoint.sh /src/entrypoint.sh

# 非rootユーザーの作成
RUN groupadd -r transformer && \
    useradd -r -m -g transformer -u 1000 transformer && \
    chown -R transformer:transformer /src && \
    chown -R transformer:transformer /home/transformer && \
    chmod +x /src/entrypoint.sh

# 環境変数の設定
ENV PYTHONUNBUFFERED=1
# デバイス設定は entrypoint.sh で CUDA 可用性を検出して自動設定されます
# ランタイムでオーバーライドする場合: docker run -e TRANSFORMER_GLOBAL_DEVICE=cpu

# 非rootユーザーに切り替え
USER transformer

# アプリケーションがリッスンするポートを公開
EXPOSE 6000

# アプリケーションのエントリーポイント（CUDA可用性を自動検出）
ENTRYPOINT ["/src/entrypoint.sh"]
