FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# 作業ディレクトリの設定
WORKDIR /src

# requirements.txtをコピーしてライブラリをインストール
COPY app/requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt && \
    python -m spacy download en_core_web_md && \
    python -m spacy download ja_core_news_md

# 環境変数の設定
ENV PYTHONUNBUFFERED=1
ENV TRANSFORMER_DEVICE=cuda

