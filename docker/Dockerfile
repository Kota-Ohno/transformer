FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# CUDA 12.xサポート付きでspaCyをインストール
# 注: 実際の動作環境でCUDA 12.xサポートが適切に機能するか確認が必要です
RUN pip install -U pip==23.1.2 setuptools==68.0.0 wheel==0.40.0 && \
    # NLP関連パッケージ
    pip install -U 'spacy[cuda12x]==3.6.1' && \
    python -m spacy download en_core_web_md && \
    python -m spacy download ja_core_news_md && \
    pip install sentencepiece==0.1.99 && \
    pip install sacrebleu==2.3.1 && \
    pip install nltk==3.8.1 && \
    # 機械学習関連パッケージ
    pip install scikit_learn==1.3.0 && \
    # データハンドリング関連
    pip install pandas==2.0.3 && \
    pip install datasets==2.14.5 && \
    # 可視化・ロギング関連
    pip install wandb==0.15.5 && \
    pip install tensorboard==2.13.0 && \
    # エラー処理強化
    pip install better-exceptions==0.3.3

# 作業ディレクトリの設定
WORKDIR /src

# 環境変数の設定
ENV PYTHONUNBUFFERED=1
ENV TRANSFORMER_DEVICE=cuda
