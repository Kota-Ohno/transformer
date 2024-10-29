FROM pytorch/torchserve:latest-gpu

RUN pip install datasets && \
    pip install -U pip setuptools wheel && \
    pip install -U 'spacy[cuda11x]' && \
    pip install scikit_learn && \
    pip install wandb && \
    python -m spacy download en_core_web_md  && \
    python -m spacy download ja_core_news_md

WORKDIR /src