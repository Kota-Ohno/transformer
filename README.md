# prototype-transformer

.envのWANDB_API_KEYを設定する。

```
docker compose up -d
docker exec -it transformer /bin/sh
```

```
python text_tokenizer.py
python train.py
python predict.py
```
