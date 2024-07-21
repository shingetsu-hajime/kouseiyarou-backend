# 校正野郎
校正用ウェブアプリのバックエンド

512MBを超えないように注意

モデルはgitlfs

## 開発コマンド

バックエンド: FastAPI

`cd backend`

▼開発時

`uvicorn main:app --reload`

▼本番時

`uvicorn main:app --host=0.0.0.0 --port=8000`


# モデル情報

データセット
https://nlp.ist.i.kyoto-u.ac.jp/?%E6%97%A5%E6%9C%AC%E8%AA%9EWikipedia%E5%85%A5%E5%8A%9B%E8%AA%A4%E3%82%8A%E3%83%87%E3%83%BC%E3%82%BF%E3%82%BB%E3%83%83%E3%83%88

よりdataにインストールして解凍する。

`tar -xzvf jwtd_v2.0.tar.gz`


## 量子化モデル
量子化前の精度　0.67
量子化後の精度　0.64
