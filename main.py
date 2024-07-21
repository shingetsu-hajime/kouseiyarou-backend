# coding: utf-8
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from analyzer import Analyzer, BertForMaskedLM_pl

import __main__

__main__.BertForMaskedLM_pl = BertForMaskedLM_pl

app = FastAPI()

analyzer = Analyzer()

# CORSを許可する設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 本番環境では、特定のオリジンを指定することを推奨
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextRequest(BaseModel):
    text: str

@app.get("/")
def read_root():
    return {"message": "Welcome to the Proofreader App"}

# 誤字脱字チェック
@app.post("/spell-check")
def spell_check(request: TextRequest):
    text = request.text
    result, diff_texts = analyzer.typo_text(text)
    return {"corrected_text": result,"count":len(diff_texts),"examples":diff_texts}

# 文末重複チェック
@app.post("/sentence-end-check")
def sentence_end_check(request: TextRequest):
    text = request.text
    result, diff_texts = analyzer.sentence_end_text(text)
    return {"corrected_text": result,"count":len(diff_texts),"examples":diff_texts}

# 単語近傍チェック
@app.post("/near-word-check")
def near_words(request: TextRequest):
    text = request.text
    result, diff_texts = analyzer.near_words(text)
    return {"corrected_text": result,"count":len(diff_texts),"examples":diff_texts}

# こそあど言葉チェック
@app.post("/kosoado-check")
def kosoado_check(request: TextRequest):
    text = request.text
    result, diff_texts = analyzer.kosoado(text)
    return {"corrected_text": result,"count":len(diff_texts),"examples":diff_texts}

# 正規化チェック
@app.post("/normalize-check")
def normalize_check(request: TextRequest):
    text = request.text
    result, diff_texts = analyzer.normalize(text)
    return {"corrected_text": result,"count":len(diff_texts),"examples":diff_texts}

# 分析
@app.get("/text-analysis")
def get_text_analysis():
    return {
        "statistics": [
            {"metric": "Total Words", "value": 1234},
            {"metric": "Total Errors", "value": 56},
            {"metric": "Grammar Errors", "value": 34},
            {"metric": "Spelling Errors", "value": 22},
        ]
    }


if __name__ == '__main__':
    print("初期化開始")
    analyzer = Analyzer()
    print("処理開始")
    result, _ = analyzer.typo_text('自衛隊が基地がある。吾輩の猫である')
    print(f"結果:{result}")