import io
import os
from contextlib import asynccontextmanager

import numpy as np
import torch
from torchvision import models, transforms
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from PIL import Image

# グローバルにモデルを保持するためのコンテナ
ml_models = {}

# カテゴリー判定用のキーワード定義
CATEGORY_MAPPING = {
    "人": ["person", "man", "woman", "child", "diver", "player", "groom", "scuba"],
    "動物": ["dog", "cat", "bird", "fish", "insect", "animal", "mammal", "snake", "spider", "horse", "bear", "elephant", "lion", "tiger"],
    "自動車": ["car", "automobile", "truck", "bus", "cab", "wagon", "jeep", "pickup", "racer"],
    "飛行機": ["plane", "airliner", "aircraft", "wing", "space_shuttle", "airship"],
}

def get_japanese_category(english_description: str):
    """
    英語のラベル名から指定された5つのカテゴリーのいずれかに分類します。
    """
    desc = english_description.lower()
    
    for jp_cat, keywords in CATEGORY_MAPPING.items():
        if any(keyword in desc for keyword in keywords):
            return jp_cat
    return "その他"

# 主要な物体の簡易日本語訳（例示的なサブセット）
TRANSLATIONS = {
    "Egyptian_cat": "エジプト猫",
    "sports_car": "スポーツカー",
    "airliner": "旅客機",
    "golden_retriever": "ゴールデンレトリバー",
    # 必要に応じて追加可能ですが、基本は元のラベルを整形して返します
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    アプリの起動時にモデルを1回だけロードし、メモリ効率を向上させます。
    """
    print("Loading AI model...")
    # 学習済み重みの指定とモデルのロード
    weights = models.MobileNet_V2_Weights.DEFAULT
    model = models.mobilenet_v2(weights=weights)
    model.eval()  # 推論モードに設定
    
    ml_models["model"] = model
    ml_models["preprocess"] = weights.transforms()
    ml_models["categories"] = weights.meta["categories"]
    yield
    # 終了処理が必要な場合はここに記述
    ml_models.clear()

app = FastAPI(lifespan=lifespan)

@app.get("/", response_class=HTMLResponse)
async def read_index():
    """メイン画面を返します。"""
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="index.html not found")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    アップロードされた画像を解析し、上位3つの予測結果を返します。
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="アップロードされたファイルは画像ではありません。")

    try:
        # 画像データの読み込み
        content = await file.read()
        img = Image.open(io.BytesIO(content)).convert('RGB')
        
        # 推論用の前処理（リサイズ、正規化等）
        preprocess = ml_models["preprocess"]
        input_tensor = preprocess(img).unsqueeze(0)
        
        # 推論実行
        with torch.no_grad():
            output = ml_models["model"](input_tensor)
        
        # ソフトマックスで確率に変換し、上位3つを取得
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        top3_prob, top3_indices = torch.topk(probabilities, 3)
        
        predictions = []
        for i in range(3):
            prob = top3_prob[i].item()
            desc = ml_models["categories"][top3_indices[i].item()]
            category = get_japanese_category(desc)
            jp_desc = TRANSLATIONS.get(desc, desc.replace('_', ' ')) # 辞書になければアンダースコアを除去
            predictions.append({"category": category, "description": jp_desc, "probability": float(prob)})

        return {
            "predictions": predictions
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"解析エラー: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    # 環境変数 PORT があればそれを使用し、なければ 9000 をデフォルトにする
    listen_port = int(os.environ.get("PORT", 9000))
    print(f"\n" + "="*50)
    print(f"AI画像解析サーバーを起動中... http://0.0.0.0:{listen_port}")
    print("="*50 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=listen_port)