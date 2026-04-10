import io
import os
from contextlib import asynccontextmanager

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, decode_predictions, preprocess_input

# TensorFlowの最適化警告(oneDNN)などを抑制し、出力をクリーンにします
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

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
    ml_models["model"] = MobileNetV2(weights='imagenet')
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
        
        # MobileNetV2の入力サイズ(224x224)にリサイズ
        img = img.resize((224, 224))
        
        # 推論用の前処理
        x = tf.keras.preprocessing.image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        
        # 推論実行
        preds = ml_models["model"].predict(x)
        decoded_results = decode_predictions(preds, top=3)[0]
        
        predictions = []
        for _, desc, prob in decoded_results:
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
    # ポート番号を指定
    listen_port = 9000
    print(f"\n" + "="*50)
    print(f"AI画像解析サーバーを起動中... http://localhost:{listen_port}")
    print("="*50 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=listen_port)