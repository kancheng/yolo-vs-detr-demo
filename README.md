# YOLO & RF-DETR Demo

這是一個簡單的「同一張影像同時比較」Demo：分別用 **YOLO（ultralytics）** 與 **RF-DETR（rfdetr）** 做目標偵測/分割推論，並將結果輸出成圖片，方便快速觀察偵測框與分割效果的差異。

### 🇬🇧 English

This is a simple demo for **side-by-side comparison on the same image**.
It performs object detection and segmentation inference using **YOLO (Ultralytics)** and **RF-DETR (rfdetr)**, and exports the results as annotated images for quick visual inspection of differences in bounding boxes and segmentation quality.

### 🇯🇵 日本語

これは、**同一画像に対して並列比較を行うシンプルなデモ**です。
**YOLO（Ultralytics）** と **RF-DETR（rfdetr）** を用いて物体検出およびセグメンテーション推論を実行し、検出結果を画像として出力することで、バウンディングボックスや分割精度の違いを直感的に確認できます。

### 🇩🇪 Deutsch

Dies ist eine einfache Demo für einen **direkten Vergleich auf demselben Bild**.
Dabei werden Objekterkennung und Segmentierungs-Inference mit **YOLO (Ultralytics)** und **RF-DETR (rfdetr)** durchgeführt. Die Ergebnisse werden als annotierte Bilder ausgegeben, sodass Unterschiede bei Bounding-Boxen und Segmentierungsqualität schnell visuell bewertet werden können.

### 🇫🇷 Français

Il s’agit d’une démonstration simple permettant une **comparaison côte à côte sur la même image**.
L’inférence de détection d’objets et de segmentation est réalisée avec **YOLO (Ultralytics)** et **RF-DETR (rfdetr)**. Les résultats sont exportés sous forme d’images annotées afin d’observer rapidement les différences entre les boîtes englobantes et la qualité de segmentation.


## Demo

### RF-DETR

```
import requests
import supervision as sv
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
from rfdetr import RFDETRSegMedium
from rfdetr.assets.coco_classes import COCO_CLASSES

model = RFDETRSegMedium()

url = "https://ultralytics.com/images/bus.jpg"
response = requests.get(url, timeout=30)
response.raise_for_status()

image = Image.open(BytesIO(response.content)).convert("RGB")
detections = model.predict(image, threshold=0.5)

labels = [COCO_CLASSES[class_id] for class_id in detections.class_id]

mask_annotator = sv.MaskAnnotator()
label_annotator = sv.LabelAnnotator()

annotated_image = mask_annotator.annotate(image.copy(), detections)
annotated_image = label_annotator.annotate(annotated_image, detections, labels)

plt.figure(figsize=(12, 8))
plt.imshow(annotated_image)
plt.axis("off")
plt.show()

annotated_image.save("rfdetr_result.png")
print("Saved: rfdetr_result.png")
```

### YOLO

```
import requests
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
from ultralytics import YOLO

# Load model
model = YOLO("yolo26n-seg.pt")
# model = YOLO("path/to/best.pt")  # load a custom model

# Download image
url = "https://ultralytics.com/images/bus.jpg"
response = requests.get(url, timeout=30)
response.raise_for_status()
image = Image.open(BytesIO(response.content)).convert("RGB")

# Predict
results = model.predict(source=image)

# Show and save results
for i, result in enumerate(results):
    annotated_image = result.plot()

    plt.figure(figsize=(12, 8))
    plt.imshow(annotated_image)
    plt.axis("off")
    plt.show()

    Image.fromarray(annotated_image[..., ::-1]).save(f"yolo_result_{i}.png")
    print(f"Saved: yolo_result_{i}.png")

    if result.masks is not None:
        xy = result.masks.xy
        xyn = result.masks.xyn
        masks = result.masks.data

        print("Number of objects:", len(xy))
        print("Mask tensor shape:", masks.shape)
    else:
        print("No masks found.")
```

## 重點特性

- YOLO：使用 `yolo26n-seg.pt` 做影像推論，並把結果存成 `yolo_result_{i}.png`
- RF-DETR：使用 `RFDETRSegMedium()` 做影像推論（`threshold=0.5`），並把結果存成 `rfdetr_result.png`
- 兩者都會從同一張示範影像 `https://ultralytics.com/images/bus.jpg` 進行推論

## 環境需求

- Python（建議 3.8+）
- 套件（以 notebook 內容為準）

### 安裝

```bash
pip install rfdetr ultralytics
```

> 備註：`rfdetr` 依賴 PyTorch（`torch>=2.2.0`）；若你尚未安裝 PyTorch，請依你的 CPU/GPU 環境先安裝相容版本的 `torch`。

---

## 快速開始

### 用法 1：直接跑 Notebook（推薦）

1. 用 Jupyter / Colab 開啟 `YOLO_Und_RF_DETR_Demo.ipynb`
2. 依序執行各個 cell
3. 執行後會在工作目錄看到輸出圖片：
   - `rfdetr_result.png`
   - `yolo_result_0.png`（以及 `yolo_result_{i}.png`）

### 用法 2：只複製程式片段

你也可以直接參考 notebook 內的兩段程式碼：

- RF-DETR：`from rfdetr import RFDETRSegMedium`，並呼叫 `model.predict(image, threshold=0.5)`
- YOLO：`from ultralytics import YOLO`，並呼叫 `model.predict(source=image)`

---

## 目錄內容

- `YOLO_Und_RF_DETR_Demo.ipynb`：YOLO 與 RF-DETR 的推論與可視化 Demo
- `README.md`：專案說明

---

## Reference - 參考資料

- ultralytics 圖片示範來源：https://ultralytics.com/images/bus.jpg
- ultralytics: https://github.com/ultralytics/ultralytics
- YOLO Docs: https://docs.ultralytics.com/zh/tasks/segment/
- RF-DETR : https://github.com/roboflow/rf-detr
