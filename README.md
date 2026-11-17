# depth-normal
normal surface for depth anything

## 何を実装するのか
- depth-anything で出したdepth情報を元に法線の向きで表示する。

### 依存ライブラリ
https://github.com/DepthAnything/Depth-Anything-V2

単眼画像に対してdepthを算出するライブラリ


https://github.com/katsunori-waragai/disparity-view

depth画像に対して画像表示するライブラリ
その中に、法線ベクトルの向きに応じた色分けをする関数が実装されている。

### 動作環境
- Ubuntu
- python
- CPU

### スクリプト
- 単眼画像の1枚画像に対して、depth-anything を用いて奥行きを出し、それを法線ベクトルの向きで表示するスクリプト

## 実装済みスクリプト

このリポジトリでは、単眼画像から深度を推定し、法線ベクトルの向きを色として可視化する Python スクリプトを実装しました。

### 依存関係のインストール

```bash
python3 -m pip install -r requirements.txt
```

### 実行方法

```bash
python3 depth_normal.py input.jpg -o normal_map.png
```

- `input.jpg` : 入力画像
- `normal_map.png` : 法線ベクトルを色で表した出力画像

### 実装の考え方

1. 単眼画像から depth map を推定する
2. 深度画像を高さフィールドとして扱う
3. x, y 方向の勾配から法線ベクトルを計算する
4. 法線ベクトルを RGB 値に写像して画像として出力する

Depth-Anything-V2 がインストール済みの場合は、その推論を優先して利用します。未インストールの環境では、CPU でも動くフォールバック処理で深度マップを生成します。