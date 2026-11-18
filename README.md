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

### 疑似シェーディングの実装

``` 
python3 depth_normal.py --demo
```

### 擬似シェーディングの実装（デモモード）

推定した深度マップから算出した単位法線ベクトルを利用し、元画像に対して仮想光源による陰影（擬似シェーディング）を付与するデモ機能です。

#### 実行方法

```bash
python3 depth_normal.py --demo
```

出力先のファイル名を指定する場合は `-o` オプションを使用します（デフォルトは `normal_map.png`）。

```bash
python3 depth_normal.py --demo -o output/result.png
```

※ `--demo` モードの実行には `scikit-image` が必要です（`pip install scikit-image`）。

#### 出力されるファイル

実行すると、`skimage.data.coffee()` のサンプル画像を対象に、以下の3つの画像が生成されます。


<img src="figures/normal_map_input.png" width=480>
* **`normal_map_input.png`**: 入力として使用された元のサンプル画像


<img src="figures/normal_map.png" width=480>
* **`normal_map.png`**: 深度勾配から生成された法線マップ画像

<img src="figures/normal_map_shaded_0.png" width=480>
<img src="figures/normal_map_shaded_1.png" width=480>

<img src="figures/normal_map_shaded_2.png" width=480>

<img src="figures/normal_map_shaded_3.png" width=480>

* **`normal_map_shaded.png`**: 法線ベクトルと仮想光源に基づき擬似シェーディングを適用した画像


効果：
異なるシェーディングの画像を見比べてください。
- コーヒーカップの縁
- テーブルの木目の様子
それらが微妙に違ってきているはずです。

#### 処理の仕組み

1. **コサイン（内積）の計算**: 各ピクセルの単位法線ベクトル $\mathbf{N}$ と、想定する平行光源の方向ベクトル $\mathbf{L}$（デフォルトは左上・手前方向 `(-1.0, -1.0, 1.0)`）の内積 $\cos\theta = \mathbf{N} \cdot \mathbf{L}$ を算出します。
2. **輝度（V）の変調**: 元画像を HSV 色空間に変換し、色相（H）と彩度（S）を維持したまま、明度（V）に対して以下の係数を掛け合わせます。
   $$V_{\text{shaded}} = V \times (1.0 + \text{strength} \times \cos\theta)$$
3. **正規化と RGB 復元**: 画像全体の明度の最大値が `1.0` となるようにスケーリングを行った後、RGB 色空間へ戻して保存します。
