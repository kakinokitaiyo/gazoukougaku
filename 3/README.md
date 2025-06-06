# 第3回課題
#  カメラキャリブレーション用スクリプト集

このリポジトリには、OpenCV を用いたカメラキャリブレーションを行うための 3 つの Python スクリプトが含まれています。

##  各スクリプトの説明

### ①camera_capture.py
ウェブカメラから画像を取得し、スペースキーで保存、q キーで終了するスクリプトです。  
キャリブレーションに使用するチェスボード画像を撮影します。  
保存先：C:\Temp\camera_test

### ②calibration.py
保存されたチェスボード画像を使ってカメラの内部パラメータ（カメラ行列・歪み係数）を推定します。  
結果は calibration_result.npz に保存され、歪み補正後のサンプル画像も出力されます。

### ③calibration_result.py
保存された calibration_result.npz を読み込み、カメラ行列と歪み係数を表示するスクリプトです。

## 実行前の準備

必要なライブラリを以下のコマンドでインストールしてください：

```bash
pip install opencv-python numpy
```

## 実験に必要なチェスボードの画像
画像をダウンロードして何かに張り付けてソースコードを実行してください。

https://github.com/opencv/opencv/blob/4.x/doc/pattern.png