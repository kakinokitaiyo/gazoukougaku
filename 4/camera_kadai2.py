import cv2
import numpy as np
import os
import itertools

# === 設定 ===
image_folder = './shasinn/sikihou'
output_folder = './result_matches/sikihou'
os.makedirs(output_folder, exist_ok=True)

valid_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.JPG']
image_files = sorted([
    f for f in os.listdir(image_folder)
    if os.path.splitext(f)[1] in valid_exts
])
if len(image_files) < 2:
    raise ValueError("画像が2枚以上必要です")

# === 特徴量比較関数 ===
def match_features(img1, img2, img1_name, img2_name, detector='SIFT'):
    if detector == 'SIFT':
        feature_detector = cv2.SIFT_create()
        norm = cv2.NORM_L2
    elif detector == 'ORB':
        feature_detector = cv2.ORB_create()
        norm = cv2.NORM_HAMMING
    else:
        raise ValueError("対応していない特徴量です")

    kp1, des1 = feature_detector.detectAndCompute(img1, None)
    kp2, des2 = feature_detector.detectAndCompute(img2, None)

    if des1 is None or des2 is None:
        print(f"[警告] 特徴記述子が検出できません: {img1_name} vs {img2_name}（{detector}）")
        return None, 0, 0

    try:
        bf = cv2.BFMatcher(norm)
        matches = bf.knnMatch(des1, des2, k=2)

        # 比率テストで良好マッチを選択（Lowe's ratio test）
        good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

        # 描画用：最大20個の良好マッチを描画
        result_img = cv2.drawMatches(
            img1, kp1, img2, kp2,
            good_matches[:20], None, flags=2
        )

        return result_img, len(matches), len(good_matches)
    except cv2.error as e:
        print(f"[エラー] マッチング失敗: {img1_name} vs {img2_name}（{detector}） → {e}")
        return None, 0, 0

# === 全画像ペアでマッチング ===
for detector in ['SIFT', 'ORB']:
    for img1_name, img2_name in itertools.combinations(image_files, 2):
        img1_path = os.path.join(image_folder, img1_name)
        img2_path = os.path.join(image_folder, img2_name)

        img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

        if img1 is None or img2 is None:
            print(f"[エラー] 画像読み込み失敗: {img1_name} または {img2_name}")
            continue

        print(f"[INFO] マッチング処理開始: {img1_name} vs {img2_name}（{detector}）")

        result, num_all_matches, num_good_matches = match_features(
            img1, img2, img1_name, img2_name, detector=detector
        )

        if result is not None:
            out_name = f"{os.path.splitext(img1_name)[0]}_vs_{os.path.splitext(img2_name)[0]}_{detector}.png"
            out_path = os.path.join(output_folder, out_name)
            cv2.imwrite(out_path, result)
            print(f"[OK] 保存しました: {out_name}（マッチ数: {num_all_matches}, グッドマッチ数: {num_good_matches}）")
        else:
            print(f"[スキップ] 出力なし: {img1_name} vs {img2_name}（{detector}）")
