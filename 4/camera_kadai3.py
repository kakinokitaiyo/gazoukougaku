# -*- coding: utf-8 -*-
import numpy as np
import cv2
import os
import itertools

MIN_MATCH_COUNT = 4

# ======== 入出力フォルダ設定 ========
image_folder = './shasinn/sikihou'
output_folder = './homography_results/sikihou'
os.makedirs(output_folder, exist_ok=True)

# 対象とする拡張子一覧
valid_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.JPG']
image_files = sorted([
    f for f in os.listdir(image_folder)
    if os.path.splitext(f)[1] in valid_exts
])

if len(image_files) < 2:
    raise ValueError("画像が2枚以上必要です")

# ======== 全画像ペアに対して処理開始 ========
for img1_name, img2_name in itertools.combinations(image_files, 2):
    img1_path = os.path.join(image_folder, img1_name)
    img2_path = os.path.join(image_folder, img2_name)

    print(f"\n[INFO] 処理開始: {img1_name} vs {img2_name}")

    # 画像読み込み
    img1 = cv2.imread(img1_path, 0)
    img2 = cv2.imread(img2_path, 0)
    img2c = cv2.imread(img2_path)

    if img1 is None or img2 is None or img2c is None:
        print(f"[ERROR] 読み込み失敗: {img1_name} または {img2_name}")
        continue

    # 特徴点抽出
    detector = cv2.AKAZE_create()
    kp1, des1 = detector.detectAndCompute(img1, None)
    kp2, des2 = detector.detectAndCompute(img2, None)

    if des1 is None or des2 is None:
        print(f"[WARN] 記述子が検出できません: {img1_name} vs {img2_name}")
        continue

    try:
        # マッチング実行
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)

        # 比率テストによる良好マッチ抽出
        good = [m for m, n in matches if m.distance < 0.7 * n.distance]

        print(f"[INFO] マッチ数: {len(matches)}, グッドマッチ数: {len(good)}")

        if len(good) > MIN_MATCH_COUNT:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

            # ホモグラフィ推定
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            if M is not None:
                print(f"[OK] ホモグラフィ検出成功")
                matchesMask = mask.ravel().tolist()

                # 対象物の枠線を射影して描画
                h, w = img1.shape
                pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
                dst = cv2.perspectiveTransform(pts, M)
                img2c = cv2.polylines(img2c, [np.int32(dst)], True, (0, 0, 255), 2, cv2.LINE_AA)
            else:
                print(f"[FAIL] ホモグラフィ行列が計算できませんでした")
                matchesMask = None
        else:
            print(f"[WARN] マッチ数不足 ({len(good)}/{MIN_MATCH_COUNT})")
            matchesMask = None

        # 対応点描画
        draw_params = dict(matchColor=(0, 255, 0),
                           singlePointColor=None,
                           matchesMask=matchesMask,
                           flags=2)

        img3 = cv2.drawMatches(img1, kp1, img2c, kp2, good, None, **draw_params)

        # ファイル保存
        out_name = f"{os.path.splitext(img1_name)[0]}_vs_{os.path.splitext(img2_name)[0]}_AKAZE_homo.png"
        out_path = os.path.join(output_folder, out_name)
        cv2.imwrite(out_path, img3)
        print(f"[SAVE] 結果保存: {out_name}")

    except cv2.error as e:
        print(f"[ERROR] OpenCVエラー発生: {e}")
