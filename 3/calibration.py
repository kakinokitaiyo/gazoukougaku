import numpy as np
import cv2 as cv
import glob
import os

# チェスボードのコーナー数（交点）
CHECKERBOARD = (9, 6)

# サブピクセル補正の終了条件
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# 3D点： (0,0,0), (1,0,0), ..., (6,5,0)
objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

objpoints = []  # 実世界の座標
imgpoints = []  # 画像上の座標
used_images = []  # 補正対象の画像名を保存

images = glob.glob('*.jpg')

for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    ret, corners = cv.findChessboardCorners(gray, CHECKERBOARD, None)

    if ret:
        print(f"✓ チェスボード検出成功: {fname}")
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)
        used_images.append(fname)

        # コーナー描画
        cv.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(0)
        cv.destroyAllWindows()
    else:
        print(f"✗ チェスボード検出失敗: {fname}")

# キャリブレーションの実行
if len(objpoints) > 0:
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    print("🎯 カメラ行列:\n", mtx)
    print("🎯 歪み係数:\n", dist)

    # 保存ディレクトリ作成
    os.makedirs("undistorted", exist_ok=True)

for fname in used_images:
    img = cv.imread(fname)
    h, w = img.shape[:2]
    print(f"\n📷 元画像サイズ: {w} x {h} ({fname})")

    newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

    # 歪み補正（黒枠を含む）
    dst = cv.undistort(img, mtx, dist, None, newcameramtx)
    print(f"🔧 補正後（切り抜き前）サイズ: {dst.shape[1]} x {dst.shape[0]}")

    # ROIでトリミング
    x, y, roi_w, roi_h = roi
    dst_cropped = dst[y:y+roi_h, x:x+roi_w]
    print(f"✂️ 補正後（ROIで切り抜き）サイズ: {roi_w} x {roi_h}")

    # 保存
    save_path = os.path.join("undistorted", f"undistorted_{os.path.basename(fname)}")
    cv.imwrite(save_path, dst_cropped)
    print(f"✅ 保存完了: {save_path}")
else:
    print("❌ キャリブレーションに使える画像がありません。")