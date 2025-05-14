import cv2
import numpy as np
import glob
import os

# チェスボードの交点の数（例：9x6の交点 → 10x7のマス）
chessboard_size = (9, 6)

# チェスボードの3D座標（Z=0の平面上）
objp = np.zeros((chessboard_size[0]*chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

objpoints = []  # 実世界の点
imgpoints = []  # 画像上の点

# 保存先フォルダ
image_dir = r"C:\Temp\camera_test"
images = glob.glob(os.path.join(image_dir, '*.jpg'))

print(f"📂 {len(images)} 枚の画像を読み込みました")

# 各画像ごとにチェスボード検出
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)
        cv2.drawChessboardCorners(img, chessboard_size, corners, ret)
        cv2.imshow('Detected Corners', img)
        cv2.waitKey(200)

cv2.destroyAllWindows()

# キャリブレーション
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# 結果の表示
print("🎯 カメラ行列（内部パラメータ）:")
print(mtx)
print("\n🎯 歪み係数:")
print(dist)

# 結果保存
np.savez(os.path.join(image_dir, "calibration_result.npz"), mtx=mtx, dist=dist)

# 補正例（最初の画像）
img = cv2.imread(images[0])
h, w = img.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

# 補正後画像を保存
cv2.imwrite(os.path.join(image_dir, "undistorted_example.jpg"), dst)

cv2.imshow("Original", img)
cv2.imshow("Undistorted", dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
