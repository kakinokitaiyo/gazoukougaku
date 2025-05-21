import cv2
import numpy as np
import glob
import os

# チェスボードの交点の数
chessboard_size = (9, 6)

# チェスボードの3D座標
objp = np.zeros((chessboard_size[0]*chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

objpoints = []
imgpoints = []

# フォルダ設定
image_dir = r"C:\Temp\camera_test"       # 読み込み元
result_dir = r"C:\Temp\camera_result"    # 保存先
os.makedirs(result_dir, exist_ok=True)

images = glob.glob(os.path.join(image_dir, '*.jpg'))
print(f"📂 {len(images)} 枚の画像を読み込みました")

# コーナー検出
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)
        print(f"✅ 使用できました: {os.path.basename(fname)}")
    else:
        print(f"❌ 使用できませんでした: {os.path.basename(fname)}")


cv2.destroyAllWindows()

# カメラキャリブレーション
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
print("🎯 カメラ行列（内部パラメータ）:")
print(mtx)
print("\n🎯 歪み係数:")
print(dist)

# 結果保存
np.savez(os.path.join(result_dir, "calibration_result.npz"), mtx=mtx, dist=dist)

# チェスボード枠線描画関数
def draw_board_outline(img, corners, board_size):
    if corners is not None and len(corners) >= board_size[0]*board_size[1]:
        tl = tuple(corners[0][0].astype(int))
        tr = tuple(corners[board_size[0]-1][0].astype(int))
        br = tuple(corners[-1][0].astype(int))
        bl = tuple(corners[-board_size[0]][0].astype(int))

        cv2.line(img, tl, tr, (0, 0, 255), 2)
        cv2.line(img, tr, br, (0, 0, 255), 2)
        cv2.line(img, br, bl, (0, 0, 255), 2)
        cv2.line(img, bl, tl, (0, 0, 255), 2)

# 補正対象画像（最初の画像）
img = cv2.imread(images[0])
h, w = img.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), alpha=0.5)
undistorted = cv2.undistort(img, mtx, dist, None, newcameramtx)
cv2.imwrite(os.path.join(result_dir, "undistorted_example.jpg"), undistorted)

# --- 赤枠なし・ラベル付きの比較画像を作成・保存 ---
img_labeled = img.copy()
undistorted_labeled = undistorted.copy()

cv2.putText(img_labeled, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
cv2.putText(undistorted_labeled, "Undistorted", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

plain_labeled_comparison = np.hstack((img_labeled, undistorted_labeled))
cv2.imwrite(os.path.join(result_dir, "plain_comparison_labeled.jpg"), plain_labeled_comparison)


# 枠線描画
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret1, corners1 = cv2.findChessboardCorners(gray, chessboard_size, None)
draw_board_outline(img, corners1, chessboard_size)
cv2.putText(img, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

gray_u = cv2.cvtColor(undistorted, cv2.COLOR_BGR2GRAY)
ret2, corners2 = cv2.findChessboardCorners(gray_u, chessboard_size, None)
draw_board_outline(undistorted, corners2, chessboard_size)
cv2.putText(undistorted, "Undistorted", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

# エッジ抽出比較
edges_before = cv2.Canny(img, 100, 200)
edges_after = cv2.Canny(undistorted, 100, 200)
edge_comparison = np.hstack((edges_before, edges_after))
cv2.imwrite(os.path.join(result_dir, "edge_comparison.jpg"), edge_comparison)

# 枠付き比較画像
board_comparison = np.hstack((img, undistorted))
cv2.imwrite(os.path.join(result_dir, "board_comparison.jpg"), board_comparison)

# 表示
cv2.imshow("Edge Comparison", edge_comparison)
cv2.imshow("Board Outline Comparison", board_comparison)
cv2.waitKey(0)
cv2.destroyAllWindows()
