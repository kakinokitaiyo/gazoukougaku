import numpy as np

# ファイルのパス
file_path = r"C:\Temp\camera_test\calibration_result.npz"

# 読み込み
data = np.load(file_path)

# 中身の確認
print("🎯 カメラ行列（mtx）:")
print(data['mtx'])

print("\n🎯 歪み係数（dist）:")
print(data['dist'])
