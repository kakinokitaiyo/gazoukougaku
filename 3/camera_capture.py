import cv2
import os

# 保存先ディレクトリ
save_dir = r"C:\Temp\camera_test"  # 保存先フォルダを変更したい場合はここを変える
os.makedirs(save_dir, exist_ok=True)

# 空きファイル名（capture_00.jpg ～ capture_99.jpg）を毎回探す関数
def get_next_available_filename(save_dir, prefix="capture_", ext=".jpg"):
    for i in range(100):  # 00～99まで対応（必要なら1000まで増やせます）
        filename = f"{prefix}{i:02d}{ext}"  # 例: capture_00.jpg, capture_01.jpg
        full_path = os.path.join(save_dir, filename)
        if not os.path.exists(full_path):
            return full_path
    return None  # 全て埋まっていた場合

# カメラ起動
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ カメラが開けませんでした")
    exit()

print("📸 スペースキーで撮影、qで終了します")

# 撮影ループ
while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ 画像の取得に失敗しました")
        break

    cv2.imshow('Camera Preview', frame)
    key = cv2.waitKey(1)

    if key == ord(' '):
        filename = get_next_available_filename(save_dir)
        if filename is None:
            print("⚠️ 保存可能な空きファイル名が見つかりません（最大枚数に達しています）")
            break

        # 保存処理
        if frame is None:
            print("❌ 撮影データが無効です")
            continue

        try:
            success = cv2.imwrite(filename, frame)
            if success:
                print(f"✅ 保存成功: {filename}")
            else:
                print(f"❌ 保存失敗: {filename}")
        except Exception as e:
            print(f"❌ 例外が発生しました: {e}")

    elif key == ord('q'):
        print("👋 終了します")
        break

# 後処理
cap.release()
cv2.destroyAllWindows()
