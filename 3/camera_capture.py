import cv2
import os

# 保存先を英数字だけのパスに変更
save_dir = r"C:\Temp\camera_test"
os.makedirs(save_dir, exist_ok=True)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ カメラが開けませんでした")
    exit()

print("📸 スペースキーで撮影、qで終了します")

count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ 画像の取得に失敗しました")
        break

    cv2.imshow('Camera Preview', frame)
    key = cv2.waitKey(1)

    if key == ord(' '):
        filename = os.path.join(save_dir, f"capture_{count:02d}.jpg")
        success = cv2.imwrite(filename, frame)
        if success:
            print(f"✅ 保存成功: {filename}")
            count += 1
        else:
            print(f"❌ 保存失敗: {filename}")

    elif key == ord('q'):
        print("👋 終了します")
        break

cap.release()
cv2.destroyAllWindows()
