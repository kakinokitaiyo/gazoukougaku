import os
import pyheif
from PIL import Image

# 入力フォルダと出力フォルダを設定
input_folder = '/home/irsl/gazoukougaku/4/shasinn/hennkan'
output_folder = os.path.join(input_folder, 'converted')
os.makedirs(output_folder, exist_ok=True)

# すべてのHEICファイルをJPEGに変換
for file in os.listdir(input_folder):
    if file.lower().endswith('.heic'):
        heic_path = os.path.join(input_folder, file)
        jpg_filename = os.path.splitext(file)[0] + '.jpg'
        jpg_path = os.path.join(output_folder, jpg_filename)

        try:
            heif_file = pyheif.read(heic_path)
            image = Image.frombytes(
                heif_file.mode,
                heif_file.size,
                heif_file.data,
                "raw",
                heif_file.mode,
                heif_file.stride,
            )
            image.save(jpg_path, "JPEG")
            print(f"変換完了: {file} → {jpg_filename}")
        except Exception as e:
            print(f"変換失敗: {file} → {e}")
