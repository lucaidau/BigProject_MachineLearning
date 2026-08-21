import os
import cv2 as cv 
import numpy as np
from pathlib import Path
from tqdm import tqdm

def load_images_from_folder(dataset_dir, target_size=(128, 128)):
    X_data = []
    Y_data = []

    label_map ={
        'COVID':0,
        'Lung_Opacity': 1,
        'Normal': 2,
        'Viral Pneumonia':3
    }

    dataset_path = Path(dataset_dir)

    for class_name, label_id in label_map.items():
        class_dir = dataset_path / class_name
        if not class_dir.exists():
            continue
        
        img_dir = class_dir / "images"
        mask_dir = class_dir / "masks"

        print(f"Đang xử lí thư mục {class_name}...")
        img_files = list(img_dir.glob("*.png"))
        for img_path in tqdm(img_files):
            mask_path = mask_dir / img_path.name
            if not mask_path.exists():
                continue

            img_read = cv.imread(str(img_path), cv.IMREAD_GRAYSCALE)
            mask_read = cv.imread(str(mask_path), cv.IMREAD_GRAYSCALE)
            
            if img_read is None or mask_read is None:
                continue

            img_resized = cv.resize(img_read, target_size, interpolation=cv.INTER_AREA)
            mask_resized = cv.resize(mask_read, target_size, interpolation=cv.INTER_NEAREST)

            _, mask_bin = cv.threshold(mask_resized, 127, 255, cv.THRESH_BINARY)

            mask_img = cv.bitwise_and(img_resized, img_resized, mask=mask_bin)

            X_data.append(mask_img)
            Y_data.append(label_id)

    X_data = np.array(X_data, dtype=np.uint8)
    Y_data = np.array(Y_data, dtype=np.int64)
    return X_data, Y_data


RAW_DATA_DIR = "Data/Raw/COVID-19_Radiography_Dataset"
x, y = load_images_from_folder(RAW_DATA_DIR, target_size=(128, 128))

print(f"\n Tổng số ảnh đã xử lí: {len(x)}")
print (f"\n Kích thước x: {x.shape}")
print (f"\n Kích thước y: {y.shape}")

os.makedirs("Data/Processed", exist_ok=True)

np.save("Data/Processed/x_resized_masked.npy", x)
np.save("Data/Processed/y_labels.npy", y)
print("\n Dữ liệu đã được lưu thành công vào thư mục Data/Processed.")