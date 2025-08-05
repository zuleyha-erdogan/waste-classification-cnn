import zipfile
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore

# 1. archive.zip dosyasını açalım
zip_path = "archive.zip"
extract_folder = "derm_dataset"

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_folder)

print(f"Veriler '{extract_folder}' klasörüne çıkarıldı.")

# 2. Eğitim ve test klasörlerini belirleyelim
train_dir = os.path.join(extract_folder, "train")
test_dir = os.path.join(extract_folder, "test")

# 3. Klasörlerdeki sınıf ve veri sayılarını yazdıran fonksiyon
def print_class_counts(directory):
    print(f"\n'{directory}' klasöründeki sınıflar ve görüntü sayıları:")
    for class_name in sorted(os.listdir(directory)):
        class_path = os.path.join(directory, class_name)
        if os.path.isdir(class_path):
            count = len(os.listdir(class_path))
            print(f"{class_name}: {count}")

print_class_counts(train_dir)
print_class_counts(test_dir)

# 4. Veri artırma ve ön işleme ayarları
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

# 5. Veri yükleyiciler
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)
