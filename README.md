# ♻️ Glass and Paper Image Classification (VGG16 - Transfer Learning)

This project is a deep learning-based image classification model that distinguishes between **glass** and **paper** images using **VGG16** with transfer learning.

## 🧠 Technologies Used

- Python
- TensorFlow & Keras
- VGG16 (pre-trained on ImageNet)
- CNN (Convolutional Neural Network)
- ImageDataGenerator (data augmentation)

## 📁 Dataset Structure

Your dataset should be organized as follows:
veriseti/
├── EGITIM/ # Training data
│ ├── cam/ # Glass images
│ └── kagit/ # Paper images
├── GECERLEME/ # Validation data
│ ├── cam/
│ └── kagit/
└── TEST/ # Test data
├── cam/
└── kagit/
## 🧩 Model Architecture

- VGG16 (convolutional base, frozen)
- Flatten layer
- Dense(128, relu)
- Dense(64, relu)
- Dense(32, relu)
- Dense(2, softmax)

## 🚀 Training

- Model is trained for 10 epochs.
- Uses categorical labels (glass and paper).
- Data augmentation is applied to training images.
- Loss function: **categorical_crossentropy**
- Optimizer: RMSprop (learning rate: 1e-5)

## 💾 Output

- The trained model is saved as `.h5` file:
- Evaluation is performed using the test dataset:
model.save('C:/KAYIT_YERİ/model15.h5')

test_loss, test_acc = model.evaluate(test_generator)
print("Test Accuracy:", test_acc) 

## 📌 Notes

- For multi-class classification, always use `categorical_crossentropy` with `softmax` output.
- Make sure your data is properly labeled and balanced for optimal performance.
