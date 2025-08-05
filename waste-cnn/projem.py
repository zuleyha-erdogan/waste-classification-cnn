# import
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras import layers

# özellik çıkarımı modeli
ozellik_katmani = VGG16(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)

for layer in ozellik_katmani.layers:
    layer.trainable = False

ozellik_katmani.summary()

# yeni model oluşturma
model = tf.keras.models.Sequential()
model.add(ozellik_katmani)
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(32, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

# model derleme
model.compile(
    loss='binary_crossentropy',
    optimizer=tf.keras.optimizers.RMSprop(1e-5),
    metrics=['acc']
)

model.summary()

 GEÇERLEME_Yolu ='veriseti / GECERLEME'
 Test_Yolu = 'veriseti / TEST'

# Veriyi hazırlama
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

validation_generator = test_datagen.flow_from_directory(
    'validation',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Eğitim TAKİP
model.fit(
    train_generator,
    steps_per_epoch=10,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=5
)

# modeli kaydet
model.save('C:/KAYIT_YERİ/model15.h5')

# test işlemi
test_generator = test_datagen.flow_from_directory(
    'TEST',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

test_loss, test_acc = model.evaluate(test_generator, steps=1)
print('Test acc:', test_acc)