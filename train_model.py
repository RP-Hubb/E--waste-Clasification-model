import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# --- 1. SETTINGS & PATHS ---
# Using the path from your screenshot (Note the 'r' for raw string)
base_dir = r'C:\Users\Arnav Sharma\Downloads\archive\modified-dataset'
train_path = os.path.join(base_dir, 'train')
val_path = os.path.join(base_dir, 'val')
test_path = os.path.join(base_dir, 'test')

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10

# --- 2. DATA PREPROCESSING & AUGMENTATION ---
# Augmenting training data to prevent overfitting
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Validation and Test should only be rescaled (normalized)
val_test_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    train_path,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val_gen = val_test_datagen.flow_from_directory(
    val_path,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# --- 3. MODEL BUILDING (TRANSFER LEARNING) ---
# We use MobileNetV2 because it's fast and small for college projects
base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
base_model.trainable = False  # Freeze pre-trained weights

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),  # Helps with generalization
    layers.Dense(train_gen.num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# --- 4. TRAINING ---
print("Starting Training...")
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS
)

# --- 5. SAVE THE MODEL & CLASSES ---
# We save the model as an .h5 file to load it in our second script (the LLM part)
model.save('ewaste_classifier_model.h5')

# Print the class indices so we know which number corresponds to which waste type
print("Class Indices:", train_gen.class_indices)

# --- 6. VISUALIZE RESULTS ---
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss')
plt.legend()
plt.show()