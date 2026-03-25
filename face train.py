from pathlib import Path
import tensorflow as tf
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support

# ----------------------------
# CONFIG
# ----------------------------

TRAIN_DIR =  Path("/Users/pragyabhatnagar/Desktop/face detect/Datasets/images/train")
VAL_DIR   =  Path("/Users/pragyabhatnagar/Desktop/face detect/Datasets/images/validation")     # 👈 validation folder
TEST_DIR  = Path("/Users/pragyabhatnagar/Desktop/face detect/Datasets/facial test")

IMG_SIZE = (48, 48)
BATCH_SIZE = 64
SEED = 42
EPOCHS = 20

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODEL_DIR / "emotion_cnn.keras"
LABELS_PATH = MODEL_DIR / "labels.txt"


# ----------------------------
# MODEL
# ----------------------------
def build_model(num_classes: int) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(48, 48, 1))
    x = tf.keras.layers.Rescaling(1.0 / 255.0)(inputs)

    x = tf.keras.layers.RandomFlip("horizontal")(x)
    x = tf.keras.layers.RandomRotation(0.08)(x)
    x = tf.keras.layers.RandomZoom(0.10)(x)

    for filters, drop in [(32, 0.25), (64, 0.25), (128, 0.30)]:
        x = tf.keras.layers.Conv2D(filters, 3, padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.MaxPooling2D()(x)
        x = tf.keras.layers.Dropout(drop)(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(256)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Dropout(0.40)(x)

    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs)


# ----------------------------
# DATA LOADER
# ----------------------------
def load_ds(folder: Path, shuffle: bool):
    return tf.keras.utils.image_dataset_from_directory(
        folder,
        color_mode="grayscale",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        seed=SEED
    )


# ----------------------------
# MAIN
# ----------------------------
def main():
    for d in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        if not d.exists():
            raise FileNotFoundError(f"Missing folder: {d}")

    print("✅ Loading datasets...")
    train_ds = load_ds(TRAIN_DIR, shuffle=True)
    val_ds   = load_ds(VAL_DIR, shuffle=False)
    test_ds  = load_ds(TEST_DIR, shuffle=False)

    # Ensure same class order
    if not (train_ds.class_names == val_ds.class_names == test_ds.class_names):
        raise ValueError("❌ Class folders mismatch across train/val/test")

    class_names = train_ds.class_names
    LABELS_PATH.write_text("\n".join(class_names))
    print("✅ Classes:", class_names)

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(AUTOTUNE)
    val_ds   = val_ds.cache().prefetch(AUTOTUNE)
    test_ds  = test_ds.cache().prefetch(AUTOTUNE)

    model = build_model(len(class_names))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            MODEL_PATH, monitor="val_accuracy",
            save_best_only=True, verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=5, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, verbose=1
        ),
    ]

    print("🚀 Training...")
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks)

    print("🧪 Testing...")
    # Extract labels and predictions
    y_true = []
    y_pred = []
    
    for images, labels in test_ds:
        y_true.extend(labels.numpy())
        preds = model.predict(images, verbose=0)
        y_pred.extend(tf.argmax(preds, axis=1).numpy())

    # Calculate metrics
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')

    print(f"\n📊 Model Performance:")
    print(f"✅ Accuracy:  {acc*100:.2f}%")
    print(f"✅ F1 Score:  {f1:.4f}")
    print(f"✅ Recall:    {recall:.4f}")
    print(f"✅ Precision: {precision:.4f}")

    print("\n📝 Detailed Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    print("✅ Model saved:", MODEL_PATH)
    print("✅ Labels saved:", LABELS_PATH)


if __name__ == "__main__":
    main()