"""

Architecture: stem conv -> 3 residual stages (8 -> 16 -> 32 channels) ->
global average pool -> dense(4). Sized to land around 15k-25k parameters
so the int8-quantized model fits comfortably in ESP-EYE's PSRAM budget.

Input: 96x96 grayscale images.
Output: 4 classes -> [clear, left_blocked, center_blocked, right_blocked]

Expected directory layout for training data:

    data/
      train/
        clear/            *.jpg or *.png
        left_blocked/
        center_blocked/
        right_blocked/
      val/
        clear/
        left_blocked/
        center_blocked/
        right_blocked/
"""

import argparse 
import os
import matplotlib.pyplot as plt

import tensorflow as tf
from keras import layers, models

SET_EPOCHS = 20
IMG_SIZE = 96
CLASS_NAMES = ["clear", "left_blocked", "center_blocked", "right_blocked"]
NUM_CLASSES = len(CLASS_NAMES)


def residual_block(x, filters, name):
    """Two 3x3 convs with a skip connection. Stride is always 1 here;
    downsampling happens in a separate conv before each stage."""
    shortcut = x

    y = layers.Conv2D(filters, 3, padding="same", use_bias=False, name=f"{name}_conv1")(x)
    y = layers.BatchNormalization(name=f"{name}_bn1")(y)
    y = layers.ReLU(name=f"{name}_relu1")(y)

    y = layers.Conv2D(filters, 3, padding="same", use_bias=False, name=f"{name}_conv2")(y)
    y = layers.BatchNormalization(name=f"{name}_bn2")(y)

    # Shortcut channel count must match; project with 1x1 conv if it doesn't.
    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, padding="same", use_bias=False, name=f"{name}_proj")(shortcut)
        shortcut = layers.BatchNormalization(name=f"{name}_proj_bn")(shortcut)

    out = layers.Add(name=f"{name}_add")([y, shortcut])
    out = layers.ReLU(name=f"{name}_out")(out)
    return out


def downsample(x, filters, name):
    """Stride-2 conv that both downsamples spatially and changes channel count."""
    y = layers.Conv2D(filters, 3, strides=2, padding="same", use_bias=False, name=f"{name}_conv")(x)
    y = layers.BatchNormalization(name=f"{name}_bn")(y)
    y = layers.ReLU(name=f"{name}_relu")(y)
    return y


def build_tiny_resnet(input_shape=(IMG_SIZE, IMG_SIZE, 1), num_classes=NUM_CLASSES):
    inputs = layers.Input(shape=input_shape, name="input")

    # Stem: 96x96x1 -> 48x48x8
    x = layers.Conv2D(8, 3, strides=2, padding="same", use_bias=False, name="stem_conv")(inputs)
    x = layers.BatchNormalization(name="stem_bn")(x)
    x = layers.ReLU(name="stem_relu")(x)

    # Stage 1: 48x48x8
    x = residual_block(x, 8, name="stage1")

    # Downsample -> 24x24x16, Stage 2
    x = downsample(x, 16, name="down1")
    x = residual_block(x, 16, name="stage2")

    # Downsample -> 12x12x32, Stage 3
    x = downsample(x, 32, name="down2")
    x = residual_block(x, 32, name="stage3")

    # Head
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="classifier")(x)

    return models.Model(inputs, outputs, name="tiny_resnet_obstacle")


def build_datasets(data_dir, batch_size):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(data_dir, "train"),
        labels="inferred",
        label_mode="categorical",
        class_names=CLASS_NAMES,
        color_mode="grayscale",
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=batch_size,
        shuffle=True,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(data_dir, "val"),
        labels="inferred",
        label_mode="categorical",
        class_names=CLASS_NAMES,
        color_mode="grayscale",
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=batch_size,
        shuffle=False,
    )

    # Normalize to [0, 1] and apply light augmentation on the training set only.
    normalize = layers.Rescaling(1.0 / 255)

    augment = tf.keras.Sequential([
        layers.RandomBrightness(0.15),
        layers.RandomContrast(0.15),
        layers.RandomRotation(0.03),  # small: obstacle course framing is roughly fixed
    ])

    train_ds = train_ds.map(lambda x, y: (normalize(augment(x)), y))
    val_ds = val_ds.map(lambda x, y: (normalize(x), y))

    return train_ds.prefetch(tf.data.AUTOTUNE), val_ds.prefetch(tf.data.AUTOTUNE)

def draw_loss_plot(history):

    # Draw a graph of the loss, which is the distance between
    # the predicted and actual values during training and validation.
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(train_loss) + 1)

    plt.plot(epochs, train_loss, 'g', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def draw_accuracy_plot(history):
    # Draw a graph of the accuracy, which is the percentage of correct predictions
    # during training and validation.
    train_acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    epochs = range(1, len(train_acc) + 1)

    plt.plot(epochs, train_acc, 'g', label='Training accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

def main():
    print(f"\nBuilding and training Tiny-ResNet on {NUM_CLASSES} classes: {CLASS_NAMES}\n")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="Path to data/ containing train/ and val/")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out", default="tiny_resnet_obstacle.keras")
    args = parser.parse_args()

    train_ds, val_ds = build_datasets(args.data_dir, args.batch_size)

    model = build_tiny_resnet()
    model.summary()
    print(f"Total parameters: {model.count_params():,}")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(args.out, save_best_only=True, monitor="val_accuracy"),
        tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True, monitor="val_accuracy"),
        tf.keras.callbacks.ReduceLROnPlateau(patience=4, factor=0.5, monitor="val_loss"),
    ]

    history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks)

    draw_loss_plot(history)
    draw_accuracy_plot(history)

    print(f"\nBest model saved to {args.out}")
    print("Next: run Model_Quantizing.py on this file to produce the model.h")


if __name__ == "__main__":
    main()