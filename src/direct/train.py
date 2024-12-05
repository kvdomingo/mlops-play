import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
from keras._tf_keras import keras

from src.direct.settings import settings

img_w, img_h, img_ch = settings.IMAGE_TARGET_SIZE


def main():
    train_ds: tf.data.Dataset = keras.utils.image_dataset_from_directory(
        settings.TRAIN_DIR,
        validation_split=0.2,
        subset="training",
        seed=settings.RANDOM_SEED,
        image_size=(img_h, img_w),
        batch_size=settings.BATCH_SIZE,
    )
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

    val_ds: tf.data.Dataset = keras.utils.image_dataset_from_directory(
        settings.TRAIN_DIR,
        validation_split=0.2,
        subset="validation",
        seed=settings.RANDOM_SEED,
        image_size=(img_h, img_w),
        batch_size=settings.BATCH_SIZE,
    )
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    total_train = len(train_ds) * settings.BATCH_SIZE
    total_val = len(val_ds) * settings.BATCH_SIZE

    model = keras.Sequential(
        [
            keras.layers.Input(settings.IMAGE_TARGET_SIZE, name="input"),
            keras.layers.Rescaling(1 / 255, name="rescale"),
            keras.layers.Conv2D(32, 3, activation="relu", name="conv2d_1"),
            keras.layers.Flatten(name="flat"),
            keras.layers.Dense(128, activation="relu", name="dense_1"),
            keras.layers.Dense(1, name="output"),
        ]
    )
    model.summary()

    continue_ = input("\nContinue [Y/n]? ")
    if continue_.lower() == "n":
        return

    os.makedirs(settings.BASE_DIR / ".checkpoints", exist_ok=True)
    os.makedirs(settings.BASE_DIR / ".tensorboard", exist_ok=True)
    os.makedirs(settings.BASE_DIR / ".logs", exist_ok=True)

    checkpoint = keras.callbacks.ModelCheckpoint(
        str(settings.BASE_DIR / ".checkpoints" / "muffin.weights.h5"),
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=True,
        mode="min",
    )
    tensorboard = keras.callbacks.TensorBoard(
        str(settings.BASE_DIR / ".tensorboard"),
        histogram_freq=5,
        write_images=True,
    )
    logger = keras.callbacks.CSVLogger(
        str(settings.BASE_DIR / ".logs" / "muffin.log.csv")
    )

    sgd = keras.optimizers.SGD(settings.LEARNING_RATE)
    loss = keras.losses.BinaryCrossentropy(from_logits=True)

    model.compile(
        loss=loss,
        optimizer=sgd,
        metrics=["accuracy"],
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=settings.EPOCHS,
        steps_per_epoch=total_train // settings.BATCH_SIZE,
        validation_steps=total_val // settings.BATCH_SIZE,
        callbacks=[checkpoint, tensorboard, logger],
    )
    results = history.history

    print(results)
    return results


if __name__ == "__main__":
    main()
