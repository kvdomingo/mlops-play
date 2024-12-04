import tensorflow as tf
from keras._tf_keras import keras

from src.direct.settings import settings

img_w, img_h, img_ch = settings.IMAGE_TARGET_SIZE


def main():
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
    model.load_weights(str(settings.BASE_DIR / ".checkpoints" / "muffin.weights.h5"))

    test_ds: tf.data.Dataset = keras.utils.image_dataset_from_directory(
        settings.TEST_DIR,
        seed=settings.RANDOM_SEED,
        image_size=(img_h, img_w),
        batch_size=settings.BATCH_SIZE,
    )
    total_test = len(test_ds) * settings.BATCH_SIZE

    evals = model.evaluate(
        test_ds, verbose=True, steps=total_test // settings.BATCH_SIZE
    )

    print(evals)
    return evals


if __name__ == "__main__":
    main()
