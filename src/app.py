from pathlib import Path

import ray
import tensorflow as tf
from keras._tf_keras import keras
from ray.air import RunConfig, ScalingConfig
from ray.air.integrations.keras import ReportCheckpointCallback
from ray.train.tensorflow import TensorflowTrainer

RANDOM_SEED = 709

BASE_DIR = Path().resolve()
DATA_DIR = BASE_DIR / "data"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
img_w, img_h, img_ch = IMAGE_TARGET_SIZE = (224, 224, 3)
BATCH_SIZE = 64
EPOCHS = 3


def generate_dataset(batch_size: int) -> tuple[tf.data.Dataset, tf.data.Dataset]:
    train_ds: tf.data.Dataset = keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        validation_split=0.2,
        subset="training",
        seed=RANDOM_SEED,
        image_size=(img_h, img_w),
        batch_size=batch_size,
    )
    train_ds = train_ds.cache().prefetch(tf.data.AUTOTUNE)

    val_ds: tf.data.Dataset = keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        validation_split=0.2,
        subset="validation",
        seed=RANDOM_SEED,
        image_size=(img_h, img_w),
        batch_size=batch_size,
    )
    val_ds = val_ds.cache().prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds


def build_model():
    model = keras.Sequential(
        [
            keras.layers.Input(IMAGE_TARGET_SIZE),
            keras.layers.Rescaling(1 / 255),
            keras.layers.Conv2D(32, 3, activation="relu"),
            keras.layers.Flatten(),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dense(1),
        ]
    )
    model.summary()
    return model


def train_func(config: dict):
    per_worker_batch_size = config.get("batch_size", BATCH_SIZE)
    epochs = config.get("epochs", EPOCHS)
    steps_per_epoch = config.get("steps_per_epoch", 70)
    num_workers = 1
    strategy = tf.distribute.MultiWorkerMirroredStrategy()
    global_batch_size = per_worker_batch_size * num_workers

    checkpoint = keras.callbacks.ModelCheckpoint(
        str(BASE_DIR / ".checkpoints" / "muffin.weights.h5"),
        monitor="val_loss",
        verbose=True,
        save_best_only=True,
        save_weights_only=True,
        mode="min",
    )
    tensorboard = keras.callbacks.TensorBoard(
        str(BASE_DIR / ".tensorboard"),
        histogram_freq=5,
        write_images=True,
    )

    train_ds, val_ds = generate_dataset(global_batch_size)

    with strategy.scope():
        learning_rate = config.get("learning_rate", 1e-3)
        sgd = keras.optimizers.SGD(learning_rate)
        loss = keras.losses.BinaryCrossentropy(from_logits=True)
        model = build_model()

        model.compile(
            loss=loss,
            optimizer=sgd,
            metrics=["accuracy"],
        )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        callbacks=[ReportCheckpointCallback(), checkpoint, tensorboard],
        verbose=True,
    )
    return history.history


def train_tensorflow():
    config = {
        "learning_rate": 1e-3,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
    }

    trainer = TensorflowTrainer(
        train_loop_per_worker=train_func,
        train_loop_config=config,
        scaling_config=ScalingConfig(num_workers=1, use_gpu=True),
        run_config=RunConfig(storage_path=str(BASE_DIR / ".storage")),
    )
    return trainer.fit()


def main():
    ray.init(num_cpus=4, num_gpus=1)
    results = train_tensorflow()
    print(results)
    return results


if __name__ == "__main__":
    main()
