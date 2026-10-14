from pathlib import Path
import tensorflow as tf

SIZE = (256, 256)
BATCH = 32


def make_datasets(root="data/chest_xray"):
    root = Path(root)

    missing = [
        str(root / folder)
        for folder in ("train", "val", "test")
        if not (root / folder).is_dir()
    ]

    if missing:
        raise FileNotFoundError(
            "Missing dataset folders: "
            + ", ".join(missing)
            + ". Expected train, val, and test folders."
        )

    args = {
        "image_size": SIZE,
        "batch_size": BATCH,
        "color_mode": "grayscale",
        "label_mode": "binary",
    }

    train = tf.keras.utils.image_dataset_from_directory(
        root / "train",
        shuffle=True,
        seed=42,
        **args,
    )

    val = tf.keras.utils.image_dataset_from_directory(
        root / "val",
        shuffle=False,
        **args,
    )

    test = tf.keras.utils.image_dataset_from_directory(
        root / "test",
        shuffle=False,
        **args,
    )

    # Alphabetical order: NORMAL = 0, PNEUMONIA = 1.
    scale = tf.keras.layers.Rescaling(1.0 / 255)

    def preprocess(images, labels):
        return scale(images), labels

    return tuple(
        dataset.map(preprocess).prefetch(tf.data.AUTOTUNE)
        for dataset in (train, val, test)
    )

