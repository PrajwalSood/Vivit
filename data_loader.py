import tensorflow as tf
import numpy as np

@tf.function
def preprocess(frames: tf.Tensor, label: tf.Tensor):
    """Preprocess the frames tensors and parse the labels."""
    # Preprocess images
    frames = tf.image.convert_image_dtype(
        frames[
            ..., tf.newaxis
        ],  # The new axis is to help for further processing with Conv3D layers
        tf.float32,
    )
    # Parse label
    label = tf.cast(label, tf.float32)
    return frames, label


def prepare_dataloader(
    videos: np.ndarray,
    labels: np.ndarray,
    loader_type: str = "train",
    batch_size: int = 32,
):
    """Utility function to prepare the dataloader."""
    dataset = tf.data.Dataset.from_tensor_slices((videos, labels))

    if loader_type == "train":
        dataset = dataset.shuffle(batch_size * 2)

    dataloader = (
        dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    return dataloader

def get_data(train_videos, train_labels, valid_videos, valid_labels, test_videos, test_labels, BATCH_SIZE):

  trainloader = prepare_dataloader(train_videos, train_labels, "train", BATCH_SIZE)
  validloader = prepare_dataloader(valid_videos, valid_labels, "valid", BATCH_SIZE)
  testloader = prepare_dataloader(test_videos, test_labels, "test", BATCH_SIZE)

  return trainloader, validloader, testloader
