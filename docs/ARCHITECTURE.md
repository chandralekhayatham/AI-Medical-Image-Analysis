# Architecture
Public labelled X-rays → TensorFlow image loader → grayscale 256×256 tensors and pixel normalization → CNN training → held-out test evaluation → saved model and plots → Flask upload page → educational model score.

Training uses a validation split, early stopping, and best-validation-loss checkpointing. The output score is not a calibrated clinical probability. This is not a clinical system.
