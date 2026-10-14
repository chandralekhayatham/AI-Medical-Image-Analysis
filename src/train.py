from pathlib import Path
import json
import matplotlib.pyplot as plt
import tensorflow as tf
from data_utils import make_datasets

ROOT = Path(__file__).resolve().parents[1]

def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(256,256,1)),
        tf.keras.layers.Conv2D(16,3,activation="relu"), tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32,3,activation="relu"), tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64,3,activation="relu"), tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.GlobalAveragePooling2D(), tf.keras.layers.Dropout(.35),
        tf.keras.layers.Dense(32,activation="relu"), tf.keras.layers.Dense(1,activation="sigmoid")])
    model.compile(optimizer="adam",loss="binary_crossentropy",
                  metrics=["accuracy",tf.keras.metrics.Precision(name="precision"),
                           tf.keras.metrics.Recall(name="recall")])
    return model

def main():
    (ROOT/"models").mkdir(exist_ok=True); (ROOT/"outputs").mkdir(exist_ok=True)
    train,val,_ = make_datasets(ROOT/"data"/"chest_xray")
    model=build_model()
    callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss",patience=3,restore_best_weights=True),
      tf.keras.callbacks.ModelCheckpoint(ROOT/"models"/"medical_image_cnn.keras",monitor="val_loss",save_best_only=True)]
    history=model.fit(train,validation_data=val,epochs=10,callbacks=callbacks)
    model.save(ROOT/"models"/"medical_image_cnn.keras")
    (ROOT/"outputs"/"training_history.json").write_text(json.dumps(
      {k:[float(v) for v in vals] for k,vals in history.history.items()},indent=2))
    fig,ax=plt.subplots(1,2,figsize=(10,4))
    ax[0].plot(history.history["accuracy"],label="train"); ax[0].plot(history.history["val_accuracy"],label="validation")
    ax[0].set_title("Accuracy"); ax[0].legend()
    ax[1].plot(history.history["loss"],label="train"); ax[1].plot(history.history["val_loss"],label="validation")
    ax[1].set_title("Loss"); ax[1].legend()
    fig.tight_layout(); fig.savefig(ROOT/"outputs"/"training_curves.png",dpi=150)
    print("Training complete. Next: python src/evaluate.py")

if __name__=="__main__": main()
