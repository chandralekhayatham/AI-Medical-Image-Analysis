from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from data_utils import make_datasets

ROOT=Path(__file__).resolve().parents[1]
def main():
    path=ROOT/"models"/"medical_image_cnn.keras"
    if not path.exists(): raise FileNotFoundError("Run python src/train.py first.")
    _,_,test=make_datasets(ROOT/"data"/"chest_xray")
    model=tf.keras.models.load_model(path)
    truth=np.concatenate([y.numpy().reshape(-1) for _,y in test]).astype(int)
    pred=(model.predict(test,verbose=1).reshape(-1)>=.5).astype(int)
    report=classification_report(truth,pred,target_names=["NORMAL","PNEUMONIA"],output_dict=True,zero_division=0)
    out=ROOT/"outputs"; out.mkdir(exist_ok=True)
    (out/"evaluation_metrics.json").write_text(json.dumps({"accuracy":float(accuracy_score(truth,pred)),
      "classification_report":report,"note":"Educational evaluation only; not clinical validation."},indent=2))
    ConfusionMatrixDisplay(confusion_matrix(truth,pred,labels=[0,1]),display_labels=["NORMAL","PNEUMONIA"]).plot(values_format="d")
    plt.title("Test Set Confusion Matrix (Educational)"); plt.tight_layout(); plt.savefig(out/"confusion_matrix.png",dpi=150)
    print("Saved evaluation_metrics.json and confusion_matrix.png. Review precision, recall and errors too.")
if __name__=="__main__": main()
