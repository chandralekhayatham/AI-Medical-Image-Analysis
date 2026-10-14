# Interview Questions
1. **Explain the project.** An educational pipeline for classifying public chest X-ray images into NORMAL/PNEUMONIA labels using a CNN, with preprocessing, evaluation, and a Flask demo.
2. **Why CNN?** Convolutional filters learn spatial patterns in image data.
3. **Why normalize?** Scaling pixels to 0–1 can support stable training.
4. **Why train/validation/test?** Training fits weights, validation supports model selection, and test estimates held-out performance.
5. **Why not accuracy alone?** Precision, recall, F1, and confusion matrix reveal different error types.
6. **What is sigmoid output?** A model score for the positive class, not automatically a calibrated probability.
7. **What is overfitting?** Learning training-specific patterns that fail to generalize.
8. **What does Flask do?** Serves the local upload interface and invokes model inference.
9. **Limitations?** Dataset bias, label quality, source differences, imbalance, and lack of clinical validation.
10. **Can it diagnose patients?** No. It is a student prototype only.
