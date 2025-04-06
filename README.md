# KitaHack

Setup
- run the code in your terminal: pip install -r requirements.txt
- Add images needed for training and validation of machine learning model into the zip files provided above (Only do so if you want to train model from scratch with your own data)
- foodWasteClassifier.keras / foodWasteClassifier.tflite is a trained ML model to differentiate food waste from non food waste 
- ConverterScript.py is used to convert the .keras file to a tensorflowlite file. (Only use if a new model is trained from scratch)
- ModelEvaluation.py is used to evaluate the precision and accuracy of the model and is used to test outputs for single image inputs
- WithCamera.py is used to open a webcam on your device for image classiffication
