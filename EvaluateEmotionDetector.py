import numpy as np
from keras.models import model_from_json
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# Load JSON and create the model
json_file = open('D:/Python Program/Design Project/CCA Project/Emotion_detection_with_CNN-main/model/New folder/emotion_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)

# Load weights into the model
emotion_model.load_weights('D:/Python Program/Design Project/CCA Project/Emotion_detection_with_CNN-main/model/New folder/emotion_model.h5')
print("Loaded model from disk")

# Initialize the image data generator with rescaling
test_data_gen = ImageDataGenerator(rescale=1./255)

# Preprocess all test images
test_generator = test_data_gen.flow_from_directory(
    'D:\Python Program\Design Project\CCA Project\Emotion_detection_with_CNN-main\Data\Test',
    target_size=(48, 48),
    batch_size=64,
    color_mode="grayscale",
    class_mode='categorical')

# Do prediction on the test data
predictions = emotion_model.predict_generator(test_generator)

# Calculate the accuracy of the model
true_labels = test_generator.classes
predicted_labels = predictions.argmax(axis=1)
accuracy = accuracy_score(true_labels, predicted_labels)

# Print the accuracy
print("Accuracy:", accuracy)

# Display the confusion matrix
print("-----------------------------------------------------------------")
c_matrix = confusion_matrix(true_labels, predicted_labels)
print(c_matrix)
cm_display = ConfusionMatrixDisplay(confusion_matrix=c_matrix, display_labels=emotion_dict)
cm_display.plot(cmap=plt.cm.Blues)
plt.show()

# Classification report
print("-----------------------------------------------------------------")
print(classification_report(true_labels, predicted_labels))
