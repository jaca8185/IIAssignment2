import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE

def dataProcess():
   data = pd.read_csv('dataset.csv')

   labels = data['emotion']
   features = data.drop('emotion', axis=1)

   scaler = StandardScaler()
   features = scaler.fit_transform(features)
   
   return features, labels, scaler


def dataSplit(features, labels):
    # Train/val/test split (70/20/10)
   tValFeatures, testFeatures, tValLabels, testLabels = train_test_split(
      features, 
      labels, 
      test_size = 0.1, 
      stratify = labels, 
      random_state = 42
   )

   trainFeatures, valFeatures, trainLabels, valLabels = train_test_split(
      tValFeatures, 
      tValLabels, 
      test_size = 0.2 / 0.9, 
      stratify = tValLabels, 
      random_state = 42
   )

   return trainFeatures, valFeatures, testFeatures, trainLabels, valLabels, testLabels


def balanceClasses(features, labels):
   smote = SMOTE(random_state = 42)
   balancedFeatures, balancedLabels = smote.fit_resample(features, labels)

   return balancedFeatures, balancedLabels


def trainModel(model, trainFeatures, trainLabels, valFeatures, valLabels):
   model.fit(trainFeatures, trainLabels)
   valPredictions = model.predict(valFeatures)

   accuracy = accuracy_score(valLabels, valPredictions)
   print(f"Validation Accuracy: {accuracy:.2f}")
   print("Classification Report: ")
   print(classification_report(valLabels, valPredictions, zero_division = 0))

   return model


def hyperTune(trainFeatures, trainLabels):
   paramGrid = {
      'n_neighbors': [3, 5, 7, 9],
      'weights': ['uniform', 'distance'],
      'metric': ['euclidean', 'manhattan', 'minkowski']
   }

   knnModel = KNeighborsClassifier()
   gridSearch = GridSearchCV(knnModel, paramGrid, cv = 5, scoring = 'balanced_accuracy', n_jobs = -1)
   gridSearch.fit(trainFeatures, trainLabels)

   print(f"Best Parameters: {gridSearch.best_params_}")

   return gridSearch.best_estimator_

# Main function
def main():
   features, labels, scaler = dataProcess()
   trainFeatures, valFeatures, testFeatures, trainLabels, valLabels, testLabels = dataSplit(features, labels)

   print("\nTraining KNN...")
   knnModel = KNeighborsClassifier(n_neighbors = 5)
   knnModel = trainModel(knnModel, trainFeatures, trainLabels, valFeatures, valLabels)

   print("\nTuning KNN Hyperparameters...")
   bestKModel = hyperTune(trainFeatures, trainLabels)

   print("\nEvaluating Best KNN on Validation Set...")
   valPredictions = bestKModel.predict(valFeatures)

   print("Validation Classification Report:")
   print(classification_report(valLabels, valPredictions, zero_division = 0))

   print("\nEvaluating Best KNN on Test Set...")
   testPredictions = bestKModel.predict(testFeatures)

   print("Test Accuracy:", accuracy_score(testLabels, testPredictions))
   print("Classification Report:")
   print(classification_report(testLabels, testPredictions, zero_division = 0))

   print("\nClassifying test_to_submit.csv...")
   testSubmit = pd.read_csv('test_to_submit.csv')
   tsScaled = scaler.transform(testSubmit)

   submit = bestKModel.predict(tsScaled)

   submission = pd.DataFrame(submit, columns = ['emotion'])
   submission.to_csv('outputs', index=False)

   print("Submission saved as outputs")

if __name__ == "__main__":
   main()
