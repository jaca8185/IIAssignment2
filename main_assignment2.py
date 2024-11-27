# Import Libraries
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

def preprocessData():
   data = pd.read_csv('dataset.csv')

   labels = data['emotion']
   features = data.drop('emotion', axis=1)

   scaler = StandardScaler()
   features = scaler.fit_transform(features)
   
   return features, labels, scaler

def splitData(features, labels):
   # Train/val/test split (70/20/10)
   trainValFeatures, testFeatures, trainValLabels, testLabels = train_test_split(
      features, 
      labels, 
      test_size = 0.1, 
      stratify = labels, 
      random_state = 27
   )

   trainFeatures, valFeatures, trainLabels, valLabels = train_test_split(
      trainValFeatures, 
      trainValLabels, 
      test_size = 0.2/0.9, 
      stratify = trainValLabels, 
      random_state = 27
   )
   return trainFeatures, valFeatures, testFeatures, trainLabels, valLabels, testLabels

# Hyperparameter tuning using GridSearchCV
def hyperParaTuning(trainFeatures, trainLabels):
   parameterGrid = {'kernel': ['rbf', 'linear'], 'C': [0.1, 1, 10], 'gamma': ['scale', 'auto']}

   gridSearch = GridSearchCV(SVC(), parameterGrid, cv=3)
   gridSearch.fit(trainFeatures, trainLabels)

   print(f"Best Parameters: {gridSearch.best_params_}")
   return gridSearch.best_estimator_

# Main function
def main():
   features, labels, scaler = preprocessData()
   trainFeatures, valFeatures, testFeatures, trainLabels, valLabels, testLabels = splitData(features, labels)


   print("\nTuning SVC with GridSearch...")
   bestSvcModel = hyperParaTuning(trainFeatures, trainLabels)


   valPredictions = bestSvcModel.predict(valFeatures)
   print("\nValidation Accuracy:", accuracy_score(valLabels, valPredictions))
   print("Validation Classification Report:\n", classification_report(valLabels, valPredictions))


   print("\nEvaluating on Test Set...")
   testPredictions = bestSvcModel.predict(testFeatures)

   print("Test Accuracy:", accuracy_score(testLabels, testPredictions))
   print("Test Classification Report:\n", classification_report(testLabels, testPredictions, zero_division = 0))


   print("\nClassifying test_to_submit.csv...")
   testToSubmit = pd.read_csv('test_to_submit.csv')
   testToSubmitScaled = scaler.transform(testToSubmit)
   submitPredictions = bestSvcModel.predict(testToSubmitScaled)

   submission = pd.DataFrame(submitPredictions, columns = ['emotion'])
   submission.to_csv('submission.csv', index = False)

   print("Submission saved as submission.csv")


if __name__ == "__main__":
   main()
