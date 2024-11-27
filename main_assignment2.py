# Import Libraries
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Function to preprocess the data
def preprocess_data():
   # Load dataset
   data = pd.read_csv('dataset.csv')

   # Split features and labels
   labels = data['emotion']
   features = data.drop('emotion', axis=1)

   # Standardize features
   scaler = StandardScaler()
   features = scaler.fit_transform(features)
   
   return features, labels, scaler

# Function to split the data
def split_data(features, labels):
   # Train/val/test split (70/20/10)
   train_val_features, test_features, train_val_labels, test_labels = train_test_split(
      features, 
      labels, 
      test_size=0.1, 
      stratify=labels, 
      random_state=42
   )
   train_features, val_features, train_labels, val_labels = train_test_split(
      train_val_features, 
      train_val_labels, 
      test_size=0.2/0.9, 
      stratify=train_val_labels, 
      random_state=42
   )
   return train_features, val_features, test_features, train_labels, val_labels, test_labels

# Hyperparameter tuning using GridSearchCV
def hyperparameter_tuning(train_features, train_labels):
   param_grid = {'kernel': ['rbf', 'linear'], 'C': [0.1, 1, 10], 'gamma': ['scale', 'auto']}
   grid_search = GridSearchCV(SVC(), param_grid, cv=3)
   grid_search.fit(train_features, train_labels)
   print(f"Best Parameters: {grid_search.best_params_}")
   return grid_search.best_estimator_

# Main function
def main():
   # Step 1: Preprocess the dataset
   features, labels, scaler = preprocess_data()

   # Step 2: Split the dataset
   train_features, val_features, test_features, train_labels, val_labels, test_labels = split_data(features, labels)

   # Step 3: Train and tune the SVC model
   print("\nTuning SVC with GridSearch...")
   best_svc_model = hyperparameter_tuning(train_features, train_labels)

   # Step 4: Evaluate the model on the validation set
   val_predictions = best_svc_model.predict(val_features)
   print("\nValidation Accuracy:", accuracy_score(val_labels, val_predictions))
   print("Validation Classification Report:\n", classification_report(val_labels, val_predictions))

   # Step 5: Evaluate the model on the test set
   print("\nEvaluating on Test Set...")
   test_predictions = best_svc_model.predict(test_features)
   print("Test Accuracy:", accuracy_score(test_labels, test_predictions))
   print("Test Classification Report:\n", classification_report(test_labels, test_predictions))

   # Step 6: Classify test_to_submit.csv
   print("\nClassifying test_to_submit.csv...")
   test_to_submit = pd.read_csv('test_to_submit.csv')
   test_to_submit_scaled = scaler.transform(test_to_submit)
   submit_predictions = best_svc_model.predict(test_to_submit_scaled)
   submission = pd.DataFrame(submit_predictions, columns=['emotion'])
   submission.to_csv('submission.csv', index=False)
   print("Submission saved as submission.csv")

if __name__ == "__main__":
   main()
