from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

def train_model(X, y):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize and train the model
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    
    # Return the trained model and test sets
    return model, X_test, y_test

def evaluate_model(model, X_test, y_test):
    # Ensure model is trained and predictions are valid
    if not hasattr(model, 'predict'):
        raise ValueError("The model has not been trained or is not valid.")
    
    # Predict using the model
    predictions = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    auc_roc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    
    # Print and return metrics
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"AUC-ROC: {auc_roc:.4f}")
    return accuracy, precision, recall, auc_roc
