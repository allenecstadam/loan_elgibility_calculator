from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def evaluate_model(model, X_test, y_test):
    
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        conf_matrix = confusion_matrix(y_test, predictions)
        report = classification_report(y_test, predictions)

        return {
            "accuracy": accuracy,
            "confusion_matrix": conf_matrix,
            "classification_report": report
        }
