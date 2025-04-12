from sklearn.linear_model import LogisticRegression
import joblib

def train_and_save_model(X_train, y_train, model_path='models/loan_model.pkl'):
    
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        joblib.dump(model, model_path)
        
        return model
    
    