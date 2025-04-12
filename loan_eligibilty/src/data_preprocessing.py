import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_process_data(filepath = r'data\credit.csv'):
   
        df = pd.read_csv(filepath)
      
        if 'Loan_ID' in df.columns:
            df.drop(columns=['Loan_ID'], inplace=True)

        
        df['Loan_Approved'] = df['Loan_Approved'].map({'Y': 1, 'N': 0})

        # Handle missing values
        fill_mode = ['Gender', 'Married', 'Dependents', 'Self_Employed', 'Loan_Amount_Term', 'Credit_History']
        for col in fill_mode:
            df[col].fillna(df[col].mode()[0], inplace=True)

        df['LoanAmount'].fillna(df['LoanAmount'].median(), inplace=True)

        df = pd.get_dummies(df, drop_first=True)

        X = df.drop('Loan_Approved', axis=1)
        y = df['Loan_Approved']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

        return X_train, X_test, y_train, y_test
