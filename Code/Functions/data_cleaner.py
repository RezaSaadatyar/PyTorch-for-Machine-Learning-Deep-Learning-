import copy
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

class DataCleaner:
    def __init__(self, dataset):
        self.data = copy.deepcopy(dataset)
        self.label_encoders = {}  # To store label encoders for categorical columns

    def drop_missing(self):
        """Remove rows with missing values."""
        print("Removing rows with missing values (applied to all columns).")
        return self.data.dropna()

    def fill_mean(self):
        """Impute missing values with the mean (for numerical columns)."""
        numerical_cols = self.data.select_dtypes(include=['number']).columns
        self.data[numerical_cols] = self.data[numerical_cols].fillna(self.data[numerical_cols].mean())
        print(f"NaN values filld with mean in numerical columns: {list(numerical_cols)}")
        return self.data

    def fill_median(self):
        """Impute missing values with the median (for numerical columns)."""
        numerical_cols = self.data.select_dtypes(include=['number']).columns
        self.data[numerical_cols] = self.data[numerical_cols].fillna(self.data[numerical_cols].median())
        print(f"NaN values filld with median in numerical columns: {list(numerical_cols)}")
        return self.data

    def fill_mode(self):
        """Impute missing values with the mode (for both numerical and categorical columns)."""
        for col in self.data.columns:
            self.data[col] = self.data[col].fillna(self.data[col].mode()[0])
        print(f"NaN values filld with mode in all columns: {list(self.data.columns)}")
        return self.data

    def fill_forward(self):
        """Forward fill missing values (for both numerical and categorical columns)."""
        self.data = self.data.fillna(method='ffill')
        print(f"All NaN values have been forward filled in every column.: {list(self.data.columns)}")
        return self.data

    def fill_backward(self):
        """Backward fill missing values (for both numerical and categorical columns)."""
        self.data = self.data.fillna(method='bfill')
        print(f"All NaN values have been backward filled in every column: {list(self.data.columns)}")
        return self.data

    def fill_interpolate(self, method='linear'):
        """Interpolate missing values (for numerical columns)."""
        numerical_cols = self.data.select_dtypes(include=['number']).columns
        self.data[numerical_cols] = self.data[numerical_cols].interpolate(method=method)
        print(f"NaN values interpolated in numerical columns: {list(numerical_cols)}")
        return self.data

    def fill_knn(self, n_neighbors=2):
        """Impute missing values using K-Nearest Neighbors (for numerical columns)."""
        numerical_cols = self.data.select_dtypes(include=['number']).columns
        imputer = KNNImputer(n_neighbors=n_neighbors)
        self.data[numerical_cols] = imputer.fit_transform(self.data[numerical_cols])
        print(f"NaN values filled using KNN in numerical columns: {list(numerical_cols)}")
        return self.data

    # def fill_predictive(self, target_column):
    #     """Impute missing values using predictive modeling (Random Forest)."""
    #     # Convert categorical target column to numerical if necessary
    #     if self.data[target_column].dtype == 'object':
    #         le = LabelEncoder()
    #         self.data[target_column] = le.fit_transform(self.data[target_column].astype(str))
    #         self.label_encoders[target_column] = le
    #         print(f"Converted categorical column '{target_column}' to numerical for imputation.")

    #     # Separate data into missing and non-missing
    #     missing_data = self.data[self.data[target_column].isna()]
    #     non_missing_data = self.data.dropna()

    #     # Features and target
    #     X = non_missing_data.drop(columns=[target_column])
    #     y = non_missing_data[target_column]

    #     # Train model
    #     if self.data[target_column].dtype == 'object':
    #         model = RandomForestClassifier()
    #     else:
    #         model = RandomForestRegressor()
    #     model.fit(X, y)

    #     # Predict missing values
    #     X_missing = missing_data.drop(columns=[target_column])
    #     predicted_values = model.predict(X_missing)

    #     # Fill missing values
    #     self.data.loc[self.data[target_column].isna(), target_column] = predicted_values

    #     # Convert numerical target column back to categorical if necessary
    #     if target_column in self.label_encoders:
    #         self.data[target_column] = self.label_encoders[target_column].inverse_transform(self.data[target_column].astype(int))
    #         print(f"Converted numerical column '{target_column}' back to categorical after imputation.")

    #     print(f"NaN values imputed using predictive modeling in column: '{target_column}'")
        return self.data