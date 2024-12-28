# predictor.py

import numpy as np
import pandas as pd
import os
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from tqdm import tqdm

class FootballPredictor:
    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.models = {
            'rf': RandomForestClassifier(n_estimators=500, max_depth=30, random_state=42, n_jobs=-1),
            'xgb': XGBClassifier(n_estimators=300, max_depth=15, learning_rate=0.05, random_state=42, 
                                 use_label_encoder=False, eval_metric='mlogloss'),
            'et': ExtraTreesClassifier(n_estimators=500, max_depth=30, random_state=42, n_jobs=-1)
        }
        self.scaler = StandardScaler()

    def train(self, X, y):
        print(f"Shape of X before encoding: {X.shape}")
        print(f"Shape of y before encoding: {y.shape}")
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
        X_scaled = self.scaler.fit_transform(X_train)
        
        # Train individual models
        for name, model in tqdm(self.models.items(), desc="Training individual models"):
            print(f"\nTraining {name}...")
            model.fit(X_scaled, y_train)
            
            y_pred = model.predict(self.scaler.transform(X_test))
            print(f"{name} Accuracy: {accuracy_score(y_test, y_pred):.4f}")
            print(f"{name} F1-Score: {f1_score(y_test, y_pred, average='macro'):.4f}")
            print(f"\nClassification Report for {name}:")
            print(classification_report(y_test, y_pred, target_names=self.label_encoder.classes_))

        # Evaluate the ensemble model
        y_pred_ensemble = self.ensemble_predict(self.scaler.transform(X_test))
        print(f"Ensemble Model Accuracy: {accuracy_score(y_test, y_pred_ensemble):.4f}")
        print(f"Ensemble Model F1-Score: {f1_score(y_test, y_pred_ensemble, average='macro'):.4f}")
        print("\nClassification Report for Ensemble Model:")
        print(classification_report(y_test, y_pred_ensemble, target_names=self.label_encoder.classes_))

    def ensemble_predict(self, X):
        predictions = []
        for model in self.models.values():
            pred = model.predict_proba(X)
            predictions.append(pred)
        
        # Average the probabilities
        avg_pred = np.mean(predictions, axis=0)
        return np.argmax(avg_pred, axis=1)

    def predict(self, X):
        if not self.models:
            raise ValueError("Models have not been trained yet")
        
        # Use self.feature_names instead of scaler's feature_names_in_
        if self.feature_names is not None:
            expected_features = self.feature_names
            if not all(feature in X.columns for feature in expected_features):
                missing_features = set(expected_features) - set(X.columns)
                raise ValueError(f"Missing features in input data: {missing_features}")
            
            # Ensure the order of features matches the order used during training
            X = X[expected_features]
        else:
            print("Warning: No feature names stored. Using all input features.")
        
        X_scaled = self.scaler.transform(X)
        ensemble_pred = self.ensemble_predict(X_scaled)
        ensemble_prob = np.mean([model.predict_proba(X_scaled) for model in self.models.values()], axis=0)
        
        # Ensure predictions and probabilities are in the correct format
        predictions = self.label_encoder.inverse_transform(ensemble_pred)
        probabilities = ensemble_prob.squeeze()  # Remove any extra dimensions
        
        # Ensure predictions and probabilities are lists or arrays
        predictions = np.atleast_1d(predictions)
        probabilities = np.atleast_2d(probabilities)
        
        return predictions, probabilities

    def print_feature_importance(self, X):
        if not self.models:
            raise ValueError("Models have not been trained yet")
        
        feature_names = X.columns
        
        print("\nFeature importance shapes:")
        for name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                print(f"{name}: {model.feature_importances_.shape}")
            else:
                print(f"{name}: No feature importance available")
        print(f"Number of feature names: {len(feature_names)}")
        
        for name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                feature_importance = model.feature_importances_
                sorted_idx = np.argsort(feature_importance)
                print(f"\nFeature Importance ({name.upper()}):")
                for idx in sorted_idx[::-1]:
                    if idx < len(feature_names):
                        print(f"{feature_names[idx]}: {feature_importance[idx]:.4f}")
            else:
                print(f"\n{name.upper()} does not provide feature importance.")

        # Print average feature importance across all models
        print("\nAverage Feature Importance:")
        avg_importance = np.zeros(len(feature_names))
        count = 0
        for name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                if len(model.feature_importances_) == len(feature_names):
                    avg_importance += model.feature_importances_
                    count += 1
        
        if count > 0:
            avg_importance /= count
            sorted_idx = np.argsort(avg_importance)
            for idx in sorted_idx[::-1]:
                print(f"{feature_names[idx]}: {avg_importance[idx]:.4f}")
        else:
            print("No feature importance available for any model.")

    def save_model(self, filepath):
        if not self.models:
            raise ValueError("Model has not been trained yet")
        
        model_data = {
            'label_encoder': self.label_encoder,
            'models': self.models,
            'scaler': self.scaler,
            'feature_names': self.scaler.feature_names_in_
        }
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")


    @classmethod
    def load_split_model(cls, chunks_dir):
        """Load model from split chunks"""
        import json
        import pickle
        
        # Load metadata
        with open(f'{chunks_dir}/metadata.json', 'r') as f:
            metadata = json.load(f)
        
        # Combine chunks
        serialized_model = b''
        for i in range(metadata['num_chunks']):
            chunk_path = f'{chunks_dir}/model_part_{i}'
            with open(chunk_path, 'rb') as f:
                serialized_model += f.read()
        
        # Deserialize model data
        model_data = pickle.loads(serialized_model)
        
        # Create new predictor instance
        predictor = cls()
        predictor.label_encoder = model_data['label_encoder']
        predictor.models = model_data['models']
        predictor.scaler = model_data['scaler']
        predictor.feature_names = model_data.get('feature_names', None)
        
        return predictor
