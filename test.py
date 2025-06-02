
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, LSTM, Dense, Dropout, BatchNormalization, Layer
from sklearn.metrics import roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from tensorflow.keras.optimizers import Adam

class ScaledDotProductAttention(Layer):
    def call(self, query, key, value):
        matmul_qk = tf.matmul(query, key, transpose_b=True)
        dk = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        output = tf.matmul(attention_weights, value)
        return output, attention_weights

custom_objects = {"ScaledDotProductAttention": ScaledDotProductAttention, "Adam": Adam}
tf.keras.utils.get_custom_objects().update(custom_objects)

def parse_args():
    parser = ArgumentParser(description="Specify Input Parameters")
    parser.add_argument('--data_test_cv', type=str, required=True, help='Path to the original test data Parquet file')
    parser.add_argument('--model_save_path', type=str, required=True, help='Path to the trained model .h5 file')
    parser.add_argument('--prediction_save_path', type=str, required=True, help='Path to save the test predictions')
    parser.add_argument('--seed', type=int, default=98765, help='Random seed for reproducibility')
    return parser.parse_args()

def main():
    args = parse_args()

    DATA_TEST_ORI = pd.read_parquet(args.data_test_cv)

    model_save_path = args.model_save_path
    prediction_save_path = args.prediction_save_path
    seed = args.seed

    np.random.seed(seed)
    tf.random.set_seed(seed)

    peptide_features = [col for col in DATA_TEST_ORI.columns if col.startswith('P')]
    mhc_features = [col for col in DATA_TEST_ORI.columns if col.startswith('M')]

    X_test_peptide = DATA_TEST_ORI[peptide_features].values
    X_test_mhc = DATA_TEST_ORI[mhc_features].values

    X_test_peptide = np.expand_dims(X_test_peptide, axis=-1)
    X_test_mhc = np.expand_dims(X_test_mhc, axis=-1)

    try:
        model = load_model(model_save_path, custom_objects=custom_objects, compile=False)
    except Exception as e:
        print(f"Failed to load model with custom objects: {e}")
        model = load_model(model_save_path)

    test_predictions = model.predict([X_test_peptide, X_test_mhc])

    os.makedirs(os.path.dirname(prediction_save_path), exist_ok=True)
    prediction_df = pd.DataFrame(test_predictions, columns=['Prediction'])
    prediction_df.to_parquet(prediction_save_path)


if __name__ == "__main__":
    main()

