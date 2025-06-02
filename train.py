import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, LSTM, Dense, Dropout, BatchNormalization, Layer
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from argparse import ArgumentParser
import parmap
from amorprot import AmorProt

def parse_args():
    parser = ArgumentParser(description="Specifying Input Parameters")
    parser.add_argument('--data_test_ori', type=str, required=True, help='Path to the original test data parquet file')
    parser.add_argument('--data_train_cv', type=str, required=True, help='Path to the training data parquet file')
    parser.add_argument('--data_test_cv', type=str, required=True, help='Path to the cross-validation test data parquet file')
    parser.add_argument('--model_save_path', type=str, required=True, help='Path to save the trained model')
    parser.add_argument('--prediction_save_path', type=str, required=True, help='Path to save the test predictions')
    parser.add_argument('--learning_curves_save_path', type=str, required=True, help='Path to save the learning curves plot')
    parser.add_argument('--attention_map_save_path', type=str, required=True, help='Path to save the attention maps')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training')
    parser.add_argument('--seed', type=int, default=98765, help='Random seed for reproducibility')
    return parser.parse_args()

args = parse_args()

DATA_TEST_ORI = pd.read_parquet(args.data_test_ori)
DATA_TRAIN_CV = pd.read_parquet(args.data_train_cv)
DATA_TEST_CV = pd.read_parquet(args.data_test_cv)

model_save_path = args.model_save_path
prediction_save_path = args.prediction_save_path
learning_curves_save_path = args.learning_curves_save_path
attention_map_save_path = args.attention_map_save_path

epochs = args.epochs
batch_size = args.batch_size
seed = args.seed

np.random.seed(seed)
tf.random.set_seed(seed)

def data_representation(data):
    ap = AmorProt(maccs=True, ecfp4=True, ecfp6=True, rdkit=True)
    TCR_list = parmap.map(make_fp, [[ap, sq] for sq in data['Peptide'].tolist()], pm_pbar=True, pm_processes=20)
    TCR_list = pd.DataFrame(np.array(TCR_list), columns=['P{}'.format(i) for i in range(1, 4264)])

    PEP_list = parmap.map(make_fp, [[ap, sq] for sq in data['MHC'].tolist()], pm_pbar=True, pm_processes=20)
    PEP_list = pd.DataFrame(np.array(PEP_list), columns=['M{}'.format(i) for i in range(1, 4264)])

    return pd.concat([TCR_list, PEP_list], axis=1)

def make_fp(inputs):
    ap, sq = inputs
    return ap.fingerprint(sq).tolist()

DATA_TEST_ORI_y = DATA_TEST_ORI[["Label"]]
DATA_TRAIN_CV_columns = DATA_TRAIN_CV.drop(['Label'], axis=1).columns
DATA_TEST_CV = pd.concat([DATA_TEST_CV[DATA_TRAIN_CV_columns], DATA_TEST_ORI_y], axis=1)

data_train = DATA_TRAIN_CV.copy()
data_test = DATA_TEST_CV.copy()

class ScaledDotProductAttention(Layer):
    def call(self, query, key, value):
        matmul_qk = tf.matmul(query, key, transpose_b=True)
        dk = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        output = tf.matmul(attention_weights, value)
        return output, attention_weights

def create_model(input_shape_peptide, input_shape_mhc):
    peptide_input = Input(shape=input_shape_peptide, name='peptide_input')
    x_peptide = Conv1D(64, 3, padding='same', activation='relu')(peptide_input)
    x_peptide = BatchNormalization()(x_peptide)
    x_peptide = Dropout(0.3)(x_peptide)

    x_peptide = Conv1D(128, 3, padding='same', activation='relu')(x_peptide)
    x_peptide = BatchNormalization()(x_peptide)
    x_peptide = Dropout(0.3)(x_peptide)

    mhc_input = Input(shape=input_shape_mhc, name='mhc_input')
    x_mhc = Conv1D(64, 3, padding='same', activation='relu')(mhc_input)
    x_mhc = BatchNormalization()(x_mhc)
    x_mhc = Dropout(0.3)(x_mhc)

    x_mhc = Conv1D(128, 3, padding='same', activation='relu')(x_mhc)
    x_mhc = BatchNormalization()(x_mhc)
    x_mhc = Dropout(0.3)(x_mhc)

    attention_layer = ScaledDotProductAttention()
    query = Dense(128)(x_peptide)
    key = Dense(128)(x_mhc)
    value = Dense(128)(x_mhc)
    attention_output, attention_scores = attention_layer(query, key, value)
    
    attention_output = MaxPooling1D(pool_size=2)(attention_output)
    attention_output = Dropout(0.3)(attention_output)

    x = LSTM(128, return_sequences=True)(attention_output)
    x = Dropout(0.3)(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = LSTM(128)(x)
    x = Dropout(0.3)(x)

    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=[peptide_input, mhc_input], outputs=outputs)
    attention_model = Model(inputs=[peptide_input, mhc_input], outputs=attention_scores)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='binary_crossentropy', metrics=[tf.keras.metrics.AUC()])
    return model, attention_model

peptide_features = [col for col in data_train.columns if col.startswith('P')]
mhc_features = [col for col in data_train.columns if col.startswith('M')]

X_train_peptide = data_train[peptide_features].values
X_train_mhc = data_train[mhc_features].values
y_train = data_train['Label'].values

X_test_peptide = data_test[peptide_features].values
X_test_mhc = data_test[mhc_features].values
y_test = data_test['Label'].values

X_train_peptide = np.expand_dims(X_train_peptide, axis=-1)
X_train_mhc = np.expand_dims(X_train_mhc, axis=-1)
X_test_peptide = np.expand_dims(X_test_peptide, axis=-1)
X_test_mhc = np.expand_dims(X_test_mhc, axis=-1)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
best_auc = 0.0
best_model = None
attention_maps = []
history_list = []

class_weight = {0: 1, 1: (len(y_train) - np.sum(y_train)) / np.sum(y_train)}

early_stopping = EarlyStopping(monitor='val_loss', patience=3)

for train_index, val_index in skf.split(X_train_peptide, y_train):
    X_tr_peptide, X_val_peptide = X_train_peptide[train_index], X_train_peptide[val_index]
    X_tr_mhc, X_val_mhc = X_train_mhc[train_index], X_train_mhc[val_index]
    y_tr, y_val = y_train[train_index], y_train[val_index]

    model, attention_model = create_model((X_tr_peptide.shape[1], 1), (X_tr_mhc.shape[1], 1))
    history = model.fit([X_tr_peptide, X_tr_mhc], y_tr, epochs=epochs, batch_size=batch_size,
                        validation_data=([X_val_peptide, X_val_mhc], y_val), callbacks=[early_stopping], 
                        class_weight=class_weight)
    history_list.append(history)

    val_predictions = model.predict([X_val_peptide, X_val_mhc])
    val_auc = roc_auc_score(y_val, val_predictions)
    if val_auc > best_auc:
        best_auc = val_auc
        best_model = model
        test_predictions = model.predict([X_test_peptide, X_test_mhc])
        attention_maps = attention_model.predict([X_test_peptide, X_test_mhc])

if best_model:
    best_model.save(model_save_path)

prediction_df = pd.DataFrame(test_predictions, columns=['Prediction'])
prediction_df.to_parquet(prediction_save_path)

def plot_learning_curves(history, save_path=None):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    if 'auc' in history.history and 'val_auc' in history.history:
        plt.plot(history.history['auc'], label='Train AUC')
        plt.plot(history.history['val_auc'], label='Validation AUC')
    plt.title('AUC')
    plt.legend()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

for history in history_list:
    plot_learning_curves(history, save_path=learning_curves_save_path)

def save_attention_map(index, attention_map_save_path, peptide_features, mhc_features, attention_map):
    attention_map_shape = attention_map.shape
    if attention_map_shape == (len(peptide_features), len(mhc_features)):
        filename = f'{attention_map_save_path}/attention_map_{index}.txt'
        np.savetxt(filename, attention_map, delimiter='\t', fmt='%f')
    else:
        print(f"Attention map shape {attention_map_shape} does not match expected shape {(len(peptide_features), len(mhc_features))}")

for i, attention_map in enumerate(attention_maps):
    save_attention_map(i, attention_map_save_path, peptide_features, mhc_features, attention_map)

def find_optimal_cutoff(y_true, y_pred_probs, desired_specificity=0.98):
    thresholds = np.arange(0.0, 1.0, 0.01)
    best_threshold = 0.5
    best_sensitivity = 0.0
    
    for threshold in thresholds:
        y_pred = (y_pred_probs >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        specificity = tn / (tn + fp)
        sensitivity = tp / (tp + fn)
        
        if specificity >= desired_specificity and sensitivity > best_sensitivity:
            best_sensitivity = sensitivity
            best_threshold = threshold
            
    return best_threshold, best_sensitivity

test_probabilities = best_model.predict([X_test_peptide, X_test_mhc])
test_probabilities = test_probabilities.ravel()  # Flatten the probabilities array

optimal_cutoff, optimal_sensitivity = find_optimal_cutoff(y_test, test_probabilities)

print(f'Optimal cutoff for desired specificity: {optimal_cutoff}')
print(f'Sensitivity at optimal cutoff: {optimal_sensitivity}')

test_predictions_optimal = (test_probabilities >= optimal_cutoff).astype(int)

prediction_df_optimal = pd.DataFrame(test_predictions_optimal, columns=['Prediction'])
prediction_df_optimal.to_parquet(f'{prediction_save_path}_optimal_cutoff.parquet')
