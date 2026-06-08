"""
    Addestramento Behavioral Cloning con TensorFlow.
        1. Legge tutti il dataset JSON nella cartella 'dataset'
        2. Applica StandardScaler + PCA
        3. Addestra il modello su una rete neurale a tre livelli (con loss pesata per sterzo e frenata).
    24 features: track(19) + speedX + angle + trackPos + rpm + distFromStart
"""

import os
import random

SEED = 42
random.seed(SEED)

# ============ SETUP. Determinismo su Windows e Mac ================
# Fissati i pesi, il comportamento in inferenza è identico tra esecuzioni e tra sistemi operativi (entro i limiti della precisione float32).
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'    # disattiva il OneDNN su Windows (causa non determinismo)
os.environ['TF_DETERMINISTIC_OPS'] = '1'     # forza operazioni deterministiche su CPU.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'     # nasconde i warning per pulizia della Shell

import json
import glob
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import joblib
import tensorflow as tf
from tensorflow import keras

# Settiamo lo stesso seme random per tutte le librerie
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ===============================================================
# ===================== 1. Configurazioni =======================
# ===============================================================
ROOT = os.path.dirname(os.path.abspath(__file__))

# 1.1 Cartelle nella directory del progetto
DATA_DIR     = os.path.join(ROOT, "..", "dataset")
MODEL_DIR    = os.path.join(ROOT, "modelli")
PIPELINE_DIR = os.path.join(ROOT, "preprocessori")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PIPELINE_DIR, exist_ok=True)

PIPELINE_FILE = os.path.join(PIPELINE_DIR, "preprocessing_pipeline.pkl")
MODEL_BEST    = os.path.join(MODEL_DIR, "best_driver_model.keras")
MODEL_LAST    = os.path.join(MODEL_DIR, "last_driver_model.keras")

# Parametri training
BATCH_SIZE          = 128
EPOCHS              = 200
LEARNING_RATE       = 1e-3
EARLY_STOP_PATIENCE = 15
LR_REDUCE_PATIENCE  = 5

# Pesi per la loss pesata. 
# Valori moderati: forza il modello a non ignorare freno e sterzo senza penalizzare eccessivamente l'acceleratore.
STEER_LOSS_WEIGHT = 1.9   # sterzo più critico: errori piccoli → fuoripista
BRAKE_LOSS_WEIGHT = 1.0   # brake è ben rappresentato dunque ha peso uguale ad accel


# =================================================================
# ===================== 1. Caricamento Dati =======================
# =================================================================
def load_training_data(folder=DATA_DIR):
    """
        La funzione carica i dati presenti nella cartella "DATA_DIR" (di default).
        Questa funzione supporta la lettura di file in formato:
        - JSON -> es. i dati grezzi letti dalla raccolta dati
        - JSONL (JSON Lines) -> es. il dataset
    """
    all_features, all_targets = [], []
    json_files = sorted(glob.glob(os.path.join(folder, "*.json")))

    if not json_files:
        raise FileNotFoundError(f"Nessun file JSON in {folder}")

    for fpath in json_files:
        with open(fpath, 'r', encoding='utf-8-sig') as f:
            raw = f.read().strip()
            if not raw:
                continue
            # Prova a leggere i dati come array JSON
            try:
                data = json.loads(raw)
                if not isinstance(data, list):
                    data = None
            except Exception:
                data = None
            # Se fallisce, interpreta come JSON Lines (una riga = un record)
            if data is None:
                data = []
                for line in raw.splitlines():
                    line = line.strip()
                    if line:
                        try:
                            data.append(json.loads(line))
                        except Exception:
                            pass
            if not data:
                print(f"  Attenzione: nessun dato valido in {os.path.basename(fpath)}")
                continue

            for entry in data:
                sens = entry.get("sensors")
                act  = entry.get("actions")
                if not sens or not act:
                    continue

                # Feature: track (19) + speedX + angle + trackPos + rpm + distFromStart
                track = sens.get("track", [200.0] * 19)[:19]
                while len(track) < 19:
                    track.append(200.0)

                feats = track + [
                    sens.get("speedX",        0.0),
                    sens.get("angle",         0.0),
                    sens.get("trackPos",      0.0),
                    sens.get("rpm",           0.0),
                    sens.get("distFromStart", 0.0),
                ]
                targets = [
                    act.get("accel", 0.0),
                    act.get("brake", 0.0),
                    act.get("steer", 0.0),
                ]
                all_features.append(feats)
                all_targets.append(targets)

    if not all_features:
        raise ValueError(f"Nessun frame valido in {folder}")

    X = np.array(all_features, dtype=np.float32)
    y = np.array(all_targets,  dtype=np.float32)

    # Statistiche del dataset per verifica proporzionalità
    accel_pct = (y[:, 0] > 0.10).mean() * 100
    brake_pct = (y[:, 1] > 0.05).mean() * 100
    coast_pct = 100 - accel_pct - brake_pct
    print(f"accel={accel_pct:.1f}%  brake={brake_pct:.1f}%  "
          f"coast={coast_pct:.1f}%  speedX_avg={X[:, 19].mean():.1f} km/h  "
          f"|steer|_avg={np.abs(y[:, 2]).mean():.4f}")

    return X, y


# ============================================================
# ===================== 2. Loss Pesata =======================
# ============================================================
def weighted_mse(y_true, y_pred):
    err_accel = tf.square(y_true[:, 0] - y_pred[:, 0])
    err_brake = tf.square(y_true[:, 1] - y_pred[:, 1])
    err_steer = tf.square(y_true[:, 2] - y_pred[:, 2])

    # Lo sterzo riceve un peso maggiore perché errori piccoli su questo output hanno un grande impatto in pista
    return tf.reduce_mean(
        err_accel
        + BRAKE_LOSS_WEIGHT * err_brake
        + STEER_LOSS_WEIGHT * err_steer
    )


# ===============================================================
# ================== 3. Architettura della NN ===================
# ===============================================================
def build_driver_model(input_dim):
    """
        Rete Neurale (NN) su quattro livelli:
            1. Dense con 256 neuroni + Dropout casuale del 20%.
            2. Dense con 128 neuroni + Dropout casuale del 20%.
            3. Dense con 64 neuroni + Dropout casuale del 10%.
            3. Linear con 3 neuroni (corrispondenti ai parametri di output)
        Nei livelli Dense si utilizza "elu" perché permette di ottenere gradienti più
        adatti ad output continui (come accel e brake).
        Il Dropout decrescentre è pensato per essere più regolarizzato nei layer più
        profondi, e meno nei layer più vicini all'output.
    """
    model = keras.models.Sequential([
        keras.layers.Input(shape=(input_dim,)),

        keras.layers.Dense(256, activation='elu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.2),

        keras.layers.Dense(128, activation='elu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.2),

        keras.layers.Dense(64, activation='elu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.1),

        keras.layers.Dense(3, activation='linear'),
    ])
    return model


# =============================================================
# ========================= 4. Main ===========================
# =============================================================
def main():
    """
        Il main organizza la pipeline di addestramento in cinque fasi:
            1. Caricamento del dataset da DATA_DIR (nel nostro caso dataset/)
            2. Applicazione dello StandardScaler + PCA (99% varianza) per ridurre ridondanza tra feature
            3. Split del dataset in train/test (80/20) con seed fisso (42) per garantire la riproducibilità
            4. Costruzione della rete NN e addestramento
            5. Salvataggio della pipeline di preprocessing e pesi (migliori + ultimi)
    """
    # 4.1. Caricamento dei dati
    print(f"Caricamento dataset da: {DATA_DIR}")
    X, y = load_training_data(DATA_DIR)
    print(f"Campioni: {len(X)}  Features: {X.shape[1]}  Target: {y.shape[1]}")

    # 4.2. StandardScaler + PCA.
    print("\nPreprocessing: StandardScaler + PCA (99% varianza)")
    preproc_pipe = Pipeline([
        ('scale', StandardScaler()),
        ('pca', PCA(n_components=0.99, random_state=SEED, svd_solver='full')),
    ])
    X_proc = preproc_pipe.fit_transform(X)
    print(f"Dimensioni dopo PCA: {X_proc.shape[1]}")

    # 4.3. Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X_proc, y, test_size=0.2, random_state=SEED
    )
    print(f"Train: {X_train.shape[0]}  Test: {X_test.shape[0]}")

    # Salva pipeline di preprocessing
    joblib.dump(preproc_pipe, PIPELINE_FILE)
    print(f"Pipeline salvata in: {PIPELINE_FILE}")

    # 4.4. Costruzione e addestramento del modello
    print("\nCostruzione e addestramento del modello...")

    # Costruzione del modello
    model = build_driver_model(X_train.shape[1])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=weighted_mse,
    )

    # Definizione delle funzioni di callback
    callbacks_list = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=EARLY_STOP_PATIENCE,
            restore_best_weights=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=LR_REDUCE_PATIENCE,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            MODEL_BEST,
            monitor='val_loss',
            save_best_only=True,
        ),
    ]

    # Addestramento
    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks_list,
        shuffle=True,
        verbose=1,
    )

    # 4.5. Salvataggio dei modelli (best + last)
    model.save(MODEL_LAST)
    print(f"\nAddestramento completato. Modello migliore: {MODEL_BEST}")


if __name__ == "__main__":
    main()
