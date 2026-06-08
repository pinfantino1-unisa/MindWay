"""
    Guida autonoma con Behavioral Cloning.
        1. Carica la pipeline di preprocessing e il modello addestrato (best).
        2. Guida in TORCS
    24 features: track(19) + speedX + angle + trackPos + rpm + distFromStart
"""

import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
import joblib
import snakeoil3_jm2 as snakeoil3

# ===============================================================
# ===================== 1. Configurazioni =======================
# ===============================================================
ROOT = os.path.dirname(os.path.abspath(__file__))

# 1.1. Definizione delle cartelle
PIPELINE_FILE = os.path.join(ROOT, "preprocessori", "preprocessing_pipeline.pkl")
MODEL_FILE    = os.path.join(ROOT, "modelli", "best_driver_model.keras")

# 1.2. Parametri Anti-stallo
STALL_SPEED_THRESH = 15.0    # soglia bassa (15 km/h) per intervenire il meno possibile sul comportamento del modello
STALL_ACCEL_FORCE  = 0.6

# 1.3. Smoothing sterzo per ridurre le oscillazioni (frame-by-frame)
STEERING_SMOOTH = 0.6     # 0.6 = 60% del nuovo segnale per frame → risposta pronta ma senza jitter.

# 1.4. Logica cambio marcia
UPSHIFT_RPM    = 14500
DOWNSHIFT_RPM  = 7500
MAX_GEAR       = 6
LOW_SPEED_GEAR = 50   # sotto questa velocità (km/h) forza marcia 1


# ===============================================================
# ==================== 2. Policy Inferenza ======================
# ===============================================================
class FastPolicy:
    """
        Questa funzione definisce la Policy di Inferenza della macchina utilizzando tf.function
    """
    def __init__(self, model):
        self.model = model
        self._compiled = tf.function(self._call, reduce_retracing=True)   # reduce_retracing=True evita ricompilazioni inutili a runtime

    def _call(self, x):
        return self.model(x, training=False)

    def __call__(self, x):
        return self._compiled(x)


# ===============================================================
# ==================== 3. Vettore di Stato ======================
# ===============================================================
def build_state_vector(telemetry):
    """
        Questa funzione costruisce il vettore di stato per la guida autonoma della macchina.
        Il vettore di stato deve corrispondere ESATTAMENTE all'ordine delle feature definito in s2.
    """
    track = telemetry.get('track', [200.0] * 19)[:19]
    while len(track) < 19:
        track.append(200.0)

    vec = track + [
        telemetry.get('speedX',        0.0),
        telemetry.get('angle',         0.0),
        telemetry.get('trackPos',      0.0),
        telemetry.get('rpm',           0.0),
        telemetry.get('distFromStart', 0.0),
    ]
    return np.array(vec, dtype=np.float32).reshape(1, -1)


# ===============================================================
# ================== 4. Logica Cambio Marcia ====================
# ===============================================================
def update_gear(current_gear, rpm, speed_kmh):
    """
        Questa funzione definisce la logica di cambio marcia (indipendente dal modello).
    """
    if speed_kmh < LOW_SPEED_GEAR:
        return 1
    if rpm > UPSHIFT_RPM and current_gear < MAX_GEAR:
        return current_gear + 1
    if rpm < DOWNSHIFT_RPM and current_gear > 1:
        return current_gear - 1
    return current_gear


# ============================================================
# ====================== 5. Main =============================
# ============================================================
def main():
    """
        La funzione carica il modello e la pipeline di preprocessing (StandardScaler + PCA).
        Dopodiché, gestisce il ciclo di guida autonoma in real-time in cinque fasi. 
        Per ogni frame il main:
            1. Legge i 24 sensori da TORCS via comunicazione UDP (snakeoil3)
            2. Applica in ordine prepocessing -> inferenza e predizione [accel, brake, steer]
            3. Corregge lo sterzo nell'ultima curva (D=3230-3300m) con rate limiter per compensare il covariate shift del dataset in quella zona
            4. Applica anti-stallo, mutua esclusione freno/gas e cambio marcia rule-based
            5. Invia i comandi a TORCS
    """
    # --- 5.0. Caricamento del modello e pipeline di preprocessing ---
    print("Caricamento modello e pipeline di preprocessing...")
    preproc = joblib.load(PIPELINE_FILE)
    model   = tf.keras.models.load_model(MODEL_FILE, compile=False)
    policy  = FastPolicy(model)

    client = snakeoil3.Client(p=3001, vision=False)
    print("\nModalita' autonoma attiva (Behavioral Cloning)")
    print("Premi Ctrl+C per arrestare.\n")

    prev_steer = 0.0

    try:
        while True:
            # --- 5.1. Lettura dei 24 sensori da SnakeOil via UDP ---
            client.get_servers_input()
            tele = client.S.d

            # --- 5.2. fase di Preprocessing + Inferenza ---

            # 5.2.1. Costruisce il vector state e applica il preprocessing
            raw_state    = build_state_vector(tele)
            proc_state   = preproc.transform(raw_state)
            tensor_state = tf.constant(proc_state, dtype=tf.float32)

            # 5.2.2. Applica la policy di Inferenza: il modello predice [accel, brake, steer]
            pred     = policy(tensor_state).numpy().flatten()
            throttle = float(np.clip(pred[0],  0.0,  1.0))
            braking  = float(np.clip(pred[1],  0.0,  1.0))
            steering = float(np.clip(pred[2], -1.0,  1.0))

            # Lettura sensori
            speed_kmh = tele.get('speedX', 0.0)
            distance  = tele.get('distFromStart', 0.0)

            # --- 5.3. Smoothing sterzo con patch ultima curva (D=3230-3300m) ---
            if 3230 <= distance <= 3300:
                
                # Policy di sterzata all'ultima curva
                if distance <= 3269:     
                    target = max(0.0, min(0.82, steering))  # Quando raggiunge la curva, sterza a sinistra (D=3230-3269m)
                else:                    
                    target = 0.0                            # Dopo la curva, continua dritto per evitare drift (D=3269-3300m)
                
                # Rate limiter: limita la variazione massima dello sterzo a 0.08 per frame (~0.0.25s) per evitare drift
                max_delta = 0.08
                delta = target - prev_steer
                if abs(delta) > max_delta:
                    steering = prev_steer + (max_delta if delta > 0 else -max_delta)
                else:
                    steering = target
            else:
                steering = STEERING_SMOOTH * steering + (1.0 - STEERING_SMOOTH) * prev_steer

            prev_steer = steering

            # --- 5.4. Ulteriori correzioni logiche ---
            
            # 5.4.1. Anti-stallo: solo per velocità molto basse (macchina ferma o quasi).
                # NON si attiva in retromarcia (speed_kmh < 0) per non peggiorare la situazione.
                # NON modifica lo sterzo: il modello gestisce la direzione anche a bassa velocità.
            if 0.0 <= speed_kmh < STALL_SPEED_THRESH:
                throttle = max(throttle, STALL_ACCEL_FORCE)
                braking  = 0.0

            # 5.4.2. Mutua esclusione freno/acceleratore.
                # Vincolo fisico: un pilota reale non frena e accelera al massimo contemporaneamente.
                # Il modello produce valori continui e può avere transizioni sovrapposte: questa regola le risolve in modo deterministico.
            if braking > 0.05:
                throttle = 0.0

            # 5.4.3. Cambio marcia (logica rule-based, non influenzata dal modello)
            current_gear = tele.get('gear', 1)
            rpm          = tele.get('rpm',  0)
            new_gear     = update_gear(current_gear, rpm, speed_kmh)

            # --- 5.5. Invia comandi a TORCS ---
            client.R.d.update({
                'steer': steering,
                'accel': throttle,
                'brake': braking,
                'gear': new_gear,
            })
            client.respond_to_server()

            # Log sintetico ogni 10 frame
            if tele.get('ticks', 0) % 10 == 0:
                print(
                    f"\rS:{steering:+5.2f} A:{throttle:4.2f} B:{braking:4.2f} "
                    f"G:{new_gear}  V:{speed_kmh:6.1f}km/h  D:{distance:6.0f}m",
                    end="", flush=True,
                )

    # Eccezione: termina il ciclo principale e chiude la socket 3001
    except KeyboardInterrupt:
        print("\n\nArresto richiesto.")

    # Invio dei dati a TORCS per resettare lo stato della macchina quando temrina la corsa
    finally:
        client.R.d.update({'steer': 0.0, 'accel': 0.0, 'brake': 1.0, 'gear': 0})
        client.respond_to_server()
        client.shutdown()
        print("Veicolo fermato e connessione chiusa.")


if __name__ == "__main__":
    main()
