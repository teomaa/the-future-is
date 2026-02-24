"""
Train a small character-level language model on English adjectives.
Outputs a TFLite model that generates adjective-like words letter by letter.

The model is deliberately small (3 fully-connected layers) so it:
  1. Can eventually run on an ESP32
  2. Produces imperfect, sometimes-gibberish words that sound adjectival
"""

import argparse
import numpy as np
import tensorflow as tf
import os
from tqdm.keras import TqdmCallback

# ---------------------------------------------------------------------------
# 1. Dataset – curated adjectives for "the future is ___"
# ---------------------------------------------------------------------------

def load_adjectives(max_len: int = 9) -> list[str]:
    """Load curated adjectives from curated_adjectives.txt.

    The file contains 1000 adjectives pre-scored for fitness with
    "the future is ___" — evocative, poetic, emotionally resonant words.
    """
    path = os.path.join(os.path.dirname(__file__), "curated_adjectives.txt")
    with open(path) as f:
        words = [w.strip().lower() for w in f if w.strip()]
    return [w for w in words if w.isalpha() and len(w) <= max_len]

# ---------------------------------------------------------------------------
# 2. Character-level encoding
# ---------------------------------------------------------------------------

# Vocabulary: lowercase letters + start token + end token
CHARS = list("abcdefghijklmnopqrstuvwxyz")
START_TOKEN = "^"
END_TOKEN = "$"
VOCAB = [START_TOKEN] + CHARS + [END_TOKEN]
CHAR_TO_IDX = {c: i for i, c in enumerate(VOCAB)}
IDX_TO_CHAR = {i: c for i, c in enumerate(VOCAB)}
VOCAB_SIZE = len(VOCAB)

MAX_WORD_LEN = 9  # max letters (not counting start/end tokens)
SEQ_LEN = MAX_WORD_LEN + 1  # input sequence length (includes start token)


def encode_word(word: str) -> list[int]:
    """Encode a word into a sequence of character indices, with start/end tokens."""
    word = word[:MAX_WORD_LEN]
    return [CHAR_TO_IDX[START_TOKEN]] + [CHAR_TO_IDX[c] for c in word] + [CHAR_TO_IDX[END_TOKEN]]


# ---------------------------------------------------------------------------
# 3. Build training pairs
# ---------------------------------------------------------------------------

def build_dataset(adjectives: list[str]):
    """
    For each adjective, create (input_char, target_char) pairs.
    Input is one-hot encoded current character; target is the next character.
    We train the model to predict the next character given the current one
    plus its position in the sequence (positional info baked into the input).
    """
    X_list = []
    y_list = []

    for word in adjectives:
        encoded = encode_word(word.lower())
        for i in range(len(encoded) - 1):
            # Input: one-hot char + normalized position
            inp = np.zeros(VOCAB_SIZE + 1, dtype=np.float32)
            inp[encoded[i]] = 1.0
            inp[-1] = i / SEQ_LEN  # position signal
            X_list.append(inp)
            y_list.append(encoded[i + 1])

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)

    # Shuffle
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    return X[indices], y[indices]


# ---------------------------------------------------------------------------
# 4. Model definition – 3 fully-connected layers, deliberately small
# ---------------------------------------------------------------------------

def build_model(input_dim: int, output_dim: int,
                 width: int = 48, depth: int = 2) -> tf.keras.Model:
    layers = [tf.keras.layers.InputLayer(input_shape=(input_dim,))]
    for _ in range(depth):
        layers.append(tf.keras.layers.Dense(width, activation="relu"))
    layers.append(tf.keras.layers.Dense(output_dim, activation="softmax"))
    model = tf.keras.Sequential(layers)
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ---------------------------------------------------------------------------
# 5. Training
# ---------------------------------------------------------------------------

def train(epochs: int = 120, batch_size: int = 64,
          width: int = 48, depth: int = 2):
    words = load_adjectives(max_len=MAX_WORD_LEN)
    print(f"Training on {len(words)} unique adjectives (max {MAX_WORD_LEN} chars)")

    X, y = build_dataset(words)
    print(f"Total training pairs: {len(X)}")

    input_dim = VOCAB_SIZE + 1  # one-hot char + position
    model = build_model(input_dim, VOCAB_SIZE, width=width, depth=depth)
    model.summary()

    # Split off a small validation set
    split = int(0.9 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
        callbacks=[TqdmCallback(verbose=1)],
    )

    # Convert to TFLite and save
    os.makedirs("output", exist_ok=True)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    tflite_path = "output/adjective_model.tflite"
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    print(f"Saved TFLite model to {tflite_path} ({len(tflite_model)} bytes)")

    return tflite_path


# ---------------------------------------------------------------------------
# 6. TFLite inference helper
# ---------------------------------------------------------------------------

def load_tflite_model(tflite_path: str):
    """Load a TFLite model and return a ready-to-use interpreter."""
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    return interpreter


def tflite_predict(interpreter, input_data: np.ndarray) -> np.ndarray:
    """Run a single forward pass through the TFLite model."""
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]["index"], input_data)
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]["index"])


# ---------------------------------------------------------------------------
# 7. Generation (for testing / evaluation)
# ---------------------------------------------------------------------------

def generate_word(interpreter, temperature: float = 1.0) -> str:
    """Generate a single word by sampling character by character."""
    input_dim = VOCAB_SIZE + 1
    char_idx = CHAR_TO_IDX[START_TOKEN]
    word = []

    for pos in range(MAX_WORD_LEN):
        inp = np.zeros((1, input_dim), dtype=np.float32)
        inp[0, char_idx] = 1.0
        inp[0, -1] = pos / SEQ_LEN

        probs = tflite_predict(interpreter, inp)[0]

        # Apply temperature
        if temperature != 1.0:
            log_probs = np.log(probs + 1e-10) / temperature
            probs = np.exp(log_probs) / np.sum(np.exp(log_probs))

        char_idx = np.random.choice(len(probs), p=probs)
        char = IDX_TO_CHAR[char_idx]

        if char == END_TOKEN:
            break
        if char == START_TOKEN:
            continue
        word.append(char)

    return "".join(word)


def test_generation(interpreter, n: int = 30):
    """Generate sample words at various temperatures."""
    print("\n--- Sample generated words ---")
    for temp in [0.5, 0.8, 1.0, 1.3]:
        words = [generate_word(interpreter, temperature=temp) for _ in range(n)]
        print(f"\nTemperature {temp}:")
        print("  " + ", ".join(words))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train adjective generator")
    parser.add_argument("--width", type=int, default=48, help="neurons per hidden layer (default: 48)")
    parser.add_argument("--depth", type=int, default=2, help="number of hidden layers (default: 2)")
    parser.add_argument("--epochs", type=int, default=120, help="training epochs (default: 120)")
    parser.add_argument("--batch-size", type=int, default=64, help="batch size (default: 64)")
    args = parser.parse_args()

    tflite_path = train(epochs=args.epochs, batch_size=args.batch_size,
                        width=args.width, depth=args.depth)
    interpreter = load_tflite_model(tflite_path)
    test_generation(interpreter)
