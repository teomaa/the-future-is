// adjective_model_larger256_4_inference.cpp
// Character-by-character word generation matching train.py logic

#include "adjective_model_larger256_4.h"
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <ctime>

using namespace embedded_ml;

// --- Vocabulary (must match train.py) ---
// Index 0: start token '^'
// Index 1-26: 'a'-'z'
// Index 27: end token '$'
static constexpr int VOCAB_SIZE = 28;
static constexpr int INPUT_DIM = 29;  // VOCAB_SIZE + 1 position
static constexpr int START_IDX = 0;
static constexpr int END_IDX = 27;
static constexpr int MAX_WORD_LEN = 9;
static constexpr int SEQ_LEN = MAX_WORD_LEN + 1;

// --- Tunable parameter ---
static constexpr float TEMPERATURE = 0.5f;

static char idx_to_char(int idx) {
    if (idx >= 1 && idx <= 26) return 'a' + (idx - 1);
    return 0;
}

// Sample from a probability distribution
static int sample(const float* probs, int n) {
    float r = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    float cumulative = 0.0f;
    for (int i = 0; i < n; i++) {
        cumulative += probs[i];
        if (r <= cumulative) return i;
    }
    return n - 1;
}

// Generate a single word
static void generate_word(char* out, int max_len) {
    float input[INPUT_DIM];
    float output[VOCAB_SIZE];

    int char_idx = START_IDX;
    int len = 0;

    for (int pos = 0; pos < MAX_WORD_LEN; pos++) {
        // Build input: one-hot char + position
        for (int i = 0; i < INPUT_DIM; i++) input[i] = 0.0f;
        input[char_idx] = 1.0f;
        input[INPUT_DIM - 1] = static_cast<float>(pos) / SEQ_LEN;

        // Run model
        adjective_model_larger256_4Model::Inference(input, output);

        // Apply temperature
        if (TEMPERATURE != 1.0f) {
            float max_logp = -1e30f;
            for (int i = 0; i < VOCAB_SIZE; i++) {
                float lp = logf(output[i] + 1e-10f) / TEMPERATURE;
                output[i] = lp;
                if (lp > max_logp) max_logp = lp;
            }
            float sum = 0.0f;
            for (int i = 0; i < VOCAB_SIZE; i++) {
                output[i] = expf(output[i] - max_logp);  // subtract max for stability
                sum += output[i];
            }
            for (int i = 0; i < VOCAB_SIZE; i++) {
                output[i] /= sum;
            }
        }

        char_idx = sample(output, VOCAB_SIZE);

        if (char_idx == END_IDX) break;
        if (char_idx == START_IDX) continue;

        char c = idx_to_char(char_idx);
        if (c && len < max_len - 1) {
            out[len++] = c;
        }
    }
    out[len] = '\0';
}

int main() {
    srand(static_cast<unsigned>(time(nullptr)));

    std::cout << "the future is..." << std::endl;
    for (int i = 0; i < 30; i++) {
        char word[MAX_WORD_LEN + 1];
        generate_word(word, sizeof(word));
        std::cout << "  " << word << std::endl;
    }

    return 0;
}
