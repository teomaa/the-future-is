#include <Arduino.h>
#include <TFT_eSPI.h>
#include "adjective_model_larger256_4.h"
#include <cmath>

using namespace embedded_ml;

// --- Vocabulary (must match train.py) ---
static constexpr int VOCAB_SIZE = 28;
static constexpr int INPUT_DIM = 29;
static constexpr int START_IDX = 0;
static constexpr int END_IDX = 27;
static constexpr int MAX_WORD_LEN = 9;
static constexpr int SEQ_LEN = MAX_WORD_LEN + 1;

// --- Tunable ---
static constexpr float TEMPERATURE = 0.5f;
static constexpr int DELAY_PRESET_MS = 3000;
static constexpr int DELAY_GENERATED_MS = 1200;
static constexpr int KEYSTROKE_MIN_MS = 60;
static constexpr int KEYSTROKE_MAX_MS = 180;
static constexpr int CURSOR_BLINK_MS = 600;

TFT_eSPI tft = TFT_eSPI();

static char idx_to_char(int idx) {
    if (idx >= 1 && idx <= 26) return 'a' + (idx - 1);
    return 0;
}

static int sample(const float* probs, int n) {
    float r = random(0, 10000) / 10000.0f;
    float cumulative = 0.0f;
    for (int i = 0; i < n; i++) {
        cumulative += probs[i];
        if (r <= cumulative) return i;
    }
    return n - 1;
}

static void generate_word(char* out, int max_len) {
    float input[INPUT_DIM];
    float output[VOCAB_SIZE];

    int char_idx = START_IDX;
    int len = 0;

    for (int pos = 0; pos < MAX_WORD_LEN; pos++) {
        for (int i = 0; i < INPUT_DIM; i++) input[i] = 0.0f;
        input[char_idx] = 1.0f;
        input[INPUT_DIM - 1] = static_cast<float>(pos) / SEQ_LEN;

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
                output[i] = expf(output[i] - max_logp);
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

// --- Preset words ---
static const char* PRESETS[] = { "bleak", "bright", "beautiful", "scary", "ai", "ass", "a mystery", "scary", "exciting", "amazing", "delightful", "expensive", "sunny", "hopeful" };
static constexpr int NUM_PRESETS = sizeof(PRESETS) / sizeof(PRESETS[0]);

// Phase: 0 = showing presets, 1 = showing generated words
static int phase = 0;
static int word_idx = 0;
static char displayed[MAX_WORD_LEN + 1] = "";  // what's currently on screen

static int keystroke_delay() {
    return random(KEYSTROKE_MIN_MS, KEYSTROKE_MAX_MS + 1);
}

static void draw_header() {
    tft.setTextDatum(MC_DATUM);
    tft.setTextSize(2);
    tft.drawString("the future is...", tft.width() / 2, tft.height() / 2 - 20);
}

static bool cursor_phase() {
    return (millis() / CURSOR_BLINK_MS) % 2 == 0;
}

static void draw_word_area(const char* word, bool with_cursor = false) {
    int centerX = tft.width() / 2;
    int wordY = tft.height() / 2 + 20;

    tft.fillRect(0, wordY - 15, tft.width(), 30, TFT_BLACK);
    tft.setTextSize(3);
    tft.setTextColor(TFT_WHITE, TFT_BLACK);

    // Draw the word center-aligned (never shifts)
    tft.setTextDatum(MC_DATUM);
    tft.drawString(word, centerX, wordY);

    // Draw cursor as a separate element to the right of the word
    if (with_cursor) {
        int wordW = tft.textWidth(word);
        int cursorX = centerX + wordW / 2;
        tft.setTextDatum(ML_DATUM);
        if (cursor_phase()) {
            tft.drawString("_", cursorX, wordY);
        }
        // When cursor off, area is already cleared — no shift
    }
}

static void blink_delay(int ms) {
    unsigned long start = millis();
    bool last = cursor_phase();
    while (millis() - start < (unsigned long)ms) {
        bool now = cursor_phase();
        if (now != last) {
            draw_word_area(displayed, true);
            last = now;
        }
        delay(10);
    }
}

static void show_word(const char* word, int delay_ms) {
    tft.fillScreen(TFT_BLACK);
    draw_header();
    draw_word_area(word);
    strncpy(displayed, word, MAX_WORD_LEN);
    displayed[MAX_WORD_LEN] = '\0';
    delay(delay_ms);
}

// Typing effect: delete back to common prefix, then type new suffix
static void type_word(const char* word, int hold_ms) {
    int old_len = strlen(displayed);
    int new_len = strlen(word);

    // Find common prefix length
    int common = 0;
    while (common < old_len && common < new_len && displayed[common] == word[common]) {
        common++;
    }

    // Delete from end back to common prefix
    for (int i = old_len; i > common; i--) {
        displayed[i - 1] = '\0';
        draw_word_area(displayed, true);
        blink_delay(keystroke_delay());
    }

    // Type new characters one by one
    for (int i = common; i < new_len; i++) {
        displayed[i] = word[i];
        displayed[i + 1] = '\0';
        draw_word_area(displayed, true);
        blink_delay(keystroke_delay());
    }

    // Hold the completed word with blinking cursor
    blink_delay(hold_ms);
}

void setup() {
    randomSeed(esp_random());
    tft.init();
    tft.setRotation(1);
    tft.fillScreen(TFT_BLACK);
    tft.setTextColor(TFT_WHITE, TFT_BLACK);
}

void loop() {
    if (phase == 0) {
        show_word(PRESETS[word_idx], DELAY_PRESET_MS);
        word_idx++;
        if (word_idx >= NUM_PRESETS) {
            phase = 1;
            word_idx = 0;
            // Transition: draw header once, then typing takes over
            tft.fillScreen(TFT_BLACK);
            draw_header();
            draw_word_area(displayed);
        }
    } else {
        char word[MAX_WORD_LEN + 1];
        generate_word(word, sizeof(word));
        type_word(word, DELAY_GENERATED_MS);
        word_idx++;
        if (word_idx >= NUM_PRESETS) {
            phase = 0;
            word_idx = 0;
            displayed[0] = '\0';
        }
    }
}
