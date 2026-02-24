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
static constexpr int DELAY_PRESET_MS = 2400;
static constexpr int DELAY_GENERATED_MS = 1200;
static constexpr int KEYSTROKE_MIN_MS = 60;
static constexpr int KEYSTROKE_MAX_MS = 180;
static constexpr int KEYSTROKE_PRESET_MS = 90;
static constexpr int CURSOR_BLINK_MS = 600;

// --- Layout (portrait 135x240) ---
static constexpr int HEADER_Y = 30;
static constexpr int HEADER_LINE2_Y = 55;
static constexpr int WORD_START_Y = 110;
static constexpr int WORD_LINE_H = 32;
static constexpr int SCREEN_PAD = 4;

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
static char displayed[MAX_WORD_LEN + 1] = "";
static bool using_serif = false;

static void apply_serif_font() {
    tft.setFreeFont(&FreeSerif18pt7b);
    tft.setTextSize(1);
}

static void apply_default_font() {
    tft.setFreeFont(NULL);
    tft.setTextSize(3);
}

static int keystroke_delay() {
    return random(KEYSTROKE_MIN_MS, KEYSTROKE_MAX_MS + 1);
}

static void draw_header() {
    tft.setFreeFont(NULL);
    tft.setTextSize(2);
    tft.setTextColor(TFT_WHITE, TFT_BLACK);
    tft.setTextDatum(ML_DATUM);
    tft.drawString("the future", SCREEN_PAD, HEADER_Y);
    tft.drawString("is...", SCREEN_PAD, HEADER_LINE2_Y);
}

static bool cursor_phase() {
    return (millis() / CURSOR_BLINK_MS) % 2 == 0;
}

// Draw word wrapped across lines, with optional blinking cursor.
// Cursor is always accounted for in line-breaking and centering.
static void draw_word_area(const char* word, bool with_cursor = false) {
    int maxW = tft.width() - SCREEN_PAD * 2;
    int centerX = tft.width() / 2;

    // Clear word area
    tft.fillRect(0, WORD_START_Y - 20, tft.width(), tft.height() - (WORD_START_Y - 20), TFT_BLACK);
    tft.setTextColor(TFT_WHITE, TFT_BLACK);

    if (using_serif) apply_serif_font(); else apply_default_font();

    // Build the full string including cursor for measurement
    int len = strlen(word);
    char full[MAX_WORD_LEN + 2];
    memcpy(full, word, len);
    if (with_cursor) {
        full[len] = '_';
        full[len + 1] = '\0';
    } else {
        full[len] = '\0';
    }
    int fullLen = strlen(full);

    if (fullLen == 0) return;

    // Break into lines based on full string (word + cursor)
    int lineStart = 0;
    int lineNum = 0;
    char lineBuf[MAX_WORD_LEN + 2];

    while (lineStart < fullLen) {
        int lineEnd = lineStart + 1;
        while (lineEnd <= fullLen) {
            memcpy(lineBuf, full + lineStart, lineEnd - lineStart);
            lineBuf[lineEnd - lineStart] = '\0';
            if (tft.textWidth(lineBuf) > maxW) {
                lineEnd--;
                break;
            }
            lineEnd++;
        }
        if (lineEnd > fullLen) lineEnd = fullLen;
        if (lineEnd == lineStart) lineEnd = lineStart + 1;

        int lineLen = lineEnd - lineStart;
        memcpy(lineBuf, full + lineStart, lineLen);
        lineBuf[lineLen] = '\0';

        int lineY = WORD_START_Y + lineNum * WORD_LINE_H;

        // Check if this line contains the cursor (last char is '_' and it's the last line segment)
        bool cursorOnThisLine = with_cursor && (lineEnd == fullLen);

        if (cursorOnThisLine) {
            // Draw the line centered including cursor space
            tft.setTextDatum(MC_DATUM);
            // Draw word chars only (not the _), but positioned for full width
            int fullLineW = tft.textWidth(lineBuf);
            int startX = centerX - fullLineW / 2;

            // Separate word part and cursor
            char wordPart[MAX_WORD_LEN + 2];
            memcpy(wordPart, lineBuf, lineLen - 1);
            wordPart[lineLen - 1] = '\0';

            tft.setTextDatum(ML_DATUM);
            tft.drawString(wordPart, startX, lineY);

            if (cursor_phase()) {
                int wordPartW = tft.textWidth(wordPart);
                tft.drawString("_", startX + wordPartW, lineY - (using_serif ? 3 : 0));
            }
        } else {
            tft.setTextDatum(MC_DATUM);
            tft.drawString(lineBuf, centerX, lineY);
        }

        lineStart = lineEnd;
        lineNum++;
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

static void type_word(const char* word, int hold_ms, bool random_strokes, bool use_serif) {
    int old_len = strlen(displayed);
    int new_len = strlen(word);

    int common = 0;
    if (using_serif == use_serif) {
        while (common < old_len && common < new_len && displayed[common] == word[common]) {
            common++;
        }
    }

    // Delete from end back to common prefix
    for (int i = old_len; i > common; i--) {
        displayed[i - 1] = '\0';
        draw_word_area(displayed, true);
        blink_delay(random_strokes ? keystroke_delay() : KEYSTROKE_PRESET_MS);
    }

    // Switch font
    using_serif = use_serif;

    if (common > 0) {
        draw_word_area(displayed, true);
    }

    // Type new characters one by one
    for (int i = common; i < new_len; i++) {
        displayed[i] = word[i];
        displayed[i + 1] = '\0';
        draw_word_area(displayed, true);
        blink_delay(random_strokes ? keystroke_delay() : KEYSTROKE_PRESET_MS);
    }

    blink_delay(hold_ms);
}

void setup() {
    randomSeed(esp_random());
    tft.init();
    tft.setRotation(0);  // Portrait, 0deg
    tft.fillScreen(TFT_BLACK);
    tft.setTextColor(TFT_WHITE, TFT_BLACK);
    draw_header();
}

void loop() {
    if (phase == 0) {
        type_word(PRESETS[word_idx], DELAY_PRESET_MS, false, true);
        word_idx++;
        if (word_idx >= NUM_PRESETS) {
            phase = 1;
            word_idx = 0;
        }
    } else {
        char word[MAX_WORD_LEN + 1];
        generate_word(word, sizeof(word));
        type_word(word, DELAY_GENERATED_MS, true, false);
        word_idx++;
        if (word_idx >= NUM_PRESETS) {
            phase = 0;
            word_idx = 0;
        }
    }
}
