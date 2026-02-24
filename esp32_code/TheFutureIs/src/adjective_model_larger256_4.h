// adjective_model_larger256_4.h
// Auto-generated inference model header
// Pure C++ implementation for embedded systems

#ifndef ADJECTIVE_MODEL_LARGER256_4_MODEL_H
#define ADJECTIVE_MODEL_LARGER256_4_MODEL_H

#include <cstddef>

namespace embedded_ml {

class adjective_model_larger256_4Model {
public:
    static constexpr size_t kInputSize = 29;
    static constexpr size_t kOutputSize = 28;


private:
    // Intermediate buffers for layer outputs

    static float buffer_11[256];

    static float buffer_12[256];

    static float buffer_13[256];

    static float buffer_14[256];

    static float buffer_15[28];



public:
    // Run inference on input data
    // input: array of size kInputSize
    // output: array of size kOutputSize (will be filled with results)
    static void Inference(const float* input, float* output);
};

} // namespace embedded_ml

#endif // ADJECTIVE_MODEL_LARGER256_4_MODEL_H

