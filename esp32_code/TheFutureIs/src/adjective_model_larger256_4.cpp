// adjective_model_larger256_4.cpp
// Auto-generated inference model implementation
// Pure C++ implementation for embedded systems

#include "adjective_model_larger256_4.h"
#include "adjective_model_larger256_4_weights.cpp"
#include "fully_connected.h"










#include "softmax.h"
#include <cstddef>

namespace embedded_ml {

void adjective_model_larger256_4Model::Inference(const float* input, float* output) {

    // Layer 0: FULLY_CONNECTED

    FullyConnected(input, weight_8_sequential_1_dense_1_MatMul, weight_9_sequential_1_dense_1_Relu_sequential_1_dense_1_BiasAdd, buffer_11, 29, 256, ActivationType::RELU);

    // Layer 1: FULLY_CONNECTED

    FullyConnected(buffer_11, weight_7_arith_constant7, weight_2_arith_constant2, buffer_12, 256, 256, ActivationType::RELU);

    // Layer 2: FULLY_CONNECTED

    FullyConnected(buffer_12, weight_6_arith_constant6, weight_1_arith_constant1, buffer_13, 256, 256, ActivationType::RELU);

    // Layer 3: FULLY_CONNECTED

    FullyConnected(buffer_13, weight_5_arith_constant5, weight_0_arith_constant, buffer_14, 256, 256, ActivationType::RELU);

    // Layer 4: FULLY_CONNECTED

    FullyConnected(buffer_14, weight_4_arith_constant4, weight_3_arith_constant3, buffer_15, 256, 28, ActivationType::NONE);

    // Layer 5: SOFTMAX

    Softmax(buffer_15, output, 28);

}


// Intermediate buffer definitions

float adjective_model_larger256_4Model::buffer_11[256];

float adjective_model_larger256_4Model::buffer_12[256];

float adjective_model_larger256_4Model::buffer_13[256];

float adjective_model_larger256_4Model::buffer_14[256];

float adjective_model_larger256_4Model::buffer_15[28];


} // namespace embedded_ml

