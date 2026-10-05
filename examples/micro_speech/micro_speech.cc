/*
 * TensorFlow Lite for Microcontrollers (TFLM) - MicroSpeech Complete Test Suite
 * 
 * Converted from C++ GTest unit test suite to a standalone Arduino Sketch.
 * Performs two categories of tests:
 *   1. Preprocessor Unit Tests (30ms clips -> Spectrogram feature math validation)
 *   2. End-to-End Classifier Tests (1000ms clips -> Audio category prediction)
 *
 * Requirements:
 *   - Board with 128KB+ SRAM (ESP32, Teensy 4.x, Raspberry Pi Pico, nRF52840, etc.)
 */

#include <Arduino.h>
#include <algorithm>
#include <cstdint>
#include <iterator>

// TensorFlow Lite for Microcontrollers Core Headers
#include "tensorflow/lite/micro/c/common.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"

// Model Definitions & Settings
#include "examples/micro_speech/models/micro_speech_models_data.h"

// Test Data Headers
#include "examples/micro_speech/testdata/micro_speech_test_data.h"

namespace {

// Reserve 64 KB memory arena for TFLM model execution and tensor allocations
constexpr size_t kArenaSize = 1024 * 64;
alignas(16) uint8_t g_arena[kArenaSize];

// Feature array buffer to store calculated 40-frequency spectra across time frames
using Features = int8_t[kFeatureCount][kFeatureSize];
Features g_features;

// Audio window calculations based on model sample frequency (16 kHz)
constexpr int kAudioSampleDurationCount =
    kFeatureDurationMs * kAudioSampleFrequency / 1000; // 30ms = 480 samples
constexpr int kAudioSampleStrideCount =
    kFeatureStrideMs * kAudioSampleFrequency / 1000;   // 20ms = 320 samples

// Op Resolvers specifying operator counts used by each model
using MicroSpeechOpResolver = tflite::MicroMutableOpResolver<4>;
using AudioPreprocessorOpResolver = tflite::MicroMutableOpResolver<18>;

// Register operators required for the MicroSpeech Neural Network Classifier
TfLiteStatus RegisterSpeechOps(MicroSpeechOpResolver& op_resolver) {
  TF_LITE_ENSURE_STATUS(op_resolver.AddReshape());
  TF_LITE_ENSURE_STATUS(op_resolver.AddFullyConnected());
  TF_LITE_ENSURE_STATUS(op_resolver.AddDepthwiseConv2D());
  TF_LITE_ENSURE_STATUS(op_resolver.AddSoftmax());
  return kTfLiteOk;
}

// Register custom DSP operators required for Audio Feature Extraction (Spectrogram generation)
TfLiteStatus RegisterPreprocessorOps(AudioPreprocessorOpResolver& op_resolver) {
  TF_LITE_ENSURE_STATUS(op_resolver.AddReshape());
  TF_LITE_ENSURE_STATUS(op_resolver.AddCast());
  TF_LITE_ENSURE_STATUS(op_resolver.AddStridedSlice());
  TF_LITE_ENSURE_STATUS(op_resolver.AddConcatenation());
  TF_LITE_ENSURE_STATUS(op_resolver.AddMul());
  TF_LITE_ENSURE_STATUS(op_resolver.AddAdd());
  TF_LITE_ENSURE_STATUS(op_resolver.AddDiv());
  TF_LITE_ENSURE_STATUS(op_resolver.AddMinimum());
  TF_LITE_ENSURE_STATUS(op_resolver.AddMaximum());
  TF_LITE_ENSURE_STATUS(op_resolver.AddWindow());
  TF_LITE_ENSURE_STATUS(op_resolver.AddFftAutoScale());
  TF_LITE_ENSURE_STATUS(op_resolver.AddRfft());
  TF_LITE_ENSURE_STATUS(op_resolver.AddEnergy());
  TF_LITE_ENSURE_STATUS(op_resolver.AddFilterBank());
  TF_LITE_ENSURE_STATUS(op_resolver.AddFilterBankSquareRoot());
  TF_LITE_ENSURE_STATUS(op_resolver.AddFilterBankSpectralSubtraction());
  TF_LITE_ENSURE_STATUS(op_resolver.AddPCAN());
  TF_LITE_ENSURE_STATUS(op_resolver.AddFilterBankLog());
  return kTfLiteOk;
}

// Pass a single 30ms window (480 int16 audio samples) through the preprocessor model
TfLiteStatus GenerateSingleFeature(const int16_t* audio_data,
                                   const int audio_data_size,
                                   int8_t* feature_output,
                                   tflite::MicroInterpreter* interpreter) {
  TfLiteTensor* input = interpreter->input(0);
  if (input == nullptr || kAudioSampleDurationCount != audio_data_size ||
      kAudioSampleDurationCount != input->dims->data[input->dims->size - 1]) {
    Serial.println("Error: Preprocessor input dimensions mismatch!");
    return kTfLiteError;
  }

  TfLiteTensor* output = interpreter->output(0);
  if (output == nullptr || kFeatureSize != output->dims->data[output->dims->size - 1]) {
    Serial.println("Error: Preprocessor output dimensions mismatch!");
    return kTfLiteError;
  }

  // Copy audio window data to preprocessor model input tensor
  std::copy_n(audio_data, audio_data_size, tflite::micro::GetTensorData<int16_t>(input));

  // Run FFT and spectral subtraction preprocessor execution
  TF_LITE_ENSURE_STATUS(interpreter->Invoke());

  // Copy output (40 quantized int8 frequency features) to output array
  std::copy_n(tflite::micro::GetTensorData<int8_t>(output), kFeatureSize, feature_output);

  return kTfLiteOk;
}

// Slice arbitrary length audio into 30ms windows with 20ms stride and extract features
TfLiteStatus GenerateFeatures(const int16_t* audio_data,
                              const size_t audio_data_size,
                              Features* features_output) {
  const tflite::Model* model = tflite::GetModel(g_audio_preprocessor_int8_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Preprocessor model schema version mismatch!");
    return kTfLiteError;
  }

  AudioPreprocessorOpResolver op_resolver;
  TF_LITE_ENSURE_STATUS(RegisterPreprocessorOps(op_resolver));

  tflite::MicroInterpreter interpreter(model, op_resolver, g_arena, kArenaSize);
  TF_LITE_ENSURE_STATUS(interpreter.AllocateTensors());

  size_t remaining_samples = audio_data_size;
  size_t feature_index = 0;
  const int16_t* audio_ptr = audio_data;

  // Slide window over audio array
  while (remaining_samples >= kAudioSampleDurationCount && feature_index < kFeatureCount) {
    TF_LITE_ENSURE_STATUS(GenerateSingleFeature(
        audio_ptr, kAudioSampleDurationCount, (*features_output)[feature_index], &interpreter));
    feature_index++;
    audio_ptr += kAudioSampleStrideCount;
    remaining_samples -= kAudioSampleStrideCount;
  }

  return kTfLiteOk;
}

// Run MicroSpeech classifier on a fully populated feature matrix and compare to expected label
TfLiteStatus LoadMicroSpeechModelAndPerformInference(const Features& features, const char* expected_label) {
  const tflite::Model* model = tflite::GetModel(g_micro_speech_quantized_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("MicroSpeech model schema version mismatch!");
    return kTfLiteError;
  }

  MicroSpeechOpResolver op_resolver;
  TF_LITE_ENSURE_STATUS(RegisterSpeechOps(op_resolver));

  tflite::MicroInterpreter interpreter(model, op_resolver, g_arena, kArenaSize);
  TF_LITE_ENSURE_STATUS(interpreter.AllocateTensors());

  TfLiteTensor* input = interpreter.input(0);
  TfLiteTensor* output = interpreter.output(0);

  if (input == nullptr || output == nullptr) {
    Serial.println("Input or Output tensor is null!");
    return kTfLiteError;
  }

  // Validate input/output dimensions match expected settings
  if (kFeatureElementCount != input->dims->data[input->dims->size - 1] ||
      kCategoryCount != output->dims->data[output->dims->size - 1]) {
    Serial.println("Tensor dimension shape mismatch!");
    return kTfLiteError;
  }

  // Copy full 49x40 matrix to model input
  std::copy_n(&features[0][0], kFeatureElementCount, tflite::micro::GetTensorData<int8_t>(input));

  // Execute neural network classification
  TF_LITE_ENSURE_STATUS(interpreter.Invoke());

  // Dequantize int8 output probabilities to floating point scale
  float output_scale = output->params.scale;
  int output_zero_point = output->params.zero_point;

  float category_predictions[kCategoryCount];
  Serial.print("Predictions for [expected: ");
  Serial.print(expected_label);
  Serial.println("]:");

  for (int i = 0; i < kCategoryCount; i++) {
    category_predictions[i] =
        (tflite::micro::GetTensorData<int8_t>(output)[i] - output_zero_point) * output_scale;
    Serial.print("  ");
    Serial.print(kCategoryLabels[i]);
    Serial.print(": ");
    Serial.println(category_predictions[i], 4);
  }

  // Determine label index with highest probability
  int prediction_index = std::distance(
      std::begin(category_predictions),
      std::max_element(std::begin(category_predictions), std::end(category_predictions)));

  Serial.print("--> Top Class Output: ");
  Serial.println(kCategoryLabels[prediction_index]);

  if (strcmp(expected_label, kCategoryLabels[prediction_index]) != 0) {
    Serial.println("[FAIL] Classification label mismatch!");
    return kTfLiteError;
  }

  Serial.println("[PASS] Classification succeeded.");
  return kTfLiteOk;
}

// Wrapper for End-to-End 1000ms Audio Tests
TfLiteStatus TestAudioSample(const char* label, const int16_t* audio_data, const size_t audio_data_size) {
  TF_LITE_ENSURE_STATUS(GenerateFeatures(audio_data, audio_data_size, &g_features));
  TF_LITE_ENSURE_STATUS(LoadMicroSpeechModelAndPerformInference(g_features, label));
  return kTfLiteOk;
}

// Unit Test: Check 30ms DSP feature calculations against baseline expected values
TfLiteStatus Test30msFeatureData(const char* test_name,
                                 const int16_t* audio_data,
                                 size_t audio_data_size,
                                 const int8_t* expected_features) {
  Serial.print("Running 30ms DSP Preprocessor Test: ");
  Serial.println(test_name);

  if (GenerateFeatures(audio_data, audio_data_size, &g_features) != kTfLiteOk) {
    Serial.println("[FAIL] Feature generation failed!");
    return kTfLiteError;
  }

  bool match = true;
  for (size_t i = 0; i < kFeatureSize; i++) {
    if (g_features[0][i] != expected_features[i]) {
      // Print first few discrepancies for debugging if math mismatch occurs
      Serial.print("  Mismatch at index ");
      Serial.print(i);
      Serial.print(": Calculated=");
      Serial.print(g_features[0][i]);
      Serial.print(", Expected=");
      Serial.println(expected_features[i]);
      match = false;
      break;
    }
  }

  if (match) {
    Serial.println("[PASS] 30ms Feature math matches expected array exactly.");
    return kTfLiteOk;
  } else {
    Serial.println("[FAIL] DSP math mismatch detected.");
    return kTfLiteError;
  }
}

}  // namespace

void setup() {
  Serial.begin(115200);
  while (!Serial && millis() < 4000); // Wait for Serial Monitor on platforms like ESP32/Teensy


}

void loop() {
  // Tests execute once in setup().
  Serial.println("\n==================================================");
  Serial.println("  TensorFlow Lite Micro - Full MicroSpeech Tests  ");
  Serial.println("==================================================\n");

  // -------------------------------------------------------------------
  // PART 1: 30ms Preprocessor Math Unit Tests
  // -------------------------------------------------------------------
  Serial.println("--- [1/6] 30ms 'NO' Feature Math Test ---");
  const int8_t expected_no_feature[kFeatureSize] = {
      126, 103, 124, 102, 124, 102, 123, 100, 118, 97, 118, 100, 118, 98,
      121, 100, 121, 98,  117, 91,  96,  74,  54,  87, 100, 87,  109, 92,
      91,  80,  64,  55,  83,  74,  74,  78,  114, 95, 101, 81,
  };
  Test30msFeatureData("NoFeatureTest", g_no_30ms_audio_data, g_no_30ms_audio_data_size, expected_no_feature);

  Serial.println("\n--- [2/6] 30ms 'YES' Feature Math Test ---");
  const int8_t expected_yes_feature[kFeatureSize] = {
      124, 105, 126, 103, 125, 101, 123, 100, 116, 98,  115, 97,  113, 90,
      91,  82,  104, 96,  117, 97,  121, 103, 126, 101, 125, 104, 126, 104,
      125, 101, 116, 90,  81,  74,  80,  71,  83,  76,  82,  71,
  };
  Test30msFeatureData("YesFeatureTest", g_yes_30ms_audio_data, g_yes_30ms_audio_data_size, expected_yes_feature);

  // -------------------------------------------------------------------
  // PART 2: 1000ms Full Inference Classifier Tests
  // -------------------------------------------------------------------
  Serial.println("\n--- [3/6] 1000ms 'NO' Word Inference Test ---");
  TestAudioSample("no", g_no_1000ms_audio_data, g_no_1000ms_audio_data_size);

  Serial.println("\n--- [4/6] 1000ms 'YES' Word Inference Test ---");
  TestAudioSample("yes", g_yes_1000ms_audio_data, g_yes_1000ms_audio_data_size);

  Serial.println("\n--- [5/6] 1000ms 'SILENCE' Inference Test ---");
  TestAudioSample("silence", g_silence_1000ms_audio_data, g_silence_1000ms_audio_data_size);

  Serial.println("\n--- [6/6] 1000ms 'NOISE' Inference Test ---");
  // Environmental noise is expected to be classified as silence
  TestAudioSample("silence", g_noise_1000ms_audio_data, g_noise_1000ms_audio_data_size);

  Serial.println("\n==================================================");
  Serial.println("               All Tests Complete!                ");
  Serial.println("==================================================");
  
  delay(5000);
}