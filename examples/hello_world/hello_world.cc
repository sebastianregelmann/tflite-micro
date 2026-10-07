/*
 * TensorFlow Lite for Microcontrollers (TFLM) - hello_world example
 * 
 * Converted from C++ hello_world example to a standalone Arduino Sketch.
 * Performs two categories of tests:
 *   1. Image with a person
 *   2. Image with no person
 *
 * Requirements:
 *   - Board with 150KB+ SRAM (ESP32, Teensy 4.x, Raspberry Pi Pico, nRF52840, etc.)
 */

#include <Arduino.h>

// TFLite Micro Headers
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_allocator.h"
#include "tensorflow/lite/micro/micro_resource_variable.h"
#include "tensorflow/lite/micro/micro_log.h"

// Include the model data
#include "examples/hello_world/model_data/hello_world_model_data.h"

// 2 kb Tensor Arena
constexpr size_t kTensorArenaSize = 1024 * 2;
uint8_t *tensor_arena = nullptr;

// model and interpreter
const tflite::Model *tfl_model = nullptr;
tflite::MicroInterpreter *interpreter = nullptr;

// input and outut
TfLiteTensor *input_tensor = nullptr;
TfLiteTensor *output_tensor = nullptr;

// benchmark variable
float x = 0;

void run_synthetic_benchmark();

void setup()
{
    Serial.begin(115200);
    delay(3000);

    Serial.println("\n--- TFLite Micro hello_world Setup ---");

    // 1. Allocate RAM for the tensor arena
    tensor_arena = (uint8_t *)malloc(kTensorArenaSize);
    if (tensor_arena == nullptr)
    {
        Serial.println("[ERROR] Failed to allocate tensor_arena in RAM!");
        return;
    }
    Serial.print("Allocated Tensor Arena (");
    Serial.print(kTensorArenaSize / 1024, (uint32_t)tensor_arena);
    Serial.println("KB) in RAM");


    // 2. Load Model
    tfl_model = tflite::GetModel(model);
    if (tfl_model == nullptr)
    {
        Serial.println("[ERROR] Failed Loading tfl_model");
        return;
    }
    Serial.println("Loaded Model");

    // 3. Load Op Resolver
    static tflite::MicroMutableOpResolver<1> resolver;
    resolver.AddFullyConnected();
    Serial.println("Loaded Op resolver");

    // 4. Instantiate Interpreter
    static tflite::MicroInterpreter static_interpreter(tfl_model, resolver, tensor_arena, kTensorArenaSize);
    interpreter = &static_interpreter;
    Serial.println("Created Interpreter \n");

    // 5. Allocate Tensors
    if (interpreter->AllocateTensors() != kTfLiteOk)
    {
        Serial.println("[ERROR] AllocateTensors() failed!");
        return;
    }

    // 6. Assing tensors
    input_tensor = interpreter->input(0);
    output_tensor = interpreter->output(0);

    Serial.print("Arena Allocated Successfully! Used Bytes: ");
    Serial.print(interpreter->arena_used_bytes());
    Serial.print(" / ");
    Serial.print(kTensorArenaSize);
    Serial.println("\n\n");

    Serial.println("Setup finished, Starting Main Loop ... \n\n");
    Serial.flush();
    delay(1000);
}

void loop()
{
    run_synthetic_benchmark();
    delay(500);
}

void run_synthetic_benchmark()
{
    if (input_tensor == nullptr || interpreter == nullptr)
    {
        Serial.println("[ERROR] Interpreter not initialized.");
        return;
    }

    // Set input value
    Serial.println("Set input value");
    input_tensor->data.f[0] = x;

    // Measure execution time
    Serial.println("Invoking interpreter...");
    uint32_t start_time = micros();
    // Invoke the model
    TfLiteStatus invoke_status = interpreter->Invoke();
    uint32_t end_time = micros();

    if (invoke_status != kTfLiteOk)
    {
        Serial.println("[ERROR] Inference execution failed!");
        return;
    }

    uint32_t duration_us = end_time - start_time;

    // Ouput the prediction
    Serial.print("Inference Status: SUCCESS\nExecution Time: ");
    Serial.print(duration_us);
    Serial.println("us");
    Serial.print("Input: "),
    Serial.print(x);
    Serial.print("\t SIN Prediction: ");
    Serial.print(output_tensor->data.f[0]);
    Serial.print("\t COS Prediction: ");
    Serial.print(output_tensor->data.f[1]);
    Serial.print("\t Correct values: ");
    Serial.print(sin(x));
    Serial.print(", ");
    Serial.println(cos(x));

    Serial.println("\n-----------------------------------------------\n");

    // update input value for next run
    x += 0.1;
    if (x > 2 * PI)
    {
        x = 0.0;
    }
}