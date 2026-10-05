/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include <Arduino.h>

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "examples/person_detection/models/person_detection_model_data.h"
#include "examples/person_detection/test_data/person_detection_test_data.h"

const tflite::Model *model = nullptr;
tflite::MicroInterpreter *interpreter = nullptr;
TfLiteTensor *input = nullptr;

// An area of memory to use for input, output, and intermediate arrays.
constexpr int kTensorArenaSize = 136 * 1024;
alignas(16) static uint8_t tensor_arena[kTensorArenaSize];

uint8_t personImage = 0;

void setup()
{
  Serial.begin(115200);
  delay(2000);

  Serial.print("Start allocating ressources for tensorflow...");
  tflite::InitializeTarget();

  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  model = tflite::GetModel(g_person_detect_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION)
  {
    Serial.print("Model provided is schema version ");
    Serial.print(model->version());
    Serial.print(" not equal to supported version");
    Serial.println(TFLITE_SCHEMA_VERSION);
    return;
  }
  Serial.println("loaded model");
  delay(100);

  // Pull in only the operation implementations we need
  // This relies on a complete list of all the ops needed by this graph.
  static tflite::MicroMutableOpResolver<5> micro_op_resolver;
  micro_op_resolver.AddAveragePool2D(tflite::Register_AVERAGE_POOL_2D_INT8());
  micro_op_resolver.AddConv2D(tflite::Register_CONV_2D_INT8());
  micro_op_resolver.AddDepthwiseConv2D(tflite::Register_DEPTHWISE_CONV_2D_INT8());
  micro_op_resolver.AddReshape();
  micro_op_resolver.AddSoftmax(tflite::Register_SOFTMAX_INT8());
  Serial.println("registered ops");
  delay(100);

  // Build an interpreter to run the model with
  static tflite::MicroInterpreter static_interpreter(model, micro_op_resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;
  Serial.println("created interpreter");
  delay(100);

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk)
  {
    Serial.println("AllocateTensors() failed");
    return;
  }

  // Get information about the memory area to use for the model's input.
  input = interpreter->input(0);
  Serial.println("Allocated Tensors");
  delay(100);

  Serial.println("Setup was successful \n\n");
  delay(1000);
}

void loop()
{
  Serial.println("---------------------------------------");

  // Load image into model
  for (int i = 0; i < kMaxImageSize; ++i)
  {
    input->data.int8[i] = personImage % 2 == 0 ? no_person_image_data[i] : person_image_data[i];
  }
  Serial.println("Loaded Image into model");

  // Run the model on this input and make sure it succeeds.
  if (kTfLiteOk != interpreter->Invoke())
  {
    MicroPrintf("Invoke failed.");
  }
  Serial.println("Invoked model");

  // Process the inference results.
  TfLiteTensor *output = interpreter->output(0);
  int8_t person_score = output->data.uint8[kPersonIndex];
  int8_t no_person_score = output->data.uint8[kNotAPersonIndex];

  // Print the result
  Serial.print("Prediction: ");
  Serial.print("person score: ");
  Serial.print(person_score);
  Serial.print("  no person score: ");
  Serial.print(no_person_score);
  Serial.print("  prediceted label: ");
  Serial.print(person_score > no_person_score ? "person" : "no person");
  Serial.print("  correct label: ");
  Serial.println(personImage % 2 == 0 ? "no person" : "person");
  Serial.println("\n\n");

  // Increase counter by one for other image in next run
  personImage++;

  delay(3000);
}
