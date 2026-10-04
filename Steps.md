1. Remove all files

2. Clone the Repository


3. Build the Project into tmp
* cd tflite-micro
* python3 tensorflow/lite/micro/tools/project_generation/create_tflm_tree.py tmp
* cd ..
* cp -r tflite-micro/tmp .
* rm -rf tflite-micro
* mv tmp src