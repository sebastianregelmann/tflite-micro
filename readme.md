# About this Project
This repository is a fork of the official [tflite-micro](https://github.com/tensorflow/tflite-micro) repository and aims to simplify the usage of that library for Microcontrollers (specificly in the PlatformIO ecosystem but not exclusivly). I want to keep a packed version of the `tflite-micro` source code for easy usage without the need of compiling/generating the library from source.

# Content
## src
This repository contains the generated library structure with all dependencies and includes needed for compilation. The code is kept in the original form (except for the examples) and was genearated using [this](https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/tools/project_generation/create_tflm_tree.py) script. 

## examples
The examples in the folder `examples` are based on the original examples of the project but are updated in a way to make them compatible with the Arduino framework. The Examples are the following:
### hello_world
Based on [this example](https://github.com/tensorflow/tflite-micro/tree/main/tensorflow/lite/micro/examples/hello_world) example but expanded to use `sin` and `cos` prediction at once. THis example is the most diffrent from the original source to show how the library can be used in a "simpler" arduino style.

### micro_speech 
Based on [this example](https://github.com/tensorflow/tflite-micro/tree/main/tensorflow/lite/micro/examples/micro_speech). For a more detailed look, read the original example.

### person_detection
Based on [this example](https://github.com/tensorflow/tflite-micro/tree/main/tensorflow/lite/micro/examples/person_detection). For a more detailed look, read the original example.


# Version
This version is release `1.0.0`.


The current version of the library is a build from [this](https://github.com/tensorflow/tflite-micro/commit/1fae6040446d86147b06a1280da4086305e4bcc0) commit to the official [tflite-micro](https://github.com/tensorflow/tflite-micro) repository. 

## Future Versions 
The library will be build once every week with the following naming convention: `1.YY.WW`.

# Future plans
I plan to automate this repository using Gihub Actions and workflows to automate the building and publishing work. So that the library that can be used by PlatformIO users is always up to date because most of the libraries that fork the `tflite-micro` project are either not up to date/maintained and/or not the original code but repackaged for a more simplified usage. 


# Original Project
* https://github.com/tensorflow/tflite-micro

# License
Apache License
                           Version 2.0

# Version
This version is release ``.

The current version of the library is a build from [this](https://github.com/tensorflow/tflite-micro/commit/904193e691ede876f4b8b70e6abe68ae464072f9) commit to the official [tflite-micro](https://github.com/tensorflow/tflite-micro) repository.