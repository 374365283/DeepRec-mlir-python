/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "include/pybind11/pybind11.h"
#include "tensorflow/core/platform/tensor_float_32_utils.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/compiler/mlir/python/mlir.h"

PYBIND11_MODULE(_pywrap_tensor_float_32_execution, m) {
  m.def("enable", &tensorflow::enable_tensor_float_32_execution);
  m.def("is_enabled", &tensorflow::tensor_float_32_execution_enabled);
  m.def("string_concat", 
        [](const std::string &input1, const std::string &input2) {
            tensorflow::GraphDef graph_def;
            std::string output = input1 + input2;
            return output;
        });

  m.def("import_graphdef",
        [](const std::string &graphdef, const std::string &pass_pipeline,
           const std::string &input_names, const std::string &input_data_types,
           const std::string &input_data_shapes, const std::string &output_names) {
          std::string output = tensorflow::ImportGraphDef(
              graphdef, pass_pipeline, input_names, input_data_types, 
              input_data_shapes, output_names);
          return output;
        });
    
   m.def("export_graphdef",
        [](const std::string &mlir_txt, const std::string &pass_pipeline) {
          std::string output = tensorflow::ExportGraphDef(
              mlir_txt, pass_pipeline);
          return output;
        }); 
}
