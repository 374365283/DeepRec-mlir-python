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

// Functions for getting information about kernels registered in the binary.
// Migrated from previous SWIG file (mlir.i) authored by aminim@.
#ifndef TENSORFLOW_COMPILER_MLIR_PYTHON_MLIR_H_
#define TENSORFLOW_COMPILER_MLIR_PYTHON_MLIR_H_

#include <string>
#include "absl/strings/string_view.h"

namespace tensorflow {

std::string ImportGraphDef(const std::string& proto, const std::string& pass_pipeline,
                           absl::string_view(input_names), absl::string_view(input_data_types),
                           absl::string_view(input_data_shapes), absl::string_view(output_names));

std::string ExportGraphDef(const std::string& mlir_txt, const std::string& pass_pipeline);

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_MLIR_PYTHON_MLIR_H_