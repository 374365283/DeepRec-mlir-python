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

#include <string>

#include "tensorflow/compiler/mlir/python/mlir.h"

#include "llvm/Support/raw_ostream.h"
#include "mlir/Parser.h"  // TF:llvm-project
#include "mlir/Pass/PassManager.h"  // TF:llvm-project
#include "mlir/Pass/PassRegistry.h"  // TF:llvm-project
#include "mlir/Pass/Pass.h" // TF:llvm-project
#include "tensorflow/stream_executor/lib/statusor.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/import_utils.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/import_model.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/export_graphdef.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/mlir_roundtrip_flags.h"
#include "absl/strings/str_split.h"

namespace tensorflow {

std::string ImportGraphDef(const std::string& proto, const std::string& pass_pipeline,
                           absl::string_view input_names, absl::string_view input_data_types,
                           absl::string_view input_data_shapes, absl::string_view output_names) {

  GraphImportConfig specs;
  auto parse_input_status = ParseInputArrayInfo(input_names, input_data_types, input_data_shapes,
                               &specs.inputs);
  if (!parse_input_status.ok()) {
    return "// error 1";
  }
  if (!output_names.empty()) {
    specs.outputs = absl::StrSplit(output_names, ',');
  }
  
  // Convert txt to graphdef.
  GraphDef graphdef;
  auto load_proto_status = tensorflow::LoadProtoFromBuffer(proto, &graphdef);
  if (!load_proto_status.ok()) {
    return "// error 2";
  }

  // Convert graphdef to tf executor dialect.
  GraphDebugInfo debug_info;
  mlir::MLIRContext context;
  auto module = ConvertGraphdefToMlir(graphdef, debug_info, specs, &context);
  if (!module.ok()) {
    return "// error 3";
  }

  // Convert tf executor dialect to tf dialect.
  if (!pass_pipeline.empty()) {
    mlir::PassManager pm(&context);
    std::string error;
    llvm::raw_string_ostream error_stream(error);
    if (failed(mlir::parsePassPipeline(pass_pipeline, pm, error_stream))) {
      return "// error 4";
    }

    if (mlir::failed(pm.run(*module.ValueOrDie()))) {
      return "// error 5";
    }
  }
  return MlirModuleToString(*module.ConsumeValueOrDie());
}

std::string ExportGraphDef(const std::string& mlir_txt, const std::string& pass_pipeline) {
  // Convert tf dialect txt to tf dialect module.
  mlir::MLIRContext context;
  mlir::OwningModuleRef module;
  module = mlir::parseSourceString(mlir_txt, &context);
  if (!module) {
      return "// error 1";
  }

  // Convert tf dialect to tf exector dialect.
  if (!pass_pipeline.empty()) {
    mlir::PassManager pm(&context);
    std::string error;
    llvm::raw_string_ostream error_stream(error);
    if (failed(mlir::parsePassPipeline(pass_pipeline, pm, error_stream))) {
      return "// error 2";
    }
    if (mlir::failed(pm.run(*module))) {
      return "// error 3";
    }
  }
  
  // Convert tf executor dialect to graphdef.
  GraphExportConfig specs;
  StatusOr<std::unique_ptr<tensorflow::GraphDef>> graphdef(ConvertMlirToGraphdef(*module, specs));
  return graphdef.ValueOrDie()->DebugString();
  
  //std::string output = mlir_txt + pass_pipeline;
  //return output;
}

}  // namespace tensorflow
