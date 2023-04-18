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

#include "llvm/Support/raw_ostream.h"
#include "mlir/Parser.h"  // TF:llvm-project
#include "mlir/Pass/PassManager.h"  // TF:llvm-project
#include "mlir/Pass/PassRegistry.h"  // TF:llvm-project
#include "mlir/Pass/Pass.h" // TF:llvm-project
#include "tensorflow/compiler/mlir/tensorflow/translate/import_model.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/error_util.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/import_utils.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/mlir_roundtrip_flags.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/export_graphdef.h"
#include "tensorflow/stream_executor/lib/statusor.h"

#include "absl/strings/str_split.h"

namespace tensorflow {

std::string ImportGraphDef(const std::string& proto, const std::string& pass_pipeline,
                           absl::string_view input_names, absl::string_view input_data_types,
                           absl::string_view input_data_shapes, absl::string_view output_names) {
  GraphDebugInfo debug_info;
  GraphImportConfig specs;
  auto s = ParseInputArrayInfo(input_names, input_data_types, input_data_shapes,
                               &specs.inputs);
  if (!s.ok()) {
    return "// error";
  }
  if (!output_names.empty()) {
    specs.outputs = absl::StrSplit(output_names, ',');
  }

  GraphDef graphdef;
  mlir::MLIRContext context;
  auto module = ConvertGraphdefToMlir(graphdef, debug_info, specs, &context);
  if (!module.ok()) {
    return "// error";
  }

  if (!pass_pipeline.empty()) {
    mlir::PassManager pm(&context);
    std::string error;
    llvm::raw_string_ostream error_stream(error);
    if (failed(mlir::parsePassPipeline(pass_pipeline, pm, error_stream))) {
      return "// error";
    }

    if (mlir::failed(pm.run(*module.ValueOrDie()))) {
      return "// error";
    }
  }
  return MlirModuleToString(*module.ConsumeValueOrDie());
}

std::string ExportGraphDef(const std::string& mlir_txt, const std::string& pass_pipeline) {
  mlir::MLIRContext context;
  mlir::StatusScopedDiagnosticHandler diagnostic_handler(&context);
  GraphExportConfig specs;

  // Convert tf dialect to tf exector dialect.
  mlir::OwningModuleRef module;
  module = mlir::parseSourceString(mlir_txt, &context);
  if (!module) {
      return "// error";
  }

  if (!pass_pipeline.empty()) {
    mlir::PassManager pm(&context);
    std::string error;
    llvm::raw_string_ostream error_stream(error);
    if (failed(mlir::parsePassPipeline(pass_pipeline, pm, error_stream))) {
      return "// error";
    }

    if (mlir::failed(pm.run(*module))) {
      return "// error";
    }
  }
  std::string tf_executor_mlir_txt = MlirModuleToString(*module);

  // Convert tf executor dialect to Graphdef.
  mlir::OwningModuleRef tf_executor_module;
  tf_executor_module = mlir::parseSourceString(tf_executor_mlir_txt, &context);
  if (!tf_executor_module) {
    return "// error";
  }
  StatusOr<std::unique_ptr<tensorflow::GraphDef>> graphdef(ConvertMlirToGraphdef(*tf_executor_module, specs));
  return graphdef.ValueOrDie()->DebugString();
}

}  // namespace tensorflow