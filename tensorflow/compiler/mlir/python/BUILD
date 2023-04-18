#load(
#    "//tensorflow:tensorflow.bzl",
#    "tf_copts",
#)

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "mlir",
    srcs = ["mlir.cc"],
    hdrs = ["mlir.h"],
    deps = [
        "//tensorflow/compiler/mlir/tensorflow:convert_graphdef",
        "//tensorflow/compiler/mlir/tensorflow:error_util",
        "//tensorflow/compiler/mlir/tensorflow:import_utils",
        "//tensorflow/stream_executor/lib",
        "@llvm-project//llvm:support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Transforms",
    ],
#    copts = tf_copts(),
    alwayslink = 1,
    visibility = [
        "//tensorflow:__subpackages__",
    ],
)

filegroup(
    name = "pywrap_mlir_hdrs",
    srcs = [
        "mlir.h",
    ],
    visibility = [
        "//tensorflow:__subpackages__",
    ],
)
