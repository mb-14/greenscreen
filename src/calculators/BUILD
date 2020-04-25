package(default_visibility = ["//visibility:public"])


cc_library(
    name = "deeplab_tensors_to_segmentation_calculator",
    srcs = ["deeplab_tensors_to_segmentation_calculator.cc"],
    visibility = ["//visibility:public"],
    deps = [
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/formats:image_frame_opencv",
        "//mediapipe/framework/port:opencv_imgcodecs",
        "//mediapipe/framework/port:opencv_imgproc",
        "//mediapipe/framework/port:vector",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/calculators/tflite:util",
        "//mediapipe/framework/formats:video_stream_header",
        "//mediapipe/framework:calculator_context",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/util:resource_util",
        "@org_tensorflow//tensorflow/lite:framework",
        "//mediapipe/gpu:gl_calculator_helper",
        "//mediapipe/gpu:gl_simple_shaders",
        "//mediapipe/gpu:gpu_buffer",
        "//mediapipe/gpu:shader_util",
        "@org_tensorflow//tensorflow/lite/delegates/gpu:gl_delegate",
        "@org_tensorflow//tensorflow/lite/delegates/gpu/gl:gl_buffer",
        "@org_tensorflow//tensorflow/lite/delegates/gpu/gl:gl_program",
        "@org_tensorflow//tensorflow/lite/delegates/gpu/gl:gl_shader",
        "@org_tensorflow//tensorflow/lite/delegates/gpu/gl:gl_texture",
    ],
    alwayslink = 1,
)
