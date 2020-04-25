package(default_visibility = ["//src:__subpackages__"])


cc_binary(
    name = "greenscreen",
    srcs = ["mediapipe_runner.cc"],
    deps = [
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/port:commandlineflags",
        "//mediapipe/framework/port:opencv_highgui",
        "//mediapipe/framework/port:opencv_imgproc",
        "//mediapipe/framework/port:opencv_video",
        "@usr_local//:libv4l2cpp",
        "//src/graphs:deeplab_segmentation",
        "//mediapipe/gpu:gpu_buffer",
        "//mediapipe/gpu:gpu_shared_data_internal",
    ],
)

cc_binary(
    name = "opencv_runner",
    srcs = ["opencv_runner.cc"],
    deps = [
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/port:commandlineflags",
        "//mediapipe/framework/port:opencv_highgui",
        "//mediapipe/framework/port:opencv_imgproc",
        "//mediapipe/framework/port:opencv_video",
        "@usr_local//:libv4l2cpp",
    ],
)