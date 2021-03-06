# Copyright 2019 The MediaPipe Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

load(
    "//mediapipe/framework/tool:mediapipe_graph.bzl",
    "mediapipe_binary_graph",
    "mediapipe_simple_subgraph",
)


package(default_visibility = ["//visibility:public"])


cc_library(
    name = "virtual_background",
    deps = [
        "//mediapipe/calculators/core:flow_limiter_calculator",
        "//mediapipe/calculators/core:mux_calculator",
        "//mediapipe/calculators/image:mask_overlay_calculator",
        "//src/calculators:demux_calculator",
        ":deeplab_segmentation_subgraph",
        ":slimnet_segmentation_subgraph"
    ]
)

mediapipe_simple_subgraph(
    name = "deeplab_segmentation_subgraph",
    graph = "deeplab_segmentation_subgraph.pbtxt",
    register_as = "DeeplabSegmentationSubgraph",
    deps = [
        "//mediapipe/calculators/image:image_transformation_calculator",
        "//mediapipe/calculators/tflite:tflite_converter_calculator",
        "//mediapipe/calculators/tflite:tflite_inference_calculator",
        "//mediapipe/calculators/image:opencv_encoded_image_to_image_frame_calculator",
        "//src/calculators:deeplab_tensors_to_segmentation_calculator",
        "//mediapipe/gpu:image_frame_to_gpu_buffer_calculator",
        "//mediapipe/gpu:gl_calculator_helper",
    ],
)

mediapipe_simple_subgraph(
    name = "slimnet_segmentation_subgraph",
    graph = "slimnet_segmentation_subgraph.pbtxt",
    register_as = "SlimnetSegmentationSubgraph",
    deps = [
        "//mediapipe/calculators/core:previous_loopback_calculator",
        "//mediapipe/calculators/image:image_transformation_calculator",
        "//mediapipe/calculators/tflite:tflite_converter_calculator",
        "//mediapipe/calculators/tflite:tflite_inference_calculator",
        "//mediapipe/calculators/tflite:tflite_tensors_to_segmentation_calculator",
        "//mediapipe/calculators/image:opencv_encoded_image_to_image_frame_calculator",
        "//mediapipe/gpu:image_frame_to_gpu_buffer_calculator",
        "//mediapipe/gpu:gl_calculator_helper",
    ],
)