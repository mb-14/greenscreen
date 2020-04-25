// Copyright 2019 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <random>
#include <vector>

#include "absl/memory/memory.h"
#include "mediapipe/calculators/tflite/tflite_converter_calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/formats/matrix.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/integral_types.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"  // NOLINT
#include "mediapipe/framework/tool/validate_type.h"
#include "tensorflow/lite/interpreter.h"

namespace mediapipe {

namespace {

constexpr char kTransposeOptionsString[] =
    "[mediapipe.TfLiteConverterCalculatorOptions.ext]: {"
    "row_major_matrix: True}";

}  // namespace

using RandomEngine = std::mt19937_64;
const uint32 kSeed = 1234;
const int kNumSizes = 8;
const int sizes[kNumSizes][2] = {{1, 1}, {12, 1}, {1, 9},   {2, 2},
                                 {5, 3}, {7, 13}, {16, 32}, {101, 2}};

class TfLiteConverterCalculatorTest : public ::testing::Test {
 protected:
  // Adds a packet with a matrix filled with random values in [0,1].
  void AddRandomMatrix(int num_rows, int num_columns, uint32 seed,
                       bool row_major_matrix = false) {
    RandomEngine random(kSeed);
    std::uniform_real_distribution<> uniform_dist(0, 1.0);
    auto matrix = ::absl::make_unique<Matrix>();
    matrix->resize(num_rows, num_columns);
    if (row_major_matrix) {
      for (int y = 0; y < num_rows; ++y) {
        for (int x = 0; x < num_columns; ++x) {
          float value = uniform_dist(random);
          (*matrix)(y, x) = value;
        }
      }
    } else {
      for (int x = 0; x < num_columns; ++x) {
        for (int y = 0; y < num_rows; ++y) {
          float value = uniform_dist(random);
          (*matrix)(y, x) = value;
        }
      }
    }
    MP_ASSERT_OK(graph_->AddPacketToInputStream(
        "matrix", Adopt(matrix.release()).At(Timestamp(0))));
  }

  std::unique_ptr<CalculatorGraph> graph_;
};

TEST_F(TfLiteConverterCalculatorTest, RandomMatrixColMajor) {
  for (int size_index = 0; size_index < kNumSizes; ++size_index) {
    const int num_rows = sizes[size_index][0];
    const int num_columns = sizes[size_index][1];

    // Run the calculator and verify that one output is generated.
    CalculatorGraphConfig graph_config =
        ::mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"(
          input_stream: "matrix"
          node {
            calculator: "TfLiteConverterCalculator"
            input_stream: "MATRIX:matrix"
            output_stream: "TENSORS:tensor"
            options {
              [mediapipe.TfLiteConverterCalculatorOptions.ext] {
                row_major_matrix: false
              }
            }
          }
        )");
    std::vector<Packet> output_packets;
    tool::AddVectorSink("tensor", &graph_config, &output_packets);

    // Run the graph.
    graph_ = absl::make_unique<CalculatorGraph>();
    MP_ASSERT_OK(graph_->Initialize(graph_config));
    MP_ASSERT_OK(graph_->StartRun({}));

    // Push the tensor into the graph.
    AddRandomMatrix(num_rows, num_columns, kSeed, /*row_major_matrix=*/false);

    // Wait until the calculator done processing.
    MP_ASSERT_OK(graph_->WaitUntilIdle());
    EXPECT_EQ(1, output_packets.size());

    // Get and process results.
    const std::vector<TfLiteTensor>& tensor_vec =
        output_packets[0].Get<std::vector<TfLiteTensor>>();
    EXPECT_EQ(1, tensor_vec.size());

    const TfLiteTensor* tensor = &tensor_vec[0];
    EXPECT_EQ(kTfLiteFloat32, tensor->type);

    // Verify that the data is correct.
    RandomEngine random(kSeed);
    std::uniform_real_distribution<> uniform_dist(0, 1.0);
    const float* tensor_buffer = tensor->data.f;
    for (int i = 0; i < num_rows * num_columns; ++i) {
      const float expected = uniform_dist(random);
      EXPECT_EQ(expected, tensor_buffer[i]) << "at i = " << i;
    }

    // Fully close graph at end, otherwise calculator+tensors are destroyed
    // after calling WaitUntilDone().
    MP_ASSERT_OK(graph_->CloseInputStream("matrix"));
    MP_ASSERT_OK(graph_->WaitUntilDone());

    graph_.reset();
  }
}

TEST_F(TfLiteConverterCalculatorTest, RandomMatrixRowMajor) {
  for (int size_index = 0; size_index < kNumSizes; ++size_index) {
    const int num_rows = sizes[size_index][0];
    const int num_columns = sizes[size_index][1];

    // Run the calculator and verify that one output is generated.
    CalculatorGraphConfig graph_config =
        ::mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"(
          input_stream: "matrix"
          node {
            calculator: "TfLiteConverterCalculator"
            input_stream: "MATRIX:matrix"
            output_stream: "TENSORS:tensor"
            options {
              [mediapipe.TfLiteConverterCalculatorOptions.ext] {
                row_major_matrix: true
              }
            }
          }
        )");
    std::vector<Packet> output_packets;
    tool::AddVectorSink("tensor", &graph_config, &output_packets);

    // Run the graph.
    graph_ = absl::make_unique<CalculatorGraph>();
    MP_ASSERT_OK(graph_->Initialize(graph_config));
    MP_ASSERT_OK(graph_->StartRun({}));

    // Push the tensor into the graph.
    AddRandomMatrix(num_rows, num_columns, kSeed, /*row_major_matrix=*/true);

    // Wait until the calculator done processing.
    MP_ASSERT_OK(graph_->WaitUntilIdle());
    EXPECT_EQ(1, output_packets.size());

    // Get and process results.
    const std::vector<TfLiteTensor>& tensor_vec =
        output_packets[0].Get<std::vector<TfLiteTensor>>();
    EXPECT_EQ(1, tensor_vec.size());

    const TfLiteTensor* tensor = &tensor_vec[0];
    EXPECT_EQ(kTfLiteFloat32, tensor->type);

    // Verify that the data is correct.
    RandomEngine random(kSeed);
    std::uniform_real_distribution<> uniform_dist(0, 1.0);
    const float* tensor_buffer = tensor->data.f;
    for (int i = 0; i < num_rows * num_columns; ++i) {
      const float expected = uniform_dist(random);
      EXPECT_EQ(expected, tensor_buffer[i]) << "at i = " << i;
    }

    // Fully close graph at end, otherwise calculator+tensors are destroyed
    // after calling WaitUntilDone().
    MP_ASSERT_OK(graph_->CloseInputStream("matrix"));
    MP_ASSERT_OK(graph_->WaitUntilDone());

    graph_.reset();
  }
}

}  // namespace mediapipe
