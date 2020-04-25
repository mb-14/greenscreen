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
//
// A command line utility to parse a text proto and output a binary proto.

#include <stdlib.h>

#include <fstream>
#include <string>

#include "mediapipe/framework/calculator.pb.h"
#include "mediapipe/framework/port/advanced_proto_inc.h"
#include "mediapipe/framework/port/canonical_errors.h"
#include "mediapipe/framework/port/commandlineflags.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"

DEFINE_string(proto_source, "",
              "The template source file containing CalculatorGraphConfig "
              "protobuf text with inline template params.");
DEFINE_string(
    proto_output, "",
    "An output template file in binary CalculatorGraphTemplate form.");

#define EXIT_IF_ERROR(status) \
  if (!status.ok()) {         \
    LOG(ERROR) << status;     \
    return EXIT_FAILURE;      \
  }

namespace mediapipe {

mediapipe::Status ReadProto(proto_ns::io::ZeroCopyInputStream* in,
                            bool read_text, const std::string& source,
                            proto_ns::Message* result) {
  if (read_text) {
    RET_CHECK(proto_ns::TextFormat::Parse(in, result))
        << "could not parse text proto: " << source;
  } else {
    RET_CHECK(result->ParseFromZeroCopyStream(in))
        << "could not parse binary proto: " << source;
  }
  return mediapipe::OkStatus();
}

mediapipe::Status WriteProto(const proto_ns::Message& message, bool write_text,
                             const std::string& dest,
                             proto_ns::io::ZeroCopyOutputStream* out) {
  if (write_text) {
    RET_CHECK(proto_ns::TextFormat::Print(message, out))
        << "could not write text proto to: " << dest;
  } else {
    RET_CHECK(message.SerializeToZeroCopyStream(out))
        << "could not write binary proto to: " << dest;
  }
  return mediapipe::OkStatus();
}

// Read a proto from a text or a binary file.
mediapipe::Status ReadFile(const std::string& proto_source, bool read_text,
                           proto_ns::Message* result) {
  std::ifstream ifs(proto_source);
  proto_ns::io::IstreamInputStream in(&ifs);
  MP_RETURN_IF_ERROR(ReadProto(&in, read_text, proto_source, result));
  return mediapipe::OkStatus();
}

// Write a proto to a text or a binary file.
mediapipe::Status WriteFile(const std::string& proto_output, bool write_text,
                            const proto_ns::Message& message) {
  std::ofstream ofs(proto_output, std::ofstream::out | std::ofstream::trunc);
  proto_ns::io::OstreamOutputStream out(&ofs);
  MP_RETURN_IF_ERROR(WriteProto(message, write_text, proto_output, &out));
  return mediapipe::OkStatus();
}

}  // namespace mediapipe

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  // Validate command line options.
  mediapipe::Status status;
  if (FLAGS_proto_source.empty()) {
    status.Update(
        ::mediapipe::InvalidArgumentError("--proto_source must be specified"));
  }
  if (FLAGS_proto_output.empty()) {
    status.Update(
        ::mediapipe::InvalidArgumentError("--proto_output must be specified"));
  }
  if (!status.ok()) {
    return EXIT_FAILURE;
  }
  mediapipe::CalculatorGraphConfig config;
  EXIT_IF_ERROR(mediapipe::ReadFile(FLAGS_proto_source, true, &config));
  EXIT_IF_ERROR(mediapipe::WriteFile(FLAGS_proto_output, false, config));
  return EXIT_SUCCESS;
}
