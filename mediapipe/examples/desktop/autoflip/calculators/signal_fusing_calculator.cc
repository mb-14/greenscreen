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

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "mediapipe/examples/desktop/autoflip/autoflip_messages.pb.h"
#include "mediapipe/examples/desktop/autoflip/calculators/signal_fusing_calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"

using mediapipe::Packet;
using mediapipe::PacketTypeSet;
using mediapipe::autoflip::DetectionSet;
using mediapipe::autoflip::SalientRegion;
using mediapipe::autoflip::SignalType;

namespace mediapipe {
namespace autoflip {

struct InputSignal {
  SalientRegion signal;
  int source;
};

struct Frame {
  std::vector<InputSignal> input_detections;
  mediapipe::Timestamp time;
};

// This calculator takes one scene change signal and an arbitrary number of
// detection signals and outputs a single list of detections.  The scores for
// the detections can be re-normalized using the options proto.  Additionally,
// if a detection has a consistent tracking id during a scene the score for that
// detection is averaged over the whole scene.
//
// Example:
//  node {
//    calculator: "SignalFusingCalculator"
//    input_stream: "scene_change"
//    input_stream: "detection_faces"
//    input_stream: "detection_custom_text"
//    output_stream: "salient_region"
//    options:{
//    [mediapipe.autoflip.SignalFusingCalculatorOptions.ext]:{
//      signal_settings{
//        type: {standard: FACE}
//        min_score: 0.5
//        max_score: 0.6
//      }
//      signal_settings{
//        type: {custom: "custom_text"}
//        min_score: 0.9
//        max_score: 1.0
//      }
//    }
//    }
//  }
class SignalFusingCalculator : public mediapipe::CalculatorBase {
 public:
  SignalFusingCalculator() {}
  SignalFusingCalculator(const SignalFusingCalculator&) = delete;
  SignalFusingCalculator& operator=(const SignalFusingCalculator&) = delete;

  static ::mediapipe::Status GetContract(mediapipe::CalculatorContract* cc);
  mediapipe::Status Open(mediapipe::CalculatorContext* cc) override;
  mediapipe::Status Process(mediapipe::CalculatorContext* cc) override;
  mediapipe::Status Close(mediapipe::CalculatorContext* cc) override;

 private:
  mediapipe::Status ProcessScene(mediapipe::CalculatorContext* cc);
  SignalFusingCalculatorOptions options_;
  std::map<std::string, SignalSettings> settings_by_type_;
  std::vector<Frame> scene_frames_;
};
REGISTER_CALCULATOR(SignalFusingCalculator);

namespace {
std::string CreateSettingsKey(const SignalType& signal_type) {
  if (signal_type.has_standard()) {
    return "standard_" + std::to_string(signal_type.standard());
  } else {
    return "custom_" + signal_type.custom();
  }
}
std::string CreateKey(const InputSignal& detection) {
  std::string id_source = std::to_string(detection.source);
  std::string id_signal = std::to_string(detection.signal.tracking_id());
  std::string id = id_source + ":" + id_signal;
  return id;
}

}  // namespace

mediapipe::Status SignalFusingCalculator::Open(
    mediapipe::CalculatorContext* cc) {
  options_ = cc->Options<SignalFusingCalculatorOptions>();
  for (const auto& setting : options_.signal_settings()) {
    settings_by_type_[CreateSettingsKey(setting.type())] = setting;
  }
  return ::mediapipe::OkStatus();
}

mediapipe::Status SignalFusingCalculator::Close(
    mediapipe::CalculatorContext* cc) {
  if (!scene_frames_.empty()) {
    MP_RETURN_IF_ERROR(ProcessScene(cc));
    scene_frames_.clear();
  }
  return ::mediapipe::OkStatus();
}

mediapipe::Status SignalFusingCalculator::ProcessScene(
    mediapipe::CalculatorContext* cc) {
  std::map<std::string, int> detection_count;
  std::map<std::string, float> multiframe_score;
  // Create a unified score for all items with temporal ids.
  for (const Frame& frame : scene_frames_) {
    for (const auto& detection : frame.input_detections) {
      if (detection.signal.has_tracking_id()) {
        // Create key for each detector type
        if (detection_count.find(CreateKey(detection)) ==
            detection_count.end()) {
          multiframe_score[CreateKey(detection)] = 0.0;
          detection_count[CreateKey(detection)] = 0;
        }
        multiframe_score[CreateKey(detection)] += detection.signal.score();
        detection_count[CreateKey(detection)]++;
      }
    }
  }

  // Average scores.
  for (auto iterator = multiframe_score.begin();
       iterator != multiframe_score.end(); iterator++) {
    multiframe_score[iterator->first] =
        iterator->second / detection_count[iterator->first];
  }

  // Process detections.
  for (const Frame& frame : scene_frames_) {
    std::unique_ptr<DetectionSet> processed_detections(new DetectionSet());
    for (auto detection : frame.input_detections) {
      float score = detection.signal.score();
      if (detection.signal.has_tracking_id()) {
        std::string id_source = std::to_string(detection.source);
        std::string id_signal = std::to_string(detection.signal.tracking_id());
        std::string id = id_source + ":" + id_signal;
        score = multiframe_score[id];
      }
      // Normalize within range.
      float min_value = 0.0;
      float max_value = 1.0;

      auto settings_it = settings_by_type_.find(
          CreateSettingsKey(detection.signal.signal_type()));
      if (settings_it != settings_by_type_.end()) {
        min_value = settings_it->second.min_score();
        max_value = settings_it->second.max_score();
        detection.signal.set_is_required(settings_it->second.is_required());
      }

      float final_score = score * (max_value - min_value) + min_value;
      detection.signal.set_score(final_score);
      *processed_detections->add_detections() = detection.signal;
    }
    cc->Outputs().Index(0).Add(processed_detections.release(), frame.time);
  }

  return ::mediapipe::OkStatus();
}

mediapipe::Status SignalFusingCalculator::Process(
    mediapipe::CalculatorContext* cc) {
  bool is_boundary = false;
  if (!cc->Inputs().Index(0).Value().IsEmpty()) {
    is_boundary = cc->Inputs().Index(0).Get<bool>();
  }

  if (is_boundary || scene_frames_.size() > options_.max_scene_size()) {
    MP_RETURN_IF_ERROR(ProcessScene(cc));
    scene_frames_.clear();
  }

  Frame frame;
  for (int i = 1; i < cc->Inputs().NumEntries(); ++i) {
    const Packet& packet = cc->Inputs().Index(i).Value();
    if (packet.IsEmpty()) {
      continue;
    }
    const auto& detection_set = packet.Get<autoflip::DetectionSet>();
    for (const auto& detection : detection_set.detections()) {
      InputSignal input;
      input.signal = detection;
      input.source = i;
      frame.input_detections.push_back(input);
    }
  }
  frame.time = cc->InputTimestamp();
  scene_frames_.push_back(frame);

  return ::mediapipe::OkStatus();
}

::mediapipe::Status SignalFusingCalculator::GetContract(
    mediapipe::CalculatorContract* cc) {
  cc->Inputs().Index(0).Set<bool>();
  for (int i = 1; i < cc->Inputs().NumEntries(); ++i) {
    cc->Inputs().Index(i).Set<autoflip::DetectionSet>();
  }
  cc->Outputs().Index(0).Set<autoflip::DetectionSet>();
  return ::mediapipe::OkStatus();
}

}  // namespace autoflip
}  // namespace mediapipe
