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

#include "mediapipe/framework/tool/name_util.h"

#include <unordered_map>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "mediapipe/framework/port/map_util.h"
#include "mediapipe/framework/tool/validate_name.h"

namespace mediapipe {
namespace tool {

std::string GetUnusedNodeName(const CalculatorGraphConfig& config,
                              const std::string& node_name_base) {
  std::set<std::string> node_names;
  for (const ::mediapipe::CalculatorGraphConfig::Node& node : config.node()) {
    if (!node.name().empty()) {
      node_names.insert(node.name());
    }
  }

  std::string candidate = node_name_base;
  int iter = 2;
  while (ContainsKey(node_names, candidate)) {
    candidate =
        absl::StrCat(node_name_base, "_", absl::StrFormat("%02d", iter));
    ++iter;
  }
  return candidate;
}

std::string GetUnusedSidePacketName(
    const CalculatorGraphConfig& config,
    const std::string& input_side_packet_name_base) {
  std::unordered_map<std::string,
                     std::vector<::mediapipe::CalculatorGraphConfig::Node>>
      input_side_packets;
  for (const ::mediapipe::CalculatorGraphConfig::Node& node : config.node()) {
    for (const auto& tag_and_name : node.input_side_packet()) {
      std::string tag;
      std::string name;
      int index;
      MEDIAPIPE_CHECK_OK(ParseTagIndexName(tag_and_name, &tag, &index, &name));
      input_side_packets[name].push_back(node);
    }
  }
  std::string candidate = input_side_packet_name_base;
  int iter = 2;
  while (::mediapipe::ContainsKey(input_side_packets, candidate)) {
    candidate = absl::StrCat(input_side_packet_name_base, "_",
                             absl::StrFormat("%02d", iter));
    ++iter;
  }
  return candidate;
}

std::string CanonicalNodeName(const CalculatorGraphConfig& graph_config,
                              int node_id) {
  const auto& node_config = graph_config.node(node_id);
  std::string node_name = node_config.name().empty() ? node_config.calculator()
                                                     : node_config.name();
  int count = 0;
  int sequence = 0;
  for (int i = 0; i < graph_config.node_size(); i++) {
    const auto& current_node_config = graph_config.node(i);
    std::string current_node_name = current_node_config.name().empty()
                                        ? current_node_config.calculator()
                                        : current_node_config.name();
    if (node_name == current_node_name) {
      ++count;
      if (i < node_id) {
        ++sequence;
      }
    }
  }
  if (count <= 1) {
    return node_name;
  }
  return absl::StrCat(node_name, "_", sequence + 1);
}

}  // namespace tool
}  // namespace mediapipe
