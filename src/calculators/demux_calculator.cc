#include "src/calculators/demux_calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/ret_check.h"

namespace mediapipe {

class DemuxCalculator : public CalculatorBase {
 public:
  static ::mediapipe::Status GetContract(CalculatorContract* cc) {
    RET_CHECK_EQ(cc->Inputs().NumEntries(), 1);
    cc->Inputs().Index(0).SetAny();
    if (cc->Outputs().HasTag("SELECT")) {
      cc->Outputs().Tag("SELECT").Set<int>();
    }
    for (CollectionItemId id = cc->Outputs().BeginId("OUTPUT");
         id < cc->Outputs().EndId("OUTPUT"); ++id) {
      cc->Outputs().Get(id).SetSameAs(&cc->Inputs().Index(0));
    }
    return ::mediapipe::OkStatus();
  }

  ::mediapipe::Status Open(CalculatorContext* cc) override {
    select_output_ = cc->Outputs().GetId("SELECT", 0);
    const auto& options = cc->Options<::mediapipe::DemuxCalculatorOptions>();
    output_data_stream_index_ = options.output_data_stream_index();
    return ::mediapipe::OkStatus();
  }

  ::mediapipe::Status Process(CalculatorContext* cc) override {
    cc->Outputs()
        .Get("OUTPUT", output_data_stream_index_)
        .AddPacket(cc->Inputs().Index(0).Value());
    if (select_output_.IsValid()) {
      cc->Outputs()
          .Get(select_output_)
          .Add(new int(output_data_stream_index_), cc->InputTimestamp());
    }
    return ::mediapipe::OkStatus();
  }

 private:
  CollectionItemId select_output_;
  int32 output_data_stream_index_;
};

REGISTER_CALCULATOR(DemuxCalculator);

}
