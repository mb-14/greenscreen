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
// An example of sending OpenCV webcam frames into a MediaPipe graph.
#include <cstdlib>
#include <libv4l2cpp/V4l2Output.h>
#include <libv4l2cpp/V4l2Device.h>
#include "mediapipe/framework/port/opencv_highgui_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/opencv_video_inc.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/port/commandlineflags.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/gpu/gl_calculator_helper.h"
#include "mediapipe/gpu/gpu_buffer.h"
#include "mediapipe/gpu/gpu_shared_data_internal.h"
#include <chrono> // for high_resolution_clock

constexpr char kInputStream[] = "input_video";
constexpr char kOutputStream[] = "output_video";
using elapsed_resolution = std::chrono::milliseconds;
DEFINE_string(
    calculator_graph_config_file, "",
    "Name of file containing text format CalculatorGraphConfig proto.");

DEFINE_string(output_device, "/dev/video4", "V4L2 device to which the output will be written");

::mediapipe::Status RunMPPGraph()
{
    std::string calculator_graph_config_contents;
    MP_RETURN_IF_ERROR(mediapipe::file::GetContents(
        FLAGS_calculator_graph_config_file, &calculator_graph_config_contents));
    LOG(INFO) << "Get calculator graph config contents: "
              << calculator_graph_config_contents;
    mediapipe::CalculatorGraphConfig config =
        mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(
            calculator_graph_config_contents);

    LOG(INFO) << "Initialize the calculator graph.";
    mediapipe::CalculatorGraph graph;
    MP_RETURN_IF_ERROR(graph.Initialize(config));

    LOG(INFO) << "Initialize the GPU.";
    ASSIGN_OR_RETURN(auto gpu_resources, mediapipe::GpuResources::Create());
    MP_RETURN_IF_ERROR(graph.SetGpuResources(std::move(gpu_resources)));
    mediapipe::GlCalculatorHelper gpu_helper;
    gpu_helper.InitializeForTest(graph.GetGpuResources().get());

    printf("Initialize the camera or load the video.\n");
    cv::VideoCapture capture;
    capture.open(0);
    RET_CHECK(capture.isOpened());

    V4L2DeviceParameters param(FLAGS_output_device.c_str(), V4L2_PIX_FMT_YUV420, 640, 480, capture.get(cv::CAP_PROP_FPS), 1);
    V4l2Output *video_output = V4l2Output::create(param, V4l2Access::IOTYPE_MMAP);
    video_output->queryFormat();

#if (CV_MAJOR_VERSION >= 3) && (CV_MINOR_VERSION >= 2)
    capture.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    capture.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    capture.set(cv::CAP_PROP_FPS, 30);
#endif

    LOG(INFO) << "Start running the calculator graph.";
    ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller poller,
                     graph.AddOutputStreamPoller(kOutputStream));
    graph.StartRun({});
    printf("\nStart grabbing and processing frames.\n");
    bool grab_frames = true;
    long frameCounter = 0;
    std::time_t timeBegin = std::time(0);
    int tick = 0;
    cv::Mat background = cv::imread("backgrounds/settlers_of_catan.jpg");
    cv::cvtColor(background, background, cv::COLOR_BGR2RGB);
    auto background_frame = absl::make_unique<mediapipe::ImageFrame>(
        mediapipe::ImageFormat::SRGB, background.cols, background.rows,
        mediapipe::ImageFrame::kGlDefaultAlignmentBoundary);
    cv::Mat background_frame_mat = mediapipe::formats::MatView(background_frame.get());
    background.copyTo(background_frame_mat);
    std::chrono::high_resolution_clock clock;
    while (grab_frames)
    {

        // Capture opencv camera or video frame.
        cv::Mat camera_frame_raw;
        capture >> camera_frame_raw;
        if (camera_frame_raw.empty())
            break; // End of video.

        auto preprocessing_time_begin = clock.now();
        cv::Mat camera_frame;
        cv::cvtColor(camera_frame_raw, camera_frame, cv::COLOR_BGR2RGB);

        cv::flip(camera_frame, camera_frame, /*flipcode=HORIZONTAL*/ 1);

        // Wrap Mat into an ImageFrame.
        auto input_frame = absl::make_unique<mediapipe::ImageFrame>(
            mediapipe::ImageFormat::SRGB, camera_frame.cols, camera_frame.rows,
            mediapipe::ImageFrame::kGlDefaultAlignmentBoundary);
        cv::Mat input_frame_mat = mediapipe::formats::MatView(input_frame.get());
        camera_frame.copyTo(input_frame_mat);
        auto preprocessing_time = std::chrono::duration_cast<elapsed_resolution>(clock.now() - preprocessing_time_begin);

        auto graph_time_begin = clock.now();
        // Send image packet into the graph.
        size_t frame_timestamp_us =
            (double)cv::getTickCount() / (double)cv::getTickFrequency() * 1e6;
        gpu_helper.RunInGlContext([&input_frame, &background_frame, &frame_timestamp_us, &graph,
                                   &gpu_helper]() -> ::mediapipe::Status {
            // Convert ImageFrame to GpuBuffer.
            auto texture = gpu_helper.CreateSourceTexture(*input_frame.get());
            auto gpu_frame = texture.GetFrame<mediapipe::GpuBuffer>();
            glFlush();
            texture.Release();
            // Send GPU image packet into the graph.
            MP_RETURN_IF_ERROR(graph.AddPacketToInputStream(
                kInputStream, mediapipe::Adopt(gpu_frame.release())
                                  .At(mediapipe::Timestamp(frame_timestamp_us))));

            auto background_texture = gpu_helper.CreateSourceTexture(*background_frame.get());
            auto background_gpu_frame = background_texture.GetFrame<mediapipe::GpuBuffer>();
            graph.AddPacketToInputStream("background_image", mediapipe::Adopt(background_gpu_frame.release()).At(mediapipe::Timestamp(frame_timestamp_us)));
            return ::mediapipe::OkStatus();
        });
        // Get the graph result packet, or stop if that fails.
        mediapipe::Packet packet;
        if (!poller.Next(&packet))
            break;
        std::unique_ptr<mediapipe::ImageFrame> output_frame;

        // Convert GpuBuffer to ImageFrame.
        MP_RETURN_IF_ERROR(gpu_helper.RunInGlContext(
            [&packet, &output_frame, &gpu_helper]() -> ::mediapipe::Status {
                auto &gpu_frame = packet.Get<mediapipe::GpuBuffer>();
                auto texture = gpu_helper.CreateSourceTexture(gpu_frame);
                output_frame = absl::make_unique<mediapipe::ImageFrame>(
                    mediapipe::ImageFormatForGpuBufferFormat(gpu_frame.format()),
                    gpu_frame.width(), gpu_frame.height(),
                    mediapipe::ImageFrame::kGlDefaultAlignmentBoundary);
                gpu_helper.BindFramebuffer(texture);
                const auto info =
                    mediapipe::GlTextureInfoForGpuBufferFormat(gpu_frame.format(), 0);
                glReadPixels(0, 0, texture.width(), texture.height(), info.gl_format,
                             info.gl_type, output_frame->MutablePixelData());
                glFlush();
                texture.Release();
                return ::mediapipe::OkStatus();
            }));

        auto graph_time = std::chrono::duration_cast<elapsed_resolution>(clock.now() - graph_time_begin);

        auto postprocessing_being = clock.now();
        // Convert back to opencv for display or saving.
        cv::Mat output_frame_mat = mediapipe::formats::MatView(output_frame.get());
        cv::resize(output_frame_mat, output_frame_mat, cv::Size(640, 480));
        cv::flip(output_frame_mat, output_frame_mat, /*flipcode=HORIZONTAL*/ 1);
        cv::cvtColor(output_frame_mat, output_frame_mat, cv::COLOR_RGB2YUV_I420);
        uchar *buffer = output_frame_mat.isContinuous() ? output_frame_mat.data : output_frame_mat.clone().data;
        uint buffer_size = output_frame_mat.total() * output_frame_mat.channels();
        size_t nb = video_output->write((char *)buffer, buffer_size);
        auto postprocessing_time = std::chrono::duration_cast<elapsed_resolution>(clock.now() - postprocessing_being);
        frameCounter++;
        std::time_t timeNow = std::time(0) - timeBegin;
        if (timeNow - tick >= 1)
        {
            tick++;
            LOG(INFO) << "FPS: " << frameCounter;
            LOG(INFO) << "Preprocessing: " << preprocessing_time.count();
            LOG(INFO) << "Graph: " << graph_time.count();
            LOG(INFO) << "Postprocessing: " << postprocessing_time.count();

            frameCounter = 0;
        }
        // Press any key to exit.
        const int pressed_key = cv::waitKey(5);
        if (pressed_key >= 0 && pressed_key != 255)
            grab_frames = false;
    }

    printf("Shutting down.\n");
    MP_RETURN_IF_ERROR(graph.CloseInputStream(kInputStream));
    return graph.WaitUntilDone();
}

int main(int argc, char **argv)
{
    google::InitGoogleLogging(argv[0]);
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    ::mediapipe::Status run_status = RunMPPGraph();
    if (!run_status.ok())
    {
        LOG(ERROR) << "Failed to run the graph: " << run_status.message();
        return EXIT_FAILURE;
    }
    else
    {
        LOG(INFO) << "Success!";
    }
    return EXIT_SUCCESS;
}
