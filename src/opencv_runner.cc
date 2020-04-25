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
#include <opencv2/objdetect.hpp>
#include "mediapipe/framework/port/opencv_highgui_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/opencv_video_inc.h"
#include "mediapipe/framework/port/opencv_video_inc.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/port/commandlineflags.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status.h"

constexpr char kInputStream[] = "input_video";
constexpr char kOutputStream[] = "output_video";

DEFINE_string(
    calculator_graph_config_file, "",
    "Name of file containing text format CalculatorGraphConfig proto.");

DEFINE_string(output_device, "/dev/video4", "V4L2 device to which the output will be written");

class Detector
{

    cv::CascadeClassifier classifier;

public:
    Detector()
    {
        if (!classifier.load("src/haarcascade_frontalface_alt2.xml"))
        {
            LOG(ERROR) << "Error loading cascade model";
        }
    }

    void detect(cv::InputArray input, cv::OutputArray output)
    {
        cv::Mat mat;
        cv::cvtColor(input, mat, cv::COLOR_BGR2GRAY);
        cv::equalizeHist(mat, mat);

        std::vector<cv::Rect> found;
        classifier.detectMultiScale(mat, found, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, cv::Size(30, 30));
        input.copyTo(mat);
        if (found.size() > 1)
        {
            cv::rectangle(mat, found[0], cv::Scalar(0, 255, 0), 2);
        }
        mat.copyTo(output);
    }
};

void RunMPPGraph()
{
    printf("Initialize the camera or load the video.\n");
    cv::VideoCapture capture;
    capture.open(0);
    // RET_CHECK(capture.isOpened());

    V4L2DeviceParameters param(FLAGS_output_device.c_str(), V4L2_PIX_FMT_YUV420, 640, 480, capture.get(cv::CAP_PROP_FPS), 1);
    V4l2Output *video_output = V4l2Output::create(param, V4l2Access::IOTYPE_MMAP);
    video_output->queryFormat();

#if (CV_MAJOR_VERSION >= 3) && (CV_MINOR_VERSION >= 2)
    capture.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    capture.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    capture.set(cv::CAP_PROP_FPS, 30);
#endif
    printf("\nStart grabbing and processing frames.\n");
    bool grab_frames = true;
    long frameCounter = 0;
    std::time_t timeBegin = std::time(0);
    int tick = 0;
    Detector detector;
    cv::Mat result; // segmentation result (4 possible values)

    while (grab_frames)
    {

        // Capture opencv camera or video frame.
        cv::Mat camera_frame_raw;
        capture >> camera_frame_raw;
        if (camera_frame_raw.empty())
            break; // End of video.

        cv::Mat camera_frame;
        cv::Mat bgModel, fgModel; // the models (internally used)
        // detector.detect(camera_frame_raw, camera_frame);
        // define bounding rectangle
        cv::Rect rectangle(40, 90, camera_frame_raw.cols - 80, camera_frame_raw.rows - 170);

        if (tick < 10)
        {
            cv::grabCut(camera_frame_raw, // input image
                        result,           // segmentation result
                        rectangle,        // rectangle containing foreground
                        bgModel, fgModel, // models
                        1,                // number of iterations
                        cv::GC_INIT_WITH_RECT);
        }
        else
        {
            cv::grabCut(camera_frame_raw, // input image
                        result,           // segmentation result
                        rectangle,        // rectangle containing foreground
                        bgModel, fgModel, // models
                        1,                // number of iterations
                        cv::GC_EVAL_FREEZE_MODEL);
        }
        camera_frame_raw.copyTo(camera_frame, result); // bg pixels not copied
        cv::cvtColor(camera_frame, camera_frame, cv::COLOR_BGR2YUV_I420);
        uchar *buffer = camera_frame.isContinuous() ? camera_frame.data : camera_frame.clone().data;
        uint buffer_size = camera_frame.total() * camera_frame.channels();
        size_t nb = video_output->write((char *)buffer, buffer_size);
        frameCounter++;
        std::time_t timeNow = std::time(0) - timeBegin;
        if (timeNow - tick >= 1)
        {
            tick++;
            LOG(INFO) << "FPS: " << frameCounter;
            frameCounter = 0;
        }
        // Press any key to exit.
        const int pressed_key = cv::waitKey(5);
        if (pressed_key >= 0 && pressed_key != 255)
            grab_frames = false;
    }

    printf("Shutting down.\n");
}

int main(int argc, char **argv)
{
    RunMPPGraph();
    return EXIT_SUCCESS;
}
