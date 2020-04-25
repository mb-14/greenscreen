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

#include <memory>
#include <string>

#include "mediapipe/calculators/image/bilateral_filter_calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_options.pb.h"
#include "mediapipe/framework/formats/image_format.pb.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/port/logging.h"
#include "mediapipe/framework/port/opencv_core_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/port/vector.h"

#if !defined(MEDIAPIPE_DISABLE_GPU)
#include "mediapipe/gpu/gl_calculator_helper.h"
#include "mediapipe/gpu/gl_simple_shaders.h"
#include "mediapipe/gpu/shader_util.h"
#endif  //  !MEDIAPIPE_DISABLE_GPU

namespace mediapipe {

namespace {
constexpr char kInputFrameTag[] = "IMAGE";
constexpr char kInputGuideTag[] = "GUIDE";
constexpr char kOutputFrameTag[] = "IMAGE";

constexpr char kInputFrameTagGpu[] = "IMAGE_GPU";
constexpr char kInputGuideTagGpu[] = "GUIDE_GPU";
constexpr char kOutputFrameTagGpu[] = "IMAGE_GPU";

enum { ATTRIB_VERTEX, ATTRIB_TEXTURE_POSITION, NUM_ATTRIBUTES };
}  // namespace

// A calculator for applying a bilateral filter to an image,
// with an optional guide image (joint blateral).
//
// Inputs:
//   One of the following two IMAGE tags:
//   IMAGE: ImageFrame containing input image - Grayscale or RGB only.
//   IMAGE_GPU: GpuBuffer containing input image - Grayscale, RGB or RGBA.
//
//   GUIDE (optional): ImageFrame guide image used to filter IMAGE. (N/A).
//   GUIDE_GPU (optional): GpuBuffer guide image used to filter IMAGE_GPU.
//
// Output:
//   One of the following two tags:
//   IMAGE:     A filtered ImageFrame - Same as input.
//   IMAGE_GPU:  A filtered GpuBuffer - RGBA
//
// Options:
//   sigma_space: Pixel radius: use (sigma_space*2+1)x(sigma_space*2+1) window.
//                This should be set based on output image pixel space.
//   sigma_color: Color variance: normalized [0-1] color difference allowed.
//
// Notes:
//   * When GUIDE is present, the output image is same size as GUIDE image;
//     otherwise, the output image is same size as input image.
//   * On GPU the kernel window is subsampled by approximately sqrt(sigma_space)
//     i.e. the step size is ~sqrt(sigma_space),
//     prioritizing performance > quality.
//   * TODO: Add CPU path for joint filter.
//
class BilateralFilterCalculator : public CalculatorBase {
 public:
  BilateralFilterCalculator() = default;
  ~BilateralFilterCalculator() override = default;

  static ::mediapipe::Status GetContract(CalculatorContract* cc);

  // From Calculator.
  ::mediapipe::Status Open(CalculatorContext* cc) override;
  ::mediapipe::Status Process(CalculatorContext* cc) override;
  ::mediapipe::Status Close(CalculatorContext* cc) override;

 private:
  ::mediapipe::Status RenderGpu(CalculatorContext* cc);
  ::mediapipe::Status RenderCpu(CalculatorContext* cc);

  ::mediapipe::Status GlSetup(CalculatorContext* cc);
  void GlRender(CalculatorContext* cc);

  mediapipe::BilateralFilterCalculatorOptions options_;
  float sigma_color_ = -1.f;
  float sigma_space_ = -1.f;

  bool use_gpu_ = false;
  bool gpu_initialized_ = false;
#if !defined(MEDIAPIPE_DISABLE_GPU)
  mediapipe::GlCalculatorHelper gpu_helper_;
  GLuint program_ = 0;
  GLuint program_joint_ = 0;
#endif  //  !MEDIAPIPE_DISABLE_GPU
};
REGISTER_CALCULATOR(BilateralFilterCalculator);

::mediapipe::Status BilateralFilterCalculator::GetContract(
    CalculatorContract* cc) {
  CHECK_GE(cc->Inputs().NumEntries(), 1);

  if (cc->Inputs().HasTag(kInputFrameTag) &&
      cc->Inputs().HasTag(kInputFrameTagGpu)) {
    return ::mediapipe::InternalError("Cannot have multiple input images.");
  }
  if (cc->Inputs().HasTag(kInputFrameTagGpu) !=
      cc->Outputs().HasTag(kOutputFrameTagGpu)) {
    return ::mediapipe::InternalError("GPU output must have GPU input.");
  }

  bool use_gpu = false;

  // Input image to filter.
#if !defined(MEDIAPIPE_DISABLE_GPU)
  if (cc->Inputs().HasTag(kInputFrameTagGpu)) {
    cc->Inputs().Tag(kInputFrameTagGpu).Set<mediapipe::GpuBuffer>();
    use_gpu |= true;
  }
#endif  //  !MEDIAPIPE_DISABLE_GPU
  if (cc->Inputs().HasTag(kInputFrameTag)) {
    cc->Inputs().Tag(kInputFrameTag).Set<ImageFrame>();
  }

  // Input guide image mask (optional)
#if !defined(MEDIAPIPE_DISABLE_GPU)
  if (cc->Inputs().HasTag(kInputGuideTagGpu)) {
    cc->Inputs().Tag(kInputGuideTagGpu).Set<mediapipe::GpuBuffer>();
    use_gpu |= true;
  }
#endif  //  !MEDIAPIPE_DISABLE_GPU
  if (cc->Inputs().HasTag(kInputGuideTag)) {
    cc->Inputs().Tag(kInputGuideTag).Set<ImageFrame>();
  }

  // Output image.
#if !defined(MEDIAPIPE_DISABLE_GPU)
  if (cc->Outputs().HasTag(kOutputFrameTagGpu)) {
    cc->Outputs().Tag(kOutputFrameTagGpu).Set<mediapipe::GpuBuffer>();
    use_gpu |= true;
  }
#endif  //  !MEDIAPIPE_DISABLE_GPU
  if (cc->Outputs().HasTag(kOutputFrameTag)) {
    cc->Outputs().Tag(kOutputFrameTag).Set<ImageFrame>();
  }

  if (use_gpu) {
#if !defined(MEDIAPIPE_DISABLE_GPU)
    MP_RETURN_IF_ERROR(mediapipe::GlCalculatorHelper::UpdateContract(cc));
#endif  //  !MEDIAPIPE_DISABLE_GPU
  }

  return ::mediapipe::OkStatus();
}

::mediapipe::Status BilateralFilterCalculator::Open(CalculatorContext* cc) {
  cc->SetOffset(TimestampDiff(0));

  options_ = cc->Options<mediapipe::BilateralFilterCalculatorOptions>();

  if (cc->Inputs().HasTag(kInputFrameTagGpu) &&
      cc->Outputs().HasTag(kOutputFrameTagGpu)) {
#if !defined(MEDIAPIPE_DISABLE_GPU)
    use_gpu_ = true;
#else
    RET_CHECK_FAIL() << "GPU processing not enabled.";
#endif
  }

  sigma_color_ = options_.sigma_color();
  sigma_space_ = options_.sigma_space();
  CHECK_GE(sigma_color_, 0.0);
  CHECK_GE(sigma_space_, 0.0);
  if (!use_gpu_) sigma_color_ *= 255.0;

  if (use_gpu_) {
#if !defined(MEDIAPIPE_DISABLE_GPU)
    MP_RETURN_IF_ERROR(gpu_helper_.Open(cc));
#endif  //  !MEDIAPIPE_DISABLE_GPU
  }

  return ::mediapipe::OkStatus();
}

::mediapipe::Status BilateralFilterCalculator::Process(CalculatorContext* cc) {
  if (use_gpu_) {
#if !defined(MEDIAPIPE_DISABLE_GPU)
    MP_RETURN_IF_ERROR(
        gpu_helper_.RunInGlContext([this, cc]() -> ::mediapipe::Status {
          if (!gpu_initialized_) {
            MP_RETURN_IF_ERROR(GlSetup(cc));
            gpu_initialized_ = true;
          }
          MP_RETURN_IF_ERROR(RenderGpu(cc));
          return ::mediapipe::OkStatus();
        }));
#endif  //  !MEDIAPIPE_DISABLE_GPU
  } else {
    MP_RETURN_IF_ERROR(RenderCpu(cc));
  }

  return ::mediapipe::OkStatus();
}

::mediapipe::Status BilateralFilterCalculator::Close(CalculatorContext* cc) {
#if !defined(MEDIAPIPE_DISABLE_GPU)
  gpu_helper_.RunInGlContext([this] {
    if (program_) glDeleteProgram(program_);
    program_ = 0;
    if (program_joint_) glDeleteProgram(program_joint_);
    program_joint_ = 0;
  });
#endif  //  !MEDIAPIPE_DISABLE_GPU

  return ::mediapipe::OkStatus();
}

::mediapipe::Status BilateralFilterCalculator::RenderCpu(
    CalculatorContext* cc) {
  if (cc->Inputs().Tag(kInputFrameTag).IsEmpty()) {
    return ::mediapipe::OkStatus();
  }

  const auto& input_frame = cc->Inputs().Tag(kInputFrameTag).Get<ImageFrame>();
  auto input_mat = mediapipe::formats::MatView(&input_frame);

  // Only 1 or 3 channel images supported by OpenCV.
  if ((input_mat.channels() == 1 || input_mat.channels() == 3)) {
    return ::mediapipe::InternalError(
        "CPU filtering supports only 1 or 3 channel input images.");
  }

  auto output_frame = absl::make_unique<ImageFrame>(
      input_frame.Format(), input_mat.cols, input_mat.rows);
  const bool has_guide_image = cc->Inputs().HasTag(kInputGuideTag) &&
                               !cc->Inputs().Tag(kInputGuideTag).IsEmpty();

  if (has_guide_image) {
    // cv::jointBilateralFilter() is in contrib module 'ximgproc'.
    return ::mediapipe::UnimplementedError(
        "CPU joint filtering support is not implemented yet.");
  } else {
    auto output_mat = mediapipe::formats::MatView(output_frame.get());
    // Prefer setting 'd = sigma_space * 2' to match GPU definition of radius.
    cv::bilateralFilter(input_mat, output_mat, /*d=*/sigma_space_ * 2.0,
                        sigma_color_, sigma_space_);
  }

  cc->Outputs()
      .Tag(kOutputFrameTag)
      .Add(output_frame.release(), cc->InputTimestamp());
  return ::mediapipe::OkStatus();
}

::mediapipe::Status BilateralFilterCalculator::RenderGpu(
    CalculatorContext* cc) {
  if (cc->Inputs().Tag(kInputFrameTagGpu).IsEmpty()) {
    return ::mediapipe::OkStatus();
  }
#if !defined(MEDIAPIPE_DISABLE_GPU)
  const auto& input_frame =
      cc->Inputs().Tag(kInputFrameTagGpu).Get<mediapipe::GpuBuffer>();
  auto input_texture = gpu_helper_.CreateSourceTexture(input_frame);

  mediapipe::GlTexture output_texture;
  const bool has_guide_image = cc->Inputs().HasTag(kInputGuideTagGpu) &&
                               !cc->Inputs().Tag(kInputGuideTagGpu).IsEmpty();

  // Setup textures and Update image in GPU shader.
  if (has_guide_image) {
    // joint bilateral filter
    glUseProgram(program_joint_);
    const auto& guide_image =
        cc->Inputs().Tag(kInputGuideTagGpu).Get<mediapipe::GpuBuffer>();
    auto guide_texture = gpu_helper_.CreateSourceTexture(guide_image);
    glUniform2f(glGetUniformLocation(program_joint_, "texel_size_guide"),
                1.0 / guide_image.width(), 1.0 / guide_image.height());
    output_texture = gpu_helper_.CreateDestinationTexture(
        guide_image.width(), guide_image.height(),
        mediapipe::GpuBufferFormat::kBGRA32);
    gpu_helper_.BindFramebuffer(output_texture);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, input_texture.name());
    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, guide_texture.name());
    GlRender(cc);
    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, 0);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, 0);
    guide_texture.Release();
  } else {
    // regular bilateral filter
    glUseProgram(program_);
    glUniform2f(glGetUniformLocation(program_, "texel_size"),
                1.0 / input_frame.width(), 1.0 / input_frame.height());
    output_texture = gpu_helper_.CreateDestinationTexture(
        input_frame.width(), input_frame.height(),
        mediapipe::GpuBufferFormat::kBGRA32);
    gpu_helper_.BindFramebuffer(output_texture);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, input_texture.name());
    GlRender(cc);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, 0);
  }
  glFlush();

  // Send out image as GPU packet.
  auto output_frame = output_texture.GetFrame<mediapipe::GpuBuffer>();
  cc->Outputs()
      .Tag(kOutputFrameTagGpu)
      .Add(output_frame.release(), cc->InputTimestamp());

  // Cleanup
  input_texture.Release();
  output_texture.Release();
#endif  //  !MEDIAPIPE_DISABLE_GPU

  return ::mediapipe::OkStatus();
}

void BilateralFilterCalculator::GlRender(CalculatorContext* cc) {
#if !defined(MEDIAPIPE_DISABLE_GPU)
  static const GLfloat square_vertices[] = {
      -1.0f, -1.0f,  // bottom left
      1.0f,  -1.0f,  // bottom right
      -1.0f, 1.0f,   // top left
      1.0f,  1.0f,   // top right
  };
  static const GLfloat texture_vertices[] = {
      0.0f, 0.0f,  // bottom left
      1.0f, 0.0f,  // bottom right
      0.0f, 1.0f,  // top left
      1.0f, 1.0f,  // top right
  };

  // vertex storage
  GLuint vbo[2];
  glGenBuffers(2, vbo);
  GLuint vao;
  glGenVertexArrays(1, &vao);
  glBindVertexArray(vao);

  // vbo 0
  glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);
  glBufferData(GL_ARRAY_BUFFER, 4 * 2 * sizeof(GLfloat), square_vertices,
               GL_STATIC_DRAW);
  glEnableVertexAttribArray(ATTRIB_VERTEX);
  glVertexAttribPointer(ATTRIB_VERTEX, 2, GL_FLOAT, 0, 0, nullptr);

  // vbo 1
  glBindBuffer(GL_ARRAY_BUFFER, vbo[1]);
  glBufferData(GL_ARRAY_BUFFER, 4 * 2 * sizeof(GLfloat), texture_vertices,
               GL_STATIC_DRAW);
  glEnableVertexAttribArray(ATTRIB_TEXTURE_POSITION);
  glVertexAttribPointer(ATTRIB_TEXTURE_POSITION, 2, GL_FLOAT, 0, 0, nullptr);

  // draw
  glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

  // cleanup
  glDisableVertexAttribArray(ATTRIB_VERTEX);
  glDisableVertexAttribArray(ATTRIB_TEXTURE_POSITION);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glBindVertexArray(0);
  glDeleteVertexArrays(1, &vao);
  glDeleteBuffers(2, vbo);

#endif  //  !MEDIAPIPE_DISABLE_GPU
}

::mediapipe::Status BilateralFilterCalculator::GlSetup(CalculatorContext* cc) {
#if !defined(MEDIAPIPE_DISABLE_GPU)
  const GLint attr_location[NUM_ATTRIBUTES] = {
      ATTRIB_VERTEX,
      ATTRIB_TEXTURE_POSITION,
  };
  const GLchar* attr_name[NUM_ATTRIBUTES] = {
      "position",
      "texture_coordinate",
  };

  // We bake our sigma values directly into the shader, so the GLSL compiler can
  // optimize appropriately.
  std::string sigma_options_string =
      "const float sigma_space = " + std::to_string(sigma_space_) +
      "; const float sigma_color = " + std::to_string(sigma_color_) + ";\n";

  // Shader to do bilateral filtering on input image based on sigma space/color.
  // Large kernel sizes are subsampled based on sqrt(sigma_space) window size,
  // denoted as 'sparsity' below.
  const std::string frag_src = GLES_VERSION_COMPAT
                               R"(
  #if __VERSION__ < 130
    #define in varying
  #endif  // __VERSION__ < 130

  #ifdef GL_ES
    #define fragColor gl_FragColor
    precision highp float;
  #else
    #define lowp
    #define mediump
    #define highp
    #define texture2D texture
    out vec4 fragColor;
  #endif  // defined(GL_ES)

    in vec2 sample_coordinate;
    uniform sampler2D input_frame;
)" + sigma_options_string + R"(
    uniform vec2 texel_size;

    const float kSparsityFactor = 0.66;  // Higher is more sparse.
    const float sparsity = max(1.0, sqrt(sigma_space) * kSparsityFactor);
    const float step = sparsity;
    const float radius = sigma_space;
    const float offset = (step > 1.0) ? (step * 0.5) : (0.0);

    float gaussian(float x, float sigma) {
      float coeff = -0.5 / (sigma * sigma * 4.0 + 1.0e-6);
      return exp((x * x) * coeff);
    }

    void main() {
      vec2 center_uv = sample_coordinate;
      vec3 center_val = texture2D(input_frame, center_uv).rgb;
      vec3 new_val = vec3(0.0);

      float space_weight = 0.0;
      float color_weight = 0.0;
      float total_weight = 0.0;

      float sigma_texel = max(texel_size.x, texel_size.y) * sigma_space;
      // Subsample kernel space.
      for (float i = -radius+offset; i <= radius; i+=step) {
        for (float j = -radius+offset; j <= radius; j+=step) {
          vec2 shift = vec2(j, i) * texel_size;
          vec2 uv = vec2(center_uv + shift);
          vec3 val = texture2D(input_frame, uv).rgb;

          space_weight = gaussian(distance(center_uv, uv), sigma_texel);
          color_weight = gaussian(distance(center_val, val), sigma_color);
          total_weight += space_weight * color_weight;

          new_val += vec3(space_weight * color_weight) * val;
        }
      }
      new_val /= vec3(total_weight);

      fragColor = vec4(new_val, 1.0);
    }
  )";

  // Create shader program and set parameters.
  mediapipe::GlhCreateProgram(mediapipe::kBasicVertexShader, frag_src.c_str(),
                              NUM_ATTRIBUTES, (const GLchar**)&attr_name[0],
                              attr_location, &program_);
  RET_CHECK(program_) << "Problem initializing the program.";
  glUseProgram(program_);
  glUniform1i(glGetUniformLocation(program_, "input_frame"), 1);

  // Shader to do joint bilateral filtering on input image based on
  // sigma space/color, and a Guide image.
  // Large kernel sizes are subsampled based on sqrt(sigma_space) window size,
  // denoted as 'sparsity' below.
  const std::string joint_frag_src = GLES_VERSION_COMPAT
                                     R"(
  #if __VERSION__ < 130
    #define in varying
  #endif  // __VERSION__ < 130

  #ifdef GL_ES
    #define fragColor gl_FragColor
    precision highp float;
  #else
    #define lowp
    #define mediump
    #define highp
    #define texture2D texture
    out vec4 fragColor;
  #endif  // defined(GL_ES)

    in vec2 sample_coordinate;
    uniform sampler2D input_frame;
    uniform sampler2D guide_frame;
)" + sigma_options_string + R"(
    uniform vec2 texel_size_guide; // size of guide and resulting filtered image

    const float kSparsityFactor = 0.66;  // Higher is more sparse.
    const float sparsity = max(1.0, sqrt(sigma_space) * kSparsityFactor);
    const float step = sparsity;
    const float radius = sigma_space;
    const float offset = (step > 1.0) ? (step * 0.5) : (0.0);

    float gaussian(float x, float sigma) {
      float coeff = -0.5 / (sigma * sigma * 4.0 + 1.0e-6);
      return exp((x * x) * coeff);
    }

    void main() {
      vec2 center_uv = sample_coordinate;
      vec3 center_val = texture2D(guide_frame, center_uv).rgb;
      vec3 new_val = vec3(0.0);

      float space_weight = 0.0;
      float color_weight = 0.0;
      float total_weight = 0.0;

      float sigma_texel = max(texel_size_guide.x, texel_size_guide.y) * sigma_space;
      // Subsample kernel space.
      for (float i = -radius+offset; i <= radius; i+=step) {
        for (float j = -radius+offset; j <= radius; j+=step) {
          vec2 shift = vec2(j, i) * texel_size_guide;
          vec2 uv = vec2(center_uv + shift);
          vec3 guide_val = texture2D(guide_frame, uv).rgb;
          vec3 out_val = texture2D(input_frame, uv).rgb;

          space_weight = gaussian(distance(center_uv, uv), sigma_texel);
          color_weight = gaussian(distance(center_val, guide_val), sigma_color);
          total_weight += space_weight * color_weight;

          new_val += vec3(space_weight * color_weight) * out_val;
        }
      }
      new_val /= vec3(total_weight);

      fragColor = vec4(new_val, 1.0);
    }
  )";

  // Create shader program and set parameters.
  mediapipe::GlhCreateProgram(
      mediapipe::kBasicVertexShader, joint_frag_src.c_str(), NUM_ATTRIBUTES,
      (const GLchar**)&attr_name[0], attr_location, &program_joint_);
  RET_CHECK(program_joint_) << "Problem initializing the program.";
  glUseProgram(program_joint_);
  glUniform1i(glGetUniformLocation(program_joint_, "input_frame"), 1);
  glUniform1i(glGetUniformLocation(program_joint_, "guide_frame"), 2);

#endif  //  !MEDIAPIPE_DISABLE_GPU

  return ::mediapipe::OkStatus();
}

}  // namespace mediapipe
