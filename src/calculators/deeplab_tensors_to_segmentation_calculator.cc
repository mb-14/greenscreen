#include <vector>

#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "mediapipe/calculators/tflite/util.h"
#include "mediapipe/framework/calculator_context.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/port/opencv_imgcodecs_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/util/resource_util.h"
#include "tensorflow/lite/interpreter.h"

#include "mediapipe/gpu/gl_calculator_helper.h"
#include "mediapipe/gpu/gl_simple_shaders.h"
#include "mediapipe/gpu/shader_util.h"
#include "tensorflow/lite/delegates/gpu/gl/gl_buffer.h"
#include "tensorflow/lite/delegates/gpu/gl/gl_program.h"
#include "tensorflow/lite/delegates/gpu/gl/gl_shader.h"
#include "tensorflow/lite/delegates/gpu/gl/gl_texture.h"
#include "tensorflow/lite/delegates/gpu/gl_delegate.h"

namespace
{
constexpr int kWorkgroupSize = 8; // Block size for GPU shader.
enum
{
    ATTRIB_VERTEX,
    ATTRIB_TEXTURE_POSITION,
    NUM_ATTRIBUTES
};
// Commonly used to compute the number of blocks to launch in a kernel.
int NumGroups(const int size, const int group_size)
{ // NOLINT
    return (size + group_size - 1) / group_size;
}
float Clamp(float val, float min, float max)
{
    return std::min(std::max(val, min), max);
}

constexpr char kTensorsGpuTag[] = "TENSORS_GPU";
constexpr char kMaskGpuTag[] = "OUTPUT_GPU";

} // namespace

namespace mediapipe
{

#if !defined(MEDIAPIPE_DISABLE_GL_COMPUTE)
using ::tflite::gpu::gl::CopyBuffer;
using ::tflite::gpu::gl::CreateReadWriteRgbaImageTexture;
using ::tflite::gpu::gl::CreateReadWriteShaderStorageBuffer;
using ::tflite::gpu::gl::GlBuffer;
using ::tflite::gpu::gl::GlProgram;
using ::tflite::gpu::gl::GlShader;
#endif //  !MEDIAPIPE_DISABLE_GPU

class DeeplabTensorsToSegmentationCalculator : public CalculatorBase
{
public:
    static ::mediapipe::Status GetContract(CalculatorContract *cc);

    ::mediapipe::Status Open(CalculatorContext *cc) override;
    ::mediapipe::Status Process(CalculatorContext *cc) override;
    ::mediapipe::Status Close(CalculatorContext *cc) override;

private:
    ::mediapipe::Status InitGpu(CalculatorContext *cc);
    ::mediapipe::Status ProcessGpu(CalculatorContext *cc);
    void GlRender();

    int tensor_width_ = 257;
    int tensor_height_ = 257;
    int tensor_channels_ = 3;
    int num_classes_ = 21;

    mediapipe::GlCalculatorHelper gpu_helper_;
    std::unique_ptr<GlProgram> mask_program_;
    std::unique_ptr<GlBuffer> tensor_buffer_;
    GLuint upsample_program_;
};
REGISTER_CALCULATOR(DeeplabTensorsToSegmentationCalculator);

// static
::mediapipe::Status DeeplabTensorsToSegmentationCalculator::GetContract(
    CalculatorContract *cc)
{
    RET_CHECK(!cc->Inputs().GetTags().empty());
    RET_CHECK(!cc->Outputs().GetTags().empty());

    cc->Inputs().Tag(kTensorsGpuTag).Set<std::vector<GlBuffer>>();

    cc->Outputs().Tag(kMaskGpuTag).Set<mediapipe::GpuBuffer>();

    MP_RETURN_IF_ERROR(mediapipe::GlCalculatorHelper::UpdateContract(cc));

    return ::mediapipe::OkStatus();
}

::mediapipe::Status DeeplabTensorsToSegmentationCalculator::Open(
    CalculatorContext *cc)
{
    cc->SetOffset(TimestampDiff(0));

    MP_RETURN_IF_ERROR(gpu_helper_.Open(cc));

    gpu_helper_.RunInGlContext([this, cc]() -> ::mediapipe::Status {
        MP_RETURN_IF_ERROR(InitGpu(cc));
        return ::mediapipe::OkStatus();
    });

    return ::mediapipe::OkStatus();
}

::mediapipe::Status DeeplabTensorsToSegmentationCalculator::Process(
    CalculatorContext *cc)
{

    MP_RETURN_IF_ERROR(
        gpu_helper_.RunInGlContext([this, cc]() -> ::mediapipe::Status {
            MP_RETURN_IF_ERROR(ProcessGpu(cc));
            return ::mediapipe::OkStatus();
        }));

    return ::mediapipe::OkStatus();
}

::mediapipe::Status DeeplabTensorsToSegmentationCalculator::Close(
    CalculatorContext *cc)
{
    gpu_helper_.RunInGlContext([this] {
        if (upsample_program_)
            glDeleteProgram(upsample_program_);
        upsample_program_ = 0;
        mask_program_.reset();
        tensor_buffer_.reset();
    });

    return ::mediapipe::OkStatus();
}

::mediapipe::Status DeeplabTensorsToSegmentationCalculator::ProcessGpu(
    CalculatorContext *cc)
{
    // Get input streams.
    const auto &input_tensors =
        cc->Inputs().Tag(kTensorsGpuTag).Get<std::vector<GlBuffer>>();

    RET_CHECK_EQ(input_tensors.size(), 1);

    // Create initial working mask texture.
    ::tflite::gpu::gl::GlTexture input_texture;
    RET_CHECK_CALL(CreateReadWriteRgbaImageTexture(
        tflite::gpu::DataType::UINT8, // GL_RGBA8
        {tensor_width_, tensor_height_}, &input_texture));

    // Copy input tensor.
    RET_CHECK_CALL(CopyBuffer(input_tensors[0], *tensor_buffer_));

    {
        glBindImageTexture(0, input_texture.id(), 0, GL_FALSE, 0,
                           GL_WRITE_ONLY, GL_RGBA8);
        RET_CHECK_CALL(tensor_buffer_->BindToIndex(2));
        const tflite::gpu::uint3 workgroups = {
            NumGroups(tensor_width_, kWorkgroupSize),
            NumGroups(tensor_height_, kWorkgroupSize), 1};

        mask_program_->Dispatch(workgroups);
    }

    mediapipe::GlTexture output_texture = gpu_helper_.CreateDestinationTexture(
        tensor_width_, tensor_height_,
        mediapipe::GpuBufferFormat::kBGRA32); // actually GL_RGBA8

    {
        gpu_helper_.BindFramebuffer(output_texture); // GL_TEXTURE0
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, input_texture.id());
        GlRender();
        glBindTexture(GL_TEXTURE_2D, 0);
        glFlush();
    }

    // Send out image as GPU packet.
    auto output_image = output_texture.GetFrame<mediapipe::GpuBuffer>();
    cc->Outputs()
        .Tag(kMaskGpuTag)
        .Add(output_image.release(), cc->InputTimestamp());

    return ::mediapipe::OkStatus();
}

void DeeplabTensorsToSegmentationCalculator::GlRender()
{
    static const GLfloat square_vertices[] = {
        1.0f, -1.0f,  // bottom right
        -1.0f, -1.0f, // bottom left
        1.0f, 1.0f,   // top right
        -1.0f, 1.0f, // top left
    };
    static const GLfloat texture_vertices[] = {
        0.0f, 1.0f, // top left
        0.0f, 0.0f, // bottom left
        1.0f, 1.0f, // top right
        1.0f, 0.0f, // bottom right
    };

    // program
    glUseProgram(upsample_program_);

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
}

::mediapipe::Status DeeplabTensorsToSegmentationCalculator::InitGpu(
    CalculatorContext *cc)
{
    MP_RETURN_IF_ERROR(gpu_helper_.RunInGlContext([this]() -> ::mediapipe::Status {
        // Create shader
        const std::string shader_src_template =
            R"( #version 310 es
precision highp float;

layout(local_size_x = $0, local_size_y = $0, local_size_z = 1) in;

layout(std430, binding = 2) readonly buffer B0 {
    float data[257][257][21];
} segmentation;

layout(rgba8, binding = 0) writeonly uniform highp image2D output_texture;

const int person_index = 0;

void main() {
    ivec2 gid = ivec2(gl_GlobalInvocationID.xy);
    int index = 0;
    float max = segmentation.data[gid.x][gid.y][index];
    for (int k = 0; k < 21; k++) {
        if (segmentation.data[gid.x][gid.y][k] > max) {
            max = segmentation.data[gid.x][gid.y][k];
            index = k;
        }
    }
    vec4 out_value;
    if (index != person_index) {
        out_value = vec4(0.0, 1.0, 0.0, 1.0);
    } else {
        out_value = vec4(0.0, 0.0, 0.0, 0.0);
    }
    imageStore(output_texture, gid, out_value);
}
)";
        const std::string shader_src = absl::Substitute(
            shader_src_template, kWorkgroupSize);

        GlShader shader;
        absl::Status status = GlShader::CompileShader(
            GL_COMPUTE_SHADER, shader_src, &shader);
        mask_program_ = absl::make_unique<GlProgram>();
        RET_CHECK_CALL(GlProgram::CreateWithShader(shader,
                                                   mask_program_.get()));

        // Buffer storage for input tensor.
        size_t tensor_length = tensor_width_ * tensor_height_ * num_classes_;
        tensor_buffer_ = absl::make_unique<GlBuffer>();
        RET_CHECK_CALL(CreateReadWriteShaderStorageBuffer<float>(
            tensor_length, tensor_buffer_.get()));

        // Vertex shader attributes.
        const GLint attr_location[NUM_ATTRIBUTES] = {
            ATTRIB_VERTEX,
            ATTRIB_TEXTURE_POSITION,
        };
        const GLchar *attr_name[NUM_ATTRIBUTES] = {
            "position",
            "texture_coordinate",
        };

        // Simple pass-through shader, used for hardware upsampling.
        std::string upsample_shader_base = R"(
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
  uniform sampler2D input_data;

  void main() {
    vec4 pix = texture2D(input_data, sample_coordinate);
    fragColor = pix;
  }
)";

        // Program
        mediapipe::GlhCreateProgram(
            mediapipe::kBasicVertexShader, upsample_shader_base.c_str(),
            NUM_ATTRIBUTES, &attr_name[0], attr_location, &upsample_program_);
        RET_CHECK(upsample_program_) << "Problem initializing the program.";

        // Parameters
        glUseProgram(upsample_program_);
        glUniform1i(glGetUniformLocation(upsample_program_, "input_data"), 1);
        return ::mediapipe::OkStatus();
    }));

    return ::mediapipe::OkStatus();
}

} // namespace mediapipe
