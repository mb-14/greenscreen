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

#import "ViewController.h"

#import "mediapipe/objc/MPPGraph.h"
#import "mediapipe/objc/MPPCameraInputSource.h"
#import "mediapipe/objc/MPPLayerRenderer.h"

static NSString* const kGraphName = @"mobile_gpu";

static const char* kInputStream = "input_video";
static const char* kOutputStream = "output_video";
static const char* kVideoQueueLabel = "com.google.mediapipe.example.videoQueue";

@interface ViewController () <MPPGraphDelegate, MPPInputSourceDelegate>

// The MediaPipe graph currently in use. Initialized in viewDidLoad, started in viewWillAppear: and
// sent video frames on _videoQueue.
@property(nonatomic) MPPGraph* mediapipeGraph;

@end

@implementation ViewController {
  /// Handles camera access via AVCaptureSession library.
  MPPCameraInputSource* _cameraSource;

  /// Inform the user when camera is unavailable.
  IBOutlet UILabel* _noCameraLabel;
  /// Display the camera preview frames.
  IBOutlet UIView* _liveView;
  /// Render frames in a layer.
  MPPLayerRenderer* _renderer;

  /// Process camera frames on this queue.
  dispatch_queue_t _videoQueue;
}

#pragma mark - Cleanup methods

- (void)dealloc {
  self.mediapipeGraph.delegate = nil;
  [self.mediapipeGraph cancel];
  // Ignore errors since we're cleaning up.
  [self.mediapipeGraph closeAllInputStreamsWithError:nil];
  [self.mediapipeGraph waitUntilDoneWithError:nil];
}

#pragma mark - MediaPipe graph methods

+ (MPPGraph*)loadGraphFromResource:(NSString*)resource {
  // Load the graph config resource.
  NSError* configLoadError = nil;
  NSBundle* bundle = [NSBundle bundleForClass:[self class]];
  if (!resource || resource.length == 0) {
    return nil;
  }
  NSURL* graphURL = [bundle URLForResource:resource withExtension:@"binarypb"];
  NSData* data = [NSData dataWithContentsOfURL:graphURL options:0 error:&configLoadError];
  if (!data) {
    NSLog(@"Failed to load MediaPipe graph config: %@", configLoadError);
    return nil;
  }

  // Parse the graph config resource into mediapipe::CalculatorGraphConfig proto object.
  mediapipe::CalculatorGraphConfig config;
  config.ParseFromArray(data.bytes, data.length);

  // Create MediaPipe graph with mediapipe::CalculatorGraphConfig proto object.
  MPPGraph* newGraph = [[MPPGraph alloc] initWithGraphConfig:config];
  [newGraph addFrameOutputStream:kOutputStream outputPacketType:MPPPacketTypePixelBuffer];
  return newGraph;
}

#pragma mark - UIViewController methods

- (void)viewDidLoad {
  [super viewDidLoad];

  _renderer = [[MPPLayerRenderer alloc] init];
  _renderer.layer.frame = _liveView.layer.bounds;
  [_liveView.layer addSublayer:_renderer.layer];
  _renderer.frameScaleMode = MPPFrameScaleModeFillAndCrop;

  dispatch_queue_attr_t qosAttribute = dispatch_queue_attr_make_with_qos_class(
      DISPATCH_QUEUE_SERIAL, QOS_CLASS_USER_INTERACTIVE, /*relative_priority=*/0);
  _videoQueue = dispatch_queue_create(kVideoQueueLabel, qosAttribute);

  _cameraSource = [[MPPCameraInputSource alloc] init];
  [_cameraSource setDelegate:self queue:_videoQueue];
  _cameraSource.sessionPreset = AVCaptureSessionPresetHigh;
  _cameraSource.cameraPosition = AVCaptureDevicePositionBack;
  // The frame's native format is rotated with respect to the portrait orientation.
  _cameraSource.orientation = AVCaptureVideoOrientationPortrait;

  self.mediapipeGraph = [[self class] loadGraphFromResource:kGraphName];
  self.mediapipeGraph.delegate = self;
  // Set maxFramesInFlight to a small value to avoid memory contention for real-time processing.
  self.mediapipeGraph.maxFramesInFlight = 2;
}

// In this application, there is only one ViewController which has no navigation to other view
// controllers, and there is only one View with live display showing the result of running the
// MediaPipe graph on the live video feed. If more view controllers are needed later, the graph
// setup/teardown and camera start/stop logic should be updated appropriately in response to the
// appearance/disappearance of this ViewController, as viewWillAppear: can be invoked multiple times
// depending on the application navigation flow in that case.
- (void)viewWillAppear:(BOOL)animated {
  [super viewWillAppear:animated];

  [_cameraSource requestCameraAccessWithCompletionHandler:^void(BOOL granted) {
    if (granted) {
      [self startGraphAndCamera];
      dispatch_async(dispatch_get_main_queue(), ^{
        [_noCameraLabel setHidden:YES];
      });
    }
  }];
}

- (void)startGraphAndCamera {
  // Start running self.mediapipeGraph.
  NSError* error;
  if (![self.mediapipeGraph startWithError:&error]) {
    NSLog(@"Failed to start graph: %@", error);
  }

  // Start fetching frames from the camera.
  dispatch_async(_videoQueue, ^{
    [_cameraSource start];
  });
}

#pragma mark - MPPGraphDelegate methods

// Receives CVPixelBufferRef from the MediaPipe graph. Invoked on a MediaPipe worker thread.
- (void)mediapipeGraph:(MPPGraph*)graph
    didOutputPixelBuffer:(CVPixelBufferRef)pixelBuffer
              fromStream:(const std::string&)streamName {
  if (streamName == kOutputStream) {
    // Display the captured image on the screen.
    CVPixelBufferRetain(pixelBuffer);
    dispatch_async(dispatch_get_main_queue(), ^{
      [_renderer renderPixelBuffer:pixelBuffer];
      CVPixelBufferRelease(pixelBuffer);
    });
  }
}

#pragma mark - MPPInputSourceDelegate methods

// Must be invoked on _videoQueue.
- (void)processVideoFrame:(CVPixelBufferRef)imageBuffer
                timestamp:(CMTime)timestamp
               fromSource:(MPPInputSource*)source {
  if (source != _cameraSource) {
    NSLog(@"Unknown source: %@", source);
    return;
  }
  [self.mediapipeGraph sendPixelBuffer:imageBuffer
                            intoStream:kInputStream
                            packetType:MPPPacketTypePixelBuffer];
}

@end
