# semantic_map_rknn

Independent ROS 2 and offline semantic mapping for OrangePi 5 Plus (RK3588).
The package runs YOLO-World and the official Rockchip MobileSAM RKNN models,
projects segmented RGB-D points into `map`, tracks repeated observations, and
exports confirmed object point clouds plus a navigation JSON map.

It does not import `semantic_map_offline`, `my_work_pkg`, or MobileSAM PyTorch
source at runtime.

## Pipeline

```text
RGB -> YOLO-World RKNN -> timestamped detection JSON
                         |
RGB + boxes -> MobileSAM encoder once + decoder per box -> masks
depth + masks + camera intrinsics -> camera-frame object points
map <- camera TF -> map-frame observations
geometry + semantic association -> confirmed object point clouds/JSON
```

The default launch can also consume `/yolo/results_json` from
`opi_yolo_rknn_recorder`; in that mode this package only runs MobileSAM,
projection, and fusion.

## Repository Contents

```text
semantic_map_rknn/       pure algorithms and ROS nodes
launch/                  complete ROS launch
config/                  mapping parameters and 80 indoor prompts
scripts/                 offline runner, checks, visualization, text cache
docs/                    model details and topic/file interfaces
test/                    hardware-independent algorithm tests
```

Large `.rknn`, `.onnx`, `.pt`, point-cloud, and output files are ignored.

## Models

Recommended board layout:

```text
/home/orangepi/models/
  mobile_sam/
    mobilesam_encoder_tiny.rknn
    mobilesam_decoder.rknn
  yolo_world_rknn/
    yolo_world_v2s_i8.rknn
    clip_text_fp16.rknn
    indoor_text_embeddings.npy       # optional but recommended
  tokenizer/clip-vit-base-patch32/   # only needed to regenerate embeddings
```

See [RKNN_MODELS.md](docs/RKNN_MODELS.md) for exact tensor shapes and conversion
versions.

## OrangePi Installation

The board runtime must match models converted with RKNN-Toolkit2 2.3.2. Install
the corresponding RKNN Toolkit Lite2 wheel supplied by Rockchip, then verify:

```bash
python3 -c "from rknnlite.api import RKNNLite; print('RKNNLite OK')"
python3 -c "import cv2, numpy, scipy; print(cv2.__version__, numpy.__version__)"
```

Build the ROS package:

```bash
cd ~/ros2_ws
source /opt/ros/humble/setup.bash
rosdep install --from-paths src --ignore-src -r -y
colcon build --packages-select semantic_map_rknn --symlink-install
source install/setup.bash
```

Check dependencies and model loading before launching ROS:

```bash
cd ~/ros2_ws/src/semantic_map_rknn
python3 scripts/check_orangepi.py \
  --sam-encoder /home/orangepi/models/mobile_sam/mobilesam_encoder_tiny.rknn \
  --sam-decoder /home/orangepi/models/mobile_sam/mobilesam_decoder.rknn \
  --yolo-model /home/orangepi/models/yolo_world_rknn/yolo_world_v2s_i8.rknn \
  --clip-text-model /home/orangepi/models/yolo_world_rknn/clip_text_fp16.rknn \
  --load-models
```

## Recommended: Cache YOLO Text Input

YOLO-World has a fixed 80-prompt text input. Generate it once on the board.
`--tokenizer-path` must point to a locally copied Hugging Face CLIP tokenizer:

```bash
python3 scripts/cache_yolo_text_embeddings.py \
  --yolo-model /home/orangepi/models/yolo_world_rknn/yolo_world_v2s_i8.rknn \
  --clip-text-model /home/orangepi/models/yolo_world_rknn/clip_text_fp16.rknn \
  --classes-path config/indoor_classes_80.txt \
  --tokenizer-path /home/orangepi/models/tokenizer/clip-vit-base-patch32 \
  --output /home/orangepi/models/yolo_world_rknn/indoor_text_embeddings.npy
```

After this, runtime does not need `transformers`, the tokenizer, or the 129 MB
CLIP RKNN model.

## ROS Usage

### Use Existing YOLO JSON

Start `opi_yolo_rknn_recorder` first, then:

```bash
ros2 launch semantic_map_rknn semantic_mapping_rknn.launch.py \
  run_yolo:=false \
  sam_encoder:=/home/orangepi/models/mobile_sam/mobilesam_encoder_tiny.rknn \
  sam_decoder:=/home/orangepi/models/mobile_sam/mobilesam_decoder.rknn \
  camera_frame:=camera_color_optical_frame \
  camera_fx:=365.1741638183594 \
  camera_fy:=365.42144775390625 \
  camera_cx:=318.27630615234375 \
  camera_cy:=243.80377197265625 \
  output_directory:=/home/orangepi/semantic_map_output
```

### Run Package YOLO-World RKNN

```bash
ros2 launch semantic_map_rknn semantic_mapping_rknn.launch.py \
  run_yolo:=true \
  yolo_model:=/home/orangepi/models/yolo_world_rknn/yolo_world_v2s_i8.rknn \
  text_embeddings:=/home/orangepi/models/yolo_world_rknn/indoor_text_embeddings.npy \
  sam_encoder:=/home/orangepi/models/mobile_sam/mobilesam_encoder_tiny.rknn \
  sam_decoder:=/home/orangepi/models/mobile_sam/mobilesam_decoder.rknn \
  output_directory:=/home/orangepi/semantic_map_output
```

`text_embeddings` takes precedence over `clip_text_model`.

### RViz Topics

Add these displays:

- PointCloud2: `/semantic_rknn/points` for per-frame projection.
- PointCloud2: `/semantic_rknn/fused_points` for confirmed object clouds.
- MarkerArray: `/semantic_rknn/object_markers`.
- Image: `/semantic_rknn/sam_debug_image`.

Set RViz Fixed Frame to `map`.

## Offline Dataset Usage On OrangePi

The dataset layout is the same Replica-compatible export used by
`semantic_map_offline`:

```text
dataset/
  cam_params.json
  traj.txt
  results/frame000000.jpg
  results/depth000000.png
  detections/000000.json       # optional
```

If `detections/*.json` exists, YOLO is not run. Process all frames with the
validated conservative parameters (these are already the defaults):

```bash
python3 scripts/evaluate_rknn_projection_tracking.py \
  --data-root /data/semantic_05/dataset \
  --output /data/semantic_05/tracking_rknn \
  --sam-encoder /home/orangepi/models/mobile_sam/mobilesam_encoder_tiny.rknn \
  --sam-decoder /home/orangepi/models/mobile_sam/mobilesam_decoder.rknn \
  --frames 0 \
  --frame-step 2 \
  --confidence 0.50 \
  --min-depth 0.3 \
  --max-depth 5.0 \
  --pixel-stride 2 \
  --min-confirmed-observations 8 \
  --progress-every 10
```

On RK3588 the offline command enables one-frame NPU pipelining by default and
pins the three model contexts to separate cores:

```text
--pipeline-prefetch
--yolo-core 0
--sam-encoder-core 1
--sam-decoder-core 2
```

The next frame is only read and detected ahead of time. SAM, projection, and
tracking still commit frames in source order, so object identities and output
ordering are unchanged. Use `--no-pipeline-prefetch` for serial diagnostics.

OrangePi 5 Plus FP16 measurements on the first 500 Replica frames:

| Mode | Elapsed | Detections | Objects |
| --- | ---: | ---: | ---: |
| Serial | 175.64 s | 777 | 7 |
| NPU pipeline | 110.55 s | 777 | 7 |

The association JSON and every saved object NPZ array were identical.

`--frame-step 2` processes source frames `0, 2, 4, ...` while preserving
each selected frame's original pose and file index. `--frames` still limits the number
of processed frames; use `--frame-step 1` for the full sequence.

Progress lines include per-frame milliseconds for `io`, `detection`,
`sam_encoder`, `sam_decoder`, `projection`, and `fusion`. After a
successful run, the output directory also contains `timing.json`. It reports
total time, percentage, call count, average time per call, and average time per
frame for every stage:

```bash
cat /data/semantic_05/tracking_rknn/timing.json
```

Model loading happens before the measured interval and is intentionally excluded.
Final object cleanup and file serialization are reported as `finalize` and
`save`. With `--pipeline-prefetch`, stage work overlaps in wall-clock time;
`overlapped_seconds` reports that overlap and stage percentages may sum above 100%.

If detection JSON is absent, additionally provide:

```text
--yolo-model .../yolo_world_v2s_i8.rknn
--text-embeddings .../indoor_text_embeddings.npy
--classes-path config/indoor_classes_80.txt
```

Generate a 2D object-colored map after processing:

```bash
python3 scripts/view_objects_2d.py \
  --objects-dir /data/semantic_05/tracking_rknn/objects \
  --min-observations 8
```

## Important Geometry Requirements

- Detection JSON timestamps must come from the RGB image header.
- `camera_frame` must be the optical frame corresponding to the projected RGB-D
  pixels, and TF must provide `map <- camera_frame` at that timestamp.
- The depth image must be registered to RGB. Resizing a 640x400 unregistered
  depth image to a 640x480 RGB mask does not perform geometric registration.
- Camera intrinsics must match the depth image used for projection.
- The default valid depth range is 0.3-5.0 m.

## RK3588 Performance Monitor

The repository includes a dependency-free, full-screen monitor for OrangePi 5
Plus. It reports per-core CPU utilization and frequency, RK3588 NPU core load,
Mali GPU load, thermal zones, RAM, swap, storage, and the busiest processes.
The compact dashboard uses the terminal alternate screen and refreshes in place,
so normal monitoring does not grow the terminal scrollback. OrangePi's eight CPU
cores occupy two rows. Use `--no-clear` only when an appended log is required.

```bash
cd /home/orangepi/ros2_ws/src/semantic_map_rknn_rk3588
sudo ./scripts/monitor_rk3588.sh --interval 1
```

Useful diagnostic modes:

```bash
# One sample for an issue report
sudo ./scripts/monitor_rk3588.sh --once

# Append samples instead of clearing the screen
sudo ./scripts/monitor_rk3588.sh --interval 2 --no-clear | tee performance.log
```

Running with `sudo` is recommended because the Rockchip NPU load is exposed at
`/sys/kernel/debug/rknpu/load`. Without permission, all other readable metrics
remain available. `btop` can be used as a complementary general Linux monitor,
but it does not replace the RK3588-specific NPU section in this script.

## Performance

MobileSAM runs the encoder once per processed image and the decoder once per
detection. Start with `frame_skip:=2` if both YOLO and SAM share the NPU. Keep
`pixel_stride=2` and `voxel_size=0.02` for initial board tests. Increasing
`denoise_interval` reduces CPU spikes from growing object point clouds.

The ROS projector intentionally drops frames when inference is slower than the
camera. It preserves timestamp correctness instead of building an unbounded
queue. Offline dataset processing does not drop frames.

See [INTERFACES.md](docs/INTERFACES.md) for exact topics and JSON fields.
