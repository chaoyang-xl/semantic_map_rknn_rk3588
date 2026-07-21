# RKNN Model Notes

## MobileSAM

The package targets the official Rockchip `rknn_model_zoo/examples/mobilesam`
conversion, not the earlier minimal conversion script.

Expected models:

```text
mobilesam_encoder_tiny.rknn  # about 37 MB
mobilesam_decoder.rknn       # about 11 MB
```

The encoder uses a `448x448` image and produces a `1x256x28x28` embedding. The
decoder is converted with five inputs:

```text
image_embeddings  1x256x28x28
point_coords       1x2x2
point_labels       1x2
mask_input         1x1x112x112
has_mask_input     1
```

Its outputs are `iou_predictions` and `low_res_masks`. The package selects the
mask with the highest IOU prediction, resizes it to `448x448`, removes the
bottom/right image padding, resizes it to the original RGB resolution, and
then applies the configured threshold and erosion.

Conversion environment used for these files:

- RKNN-Toolkit2 `2.3.2`
- `onnx==1.16.1`
- `numpy==1.26.4`
- `protobuf==4.25.4`
- `setuptools<81`

The generated `check*.onnx` files are converter diagnostics and are not runtime
models.

## YOLO-World

Expected models:

```text
yolo_world_v2s_i8.rknn
clip_text_fp16.rknn
```

The converted detector has a fixed `1x80x512` text input. Therefore a class
file may contain at most 80 non-empty prompts. `config/indoor_classes_80.txt`
is the default project list and intentionally excludes `person`, wall, floor,
ceiling, and unknown.

The CLIP text model only needs to run when prompts change. Cache its result on
the OrangePi and use the NPY on subsequent launches. The cached array is
`1x80x512`; unused prompt slots are zero-filled.

## Runtime Compatibility

OrangePi uses `rknnlite.api.RKNNLite`. The host Toolkit backend is retained for
model/debug checks, but an x86 computer without a connected RK3588 cannot run
these RKNN files. Keep the board's RKNN runtime/driver compatible with Toolkit2
`2.3.2`.
