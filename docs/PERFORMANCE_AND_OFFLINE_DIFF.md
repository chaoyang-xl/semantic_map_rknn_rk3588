# RK3588 性能总结与 semantic_map_offline 迁移差异

> 统计日期：2026-07-23
> 测试数据：Replica 2000 帧，结果目录 `/home/weiyu/Desktop/orangepi`
> 部署平台：OrangePi 5 Plus / RK3588，RKNN Runtime 2.3.2
> 推理模型：YOLO-World v2s FP16（对照组含 INT8）+ MobileSAM RKNN encoder/decoder

## 1. 结论

在检测结果相同的四组可比实验中，2000 帧总耗时从
`1423.628 s` 降到 `549.299 s`：

- 总耗时下降 **61.42%**。
- 吞吐率从 **1.40 FPS** 提升到 **3.64 FPS**，约为原来的 **2.59 倍**。
- 输入检测、成功投影和最终对象数量保持为 `6159 / 6159 / 19`。
- 最终版本与其直接前序版本的关联 JSON 和全部对象 NPZ 数组完全一致。
- 没有跨帧复用 SAM embedding，避免相机运动导致旧特征与当前检测框错位。

当前推荐版本是 `replica_fp16_pipeline3_full` 对应的三级流水线实现。

## 2. 同配置性能对比

这四组均处理 2000 帧，使用相同 FP16 YOLO-World、相同类别和阈值，并产生
6159 个检测、6159 次投影及 19 个确认对象，因此可以直接比较。

| 结果目录 | 主要改动 | 总耗时/s | 吞吐/FPS | Fusion 工作量/s | 相对前版 |
| --- | --- | ---: | ---: | ---: | ---: |
| `replica_fp16_pipeline_full` | 初始 YOLO 预取流水线 | 1423.628 | 1.40 | 902.365 | 基线 |
| `replica_fp16_light_full` | 关联点数上限、关闭周期全图去噪 | 1160.313 | 1.72 | 640.120 | -18.50% |
| `replica_fp16_knn_full` | 有界 kNN 单帧空间聚类 | 811.579 | 2.46 | 266.652 | -30.06% |
| `replica_fp16_pipeline3_full` | YOLO、SAM、fusion 三级流水线 | **549.299** | **3.64** | 317.866 | **-32.32%** |

三级流水线下 Fusion 的累计工作时间比上一版略高，是并行运行造成 CPU/NPU
资源争用后的测量结果。它与其他阶段重叠执行，不能直接相加为墙钟时间。
最终墙钟时间减少了 262.280 秒，才是调度优化的有效指标。

### 分阶段计时

| 结果目录 | IO/s | YOLO/s | SAM encoder/s | SAM decoder/s | 投影/s | Fusion/s | 重叠/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| pipeline_full | 54.697 | 357.155 | 393.306 | 99.014 | 15.997 | 902.365 | 404.424 |
| light_full | 53.340 | 357.367 | 393.107 | 98.885 | 15.414 | 640.120 | 403.133 |
| knn_full | 50.568 | 385.505 | 421.070 | 101.328 | 14.180 | 266.652 | 428.834 |
| pipeline3_full | 75.947 | 321.794 | 367.488 | 131.280 | 29.053 | 317.866 | **695.333** |

## 3. 优化效果来源

### 3.1 轻量点云策略

`light_full` 引入：

- 输出点云继续使用 `voxel_size=0.02`，不牺牲最终 RGB 点云分辨率。
- 关联几何最多使用 `association_max_points=4096` 个代表点。
- `denoise_interval=0`，运行中不再周期性扫描所有历史对象。
- 保存前仍执行一次最终去噪。

相对基线，总耗时下降 18.50%，Fusion 工作量下降 29.06%。

### 3.2 有界空间聚类

全量 profile 显示，旧 Fusion 的主要瓶颈不是点云合并，而是
`_clean_observation()` 中的半径邻居聚类：

- `_clean_observation`：636.918 s
- `largest_spatial_cluster_indices`：510.400 s
- 真正的对象关联 `_assign`：46.068 s
- 点云合并 `_merge_observation`：11.108 s

旧实现会为密集点云展开每个点的全部半径邻居。后半段检测框点数增加后，
Python 邻居集合快速增大。新实现使用最多 32 邻居的 kNN 连通图，使复杂度和
内存规模有界，同时保留完整输出点云。

### 3.3 三级流水线

旧流水线只重叠“下一帧 YOLO”和“当前帧 SAM + Fusion”。新流水线拆成：

1. RGB-D 读取和 YOLO-World 推理。
2. MobileSAM encoder/decoder 和 3D 投影。
3. 严格按源帧顺序执行对象关联及点云融合。

三个 RKNN context 分别配置到 NPU core 0、1、2。每个 RKNN session 仍只由
单一工作线程访问，Fusion 仍按帧顺序提交，因此没有改变输出顺序和对象身份。

## 4. 输出质量一致性

以 `replica_fp16_pipeline_full` 为初始基线：

| 版本 | 关联身份差异 | 最终点数 | 最终对象 |
| --- | ---: | ---: | ---: |
| pipeline_full | 0 | 85628 | 19 |
| light_full | 0 | 85628 | 19 |
| knn_full | 1 个临时候选关联 | 85610 | 19 |
| pipeline3_full | 与 knn_full 完全一致 | 85610 | 19 |

说明：

- `light_full` 的最终 NPZ 与初始基线逐数组相同。
- kNN 版本总计减少 18 点，仅占 0.021%。
- 唯一关联差异发生在最终会被清除的 candidate 上。
- `pipeline3_full` 与 `knn_full` 的 6159 条关联记录完全相同。
- 两者的 19 个对象 NPZ 文件逐数组完全相同。
- 两者的语义对象 JSON 除生成时间外完全相同。

## 5. 其他历史结果

下表包含配置不同的历史测试，只用于观察趋势，不能与上面的同配置链直接计算
算法加速比。

| 结果目录 | 模型/配置 | 检测数 | 对象数 | 总耗时/s | FPS |
| --- | --- | ---: | ---: | ---: | ---: |
| `replica_output_orangepi_int8` | 早期 INT8 | 6658 | 20 | 2266.449 | 0.88 |
| `replica_output_orangepi_fp16` | 早期 FP16 | 6605 | 21 | 1917.938 | 1.04 |
| `replica_fp16_workers4` | FP16 + 4 CPU workers | 6605 | 21 | 1890.209 | 1.06 |
| `replica_fp16_timing` | 早期计时版，更多检测 | 7993 | 24 | 2068.502 | 0.97 |
| `tracking_rknn_pipeline3_full_config_0.3` | 三级流水线，confidence 0.3 | 7993 | 24 | 710.163 | 2.82 |

历史数据说明：

- 在相同 6605 个检测的早期测试中，FP16 比 INT8 总耗时低约 15.38%。
  这不表示 FP16 算子本身必然更快，整体时间还受模型转换质量和 Fusion 影响。
- 4 CPU worker 只比对应 FP16 单 worker 快 1.45%，收益很小，因此已回退。
- confidence 0.3 产生 7993 个检测，比 6159 多约 29.8%，总耗时也相应增加。
- 两个三级流水线结果约为 11.2 detections/s，说明当前耗时与检测框数量近似成正比。

## 6. 与 semantic_map_offline 的代码差异

`semantic_map_rknn` 是面向 RK3588 部署的独立包，不是
`semantic_map_offline` 全部实验功能的复制。迁移原则是保留数据接口与核心
几何算法，替换推理后端，并删除板端不需要的数据制作工具。

### 6.1 保持不变或仅改包名

以下模块的算法内容相同：

- `bbox_projection.py`
- `object_map_io.py`
- `point_cloud_io.py`
- `top_down_projection.py`

`mask_projection.py` 仅把
`semantic_map_offline.bbox_projection` 导入路径改为
`semantic_map_rknn.bbox_projection`。

`offline_projector_node.py` 的主体逻辑保持不变，主要修改包内导入路径和默认
节点名称。JSON、PLY、NPZ 和俯视图输出格式因此可以继续被导航及可视化工具使用。

### 6.2 推理后端替换

```diff
- Ultralytics YOLOWorld + PyTorch + 本地 CLIP checkpoint
+ YoloWorldRknn + RKNNLite/RKNN Toolkit
+ 固定 80 个 prompt slot
+ 可缓存 indoor_text_embeddings.npy，板端运行不依赖 Torch

- MobileSAM Python 源码 + mobile_sam.pt
+ 官方 RKNN model zoo encoder/decoder
+ encoder 输出布局转换
+ decoder 的 low_res_masks resize/crop 回原始图像
```

新增核心模块：

- `rknn_runtime.py`：统一 RKNNLite/Toolkit、NPU core mask、session 锁和资源释放。
- `yolo_world_rknn.py`：letterbox、文本 embedding、RKNN 输出后处理和 NMS。
- `mobilesam_rknn.py`：官方 448 输入 encoder、box prompt decoder 和 mask 后处理。
- `dataset_pipeline.py`：Replica 格式全链路、阶段计时和三级流水线。
- `yolo_world_node.py`、`sam_projector_node.py`：板端 ROS 2 推理节点。

### 6.3 跟踪与点云融合修改

`object_tracker.py` 从约 541 行增加到约 741 行，主要变化：

```diff
- 每次关联重新计算 centroid、AABB 和 cKDTree
+ GeometryIndex 缓存 centroid、bounds、cKDTree
+ 点云更新后自动失效并重建缓存

- 所有历史点参与最近邻关联
+ 输出保留完整点云
+ 关联搜索最多使用 4096 个代表点

- 先拼接全部点，再对完整数组重新体素化
+ 使用打包 voxel code、union1d 和 searchsorted 增量合并

- 对候选 track 直接执行完整几何查询
+ centroid + AABB broad phase 提前排除不可能关联

- 默认每 20 帧对所有 track 运行 DBSCAN
+ denoise_interval=0
+ 只在 finalize 保存前执行完整去噪
```

`spatial_filter.py` 的变化：

```diff
- query_ball_point 为每个点展开全部半径邻居
- Python set/stack 遍历完整 DBSCAN 邻接
+ cKDTree 查询最多 32 个邻居
+ scipy sparse connected_components 计算连通分量
+ 保持最大空间簇和边界点
```

### 6.4 ROS 节点与配置

`semantic_map_offline` 的主要入口包括 CPU/GPU 离线投影、MobileSAM、数据解码、
Cartographer 导出和 YOLO 记录节点。

`semantic_map_rknn` 收敛为三个部署节点：

- `yolo_world_rknn_node`
- `sam_rknn_projector_node`
- `object_fusion_node`

新增统一启动文件 `launch/semantic_mapping_rknn.launch.py` 和参数文件
`config/semantic_mapping.yaml`。ROS 融合节点新增
`association_max_points`，默认关闭周期性全图去噪，并把输出 source 标识改为
`semantic_map_rknn_ros`。

### 6.5 未迁移的离线工具

以下功能仍由 `semantic_map_offline` 负责，没有放入 RKNN 部署包：

- Cartographer pbstream、最终优化轨迹和 Replica 格式数据集导出。
- rosbag 解码、里程计/相机 TF 辅助节点。
- 单帧投影反投影 IoU 评估。
- PyTorch 非 SAM 与 MobileSAM 对照评估。
- occupancy map 叠加和 SLAM 地图实验脚本。
- Meeting Room、semantic_01、semantic_05 等数据准备 launch。

因此推荐职责划分是：

```text
semantic_map_offline
  数据制作、Cartographer 最终轨迹导出、PC/GPU 算法验证

semantic_map_rknn
  OrangePi/RK3588 推理、ROS 在线语义投影、板端离线全量处理
```

## 7. 推荐全量参数

```bash
python3 scripts/evaluate_rknn_projection_tracking.py \
  --data-root /home/orangepi/my_data \
  --output /home/orangepi/my_data/semantic_05/tracking_rknn_pipeline3_full \
  --sam-encoder /home/orangepi/models/mobile_sam/mobilesam_encoder_tiny.rknn \
  --sam-decoder /home/orangepi/models/mobile_sam/mobilesam_decoder.rknn \
  --yolo-model /home/orangepi/models/yolo_world_rknn/yolo_world_v2s_fp16.rknn \
  --text-embeddings /home/orangepi/models/yolo_world_rknn/indoor_text_embeddings.npy \
  --classes-path config/indoor_classes_80.txt \
  --rknn-backend lite \
  --pipeline-prefetch \
  --yolo-core 0 \
  --sam-encoder-core 1 \
  --sam-decoder-core 2 \
  --frames 0 \
  --frame-step 1 \
  --confidence 0.50 \
  --min-depth 0.3 \
  --max-depth 5.0 \
  --pixel-stride 2 \
  --voxel-size 0.02 \
  --association-max-points 4096 \
  --overlap-radius 0.04 \
  --max-centroid-distance-m 0.75 \
  --min-geometric-overlap 0.08 \
  --association-threshold 0.50 \
  --geometry-weight 0.70 \
  --semantic-weight 0.30 \
  --observation-cluster-eps 0.10 \
  --observation-cluster-min-points 10 \
  --max-extent-growth 1.50 \
  --denoise-interval 0 \
  --map-merge-interval 0 \
  --min-confirmed-observations 8 \
  --candidate-max-missed-frames 30 \
  --progress-every 100
```

## 8. 后续优化边界

- 不建议直接跨帧复用 SAM encoder embedding；除非同时加入图像运动判断或特征
  warp，否则机器人运动时会降低 mask 与当前检测框的几何一致性。
- 若优先追求速度，`frame_step` 和检测置信度仍是最有效的负载控制参数。
- confidence 从 0.50 降到 0.30 会明显增加 decoder、投影和 Fusion 次数。
- 在线 ROS 流程会在推理赶不上相机时主动丢帧；本报告的离线流程不会丢帧。
- 性能回归必须同时记录模型、类别文件、confidence、frame_step 和检测数量，
  否则不同目录的总耗时不能直接比较。
