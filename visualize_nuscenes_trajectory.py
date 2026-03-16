import matplotlib.pyplot as plt

from nuscenes.nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.prediction import PredictHelper
from nuscenes.prediction.splits import get_prediction_challenge_split


# ===== 配置 =====
# 你的 v1.0-mini 根目录（包含 v1.0-mini 元数据和 maps）
DATAROOT = "/home/wbk/dongbeidaxue/object_tracking/v1.0-mini"
VERSION = "v1.0-mini"
# v1.0-mini 中大部分场景位于 boston-seaport，如果报错可根据场景 log 中的 map_name 调整
MAP_NAME = "boston-seaport"
SPLIT = "mini_train"  # mini_train 或 mini_val


def main() -> None:
    # 1. 初始化 nuScenes 与 map、prediction helper
    nusc = NuScenes(version=VERSION, dataroot=DATAROOT, verbose=True)
    helper = PredictHelper(nusc)
    nusc_map = NuScenesMap(dataroot=DATAROOT, map_name=MAP_NAME)

    # 2. 选择一个用于 prediction 的 agent-sample 对
    pairs = get_prediction_challenge_split(SPLIT, dataroot=DATAROOT)
    if len(pairs) == 0:
        raise RuntimeError(
            "预测 split 为空，请确认已下载 map expansion / prediction_scenes.json"
        )

    instance_token, sample_token = pairs[0]
    print(f"Using instance={instance_token}, sample={sample_token}")

    # 3. 获取历史 / 未来轨迹（全局坐标）
    past = helper.get_past_for_agent(
        instance_token,
        sample_token,
        seconds=2.0,
        in_agent_frame=False,
    )  # [T_past, 2]
    future = helper.get_future_for_agent(
        instance_token,
        sample_token,
        seconds=6.0,
        in_agent_frame=False,
    )  # [T_fut, 2]

    ann = helper.get_sample_annotation(instance_token, sample_token)
    x_center, y_center = ann["translation"][0], ann["translation"][1]

    # 4. 渲染局部地图（车道中心线等）
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.set_aspect("equal")

    patch_radius = 80.0  # 米
    nusc_map.render_map_patch(
        (
            x_center - patch_radius,
            y_center - patch_radius,
            x_center + patch_radius,
            y_center + patch_radius,
        ),
        layer_names=["lane", "lane_connector", "road_segment"],
        figsize=(8, 8),
        ax=ax,
    )

    # 5. 画历史轨迹（蓝）、未来轨迹（红）、当前点（黑）
    if past.shape[0] > 0:
        ax.plot(past[:, 0], past[:, 1], "b.-", label="Past trajectory")
    if future.shape[0] > 0:
        ax.plot(future[:, 0], future[:, 1], "r.-", label="Future trajectory")

    ax.plot(x_center, y_center, "ko", markersize=8, label="Current position")

    ax.set_title(f"Instance {instance_token[:8]}..., sample {sample_token[:8]}...")
    ax.legend(loc="upper right")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

