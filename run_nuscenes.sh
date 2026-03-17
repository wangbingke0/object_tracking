#!/bin/bash
#
# nuScenes IMM-UKF-JPDA 跟踪验证 一键运行脚本
#
# 用法:
#   ./run_nuscenes.sh                         # 运行全部场景（快速，无可视化）
#   ./run_nuscenes.sh scene-0061              # 运行指定场景
#   ./run_nuscenes.sh all --no-eval           # 只跑跟踪，不评估
#   ./run_nuscenes.sh scene-0061 --viz        # 带 RViz 可视化（2Hz 慢速回放）
#   ./run_nuscenes.sh scene-0061 noPre        # 如果已有预处理文件则跳过预处理
#   ./run_nuscenes.sh scene-0061 --10hz       # 插值到 10Hz（dt≈0.1s）
#   ./run_nuscenes.sh scene-0061 --20hz --viz # 插值到 20Hz + 可视化
#   ./run_nuscenes.sh scene-0103 --20hz --viz --noPre # 插值到 20Hz + 可视化 + 跳过预处理
#   ./run_nuscenes.sh scene-0061 --interp=10  # 自定义插值帧率
#   ./run_nuscenes.sh scene-0061 --pred-horizon=5.0 --pred-step=0.1
#

set -e

# ======================== 路径配置 ========================
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
NUSCENES_DATAROOT="$SCRIPT_DIR/v1.0-mini"
PREPROCESS_DIR="$SCRIPT_DIR/nuscenes_preprocessed"
CATKIN_WS="$HOME/dongbeidaxue/nuscenes_ws"

# ======================== 参数解析 ========================
SCENE="${1:-all}"
SKIP_EVAL=false
WITH_VIZ=false
SKIP_PREPROCESS=false
INTERP_HZ=0
PRED_HORIZON=5.0
PRED_STEP=0.1

for arg in "$@"; do
    case $arg in
        --noPre)      SKIP_PREPROCESS=true ;;
        --no-eval)  SKIP_EVAL=true ;;
        --viz)      WITH_VIZ=true ;;
        --interp=*) INTERP_HZ="${arg#*=}" ;;
        --10hz)     INTERP_HZ=10 ;;
        --20hz)     INTERP_HZ=20 ;;
        --pred-horizon=*) PRED_HORIZON="${arg#*=}" ;;
        --pred-step=*)    PRED_STEP="${arg#*=}" ;;
    esac
done

# 过滤掉选项参数，保留场景名
if [[ "$SCENE" == --* || "$SCENE" == "noPre" ]]; then
    SCENE="all"
fi

# ======================== 颜色输出 ========================
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC}  $1"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }
header(){ echo -e "\n${CYAN}========== $1 ==========${NC}\n"; }

# ======================== Step 0: 环境检查 ========================
header "环境检查"

if [ ! -d "$NUSCENES_DATAROOT" ]; then
    error "未找到 nuScenes 数据集: $NUSCENES_DATAROOT"
fi
info "nuScenes 数据集: $NUSCENES_DATAROOT"

source /opt/ros/noetic/setup.bash 2>/dev/null || error "未找到 ROS Noetic"
info "ROS Noetic: OK"

if [ -f "$CATKIN_WS/devel/setup.bash" ]; then
    source "$CATKIN_WS/devel/setup.bash"
    info "catkin workspace: $CATKIN_WS"
else
    warn "未找到编译好的 workspace，将尝试编译..."
fi

# ======================== Step 1: 编译 ========================
header "编译 nuscenes_tracking"

if [ ! -d "$CATKIN_WS/src" ]; then
    mkdir -p "$CATKIN_WS/src"
    ln -sf "$SCRIPT_DIR" "$CATKIN_WS/src/object_tracking"
    info "创建 catkin workspace 并链接项目"
fi

cd "$CATKIN_WS"
catkin_make --only-pkg-with-deps object_tracking 2>&1 | tail -5
source "$CATKIN_WS/devel/setup.bash"

if ! rospack find object_tracking > /dev/null 2>&1; then
    error "编译失败，请检查错误信息"
fi
info "编译成功"

# ======================== Step 2: 预处理 nuScenes 数据 ========================
header "预处理 nuScenes 数据"

if [ "$SCENE" = "all" ]; then
    SCENE_ARG=""
else
    SCENE_ARG="--scene $SCENE"
fi

mkdir -p "$PREPROCESS_DIR"

INTERP_ARG=""
if [ "$INTERP_HZ" -gt 0 ] 2>/dev/null; then
    INTERP_ARG="--interp_hz $INTERP_HZ"
    info "插值帧率: ${INTERP_HZ} Hz"
fi

PREPROCESS_NEEDED=true
if [ "$SKIP_PREPROCESS" = true ]; then
    if [ "$SCENE" = "all" ]; then
        if ls "$PREPROCESS_DIR"/scene-*.txt >/dev/null 2>&1; then
            PREPROCESS_NEEDED=false
            info "检测到已有 scene-*.txt，按 noPre 跳过预处理"
        else
            warn "未找到现有预处理文件，仍将执行预处理"
        fi
    else
        TARGET_PRE_FILE="$PREPROCESS_DIR/${SCENE}.txt"
        if [ -f "$TARGET_PRE_FILE" ]; then
            PREPROCESS_NEEDED=false
            info "检测到已有预处理文件: $TARGET_PRE_FILE，按 noPre 跳过预处理"
        else
            warn "未找到 $TARGET_PRE_FILE，仍将执行预处理"
        fi
    fi
fi

if [ "$PREPROCESS_NEEDED" = true ]; then
    python3 "$SCRIPT_DIR/scripts/preprocess_nuscenes.py" \
        --dataroot "$NUSCENES_DATAROOT" \
        --output_dir "$PREPROCESS_DIR" \
        $SCENE_ARG $INTERP_ARG
else
    info "跳过预处理"
fi

info "预处理完成: $PREPROCESS_DIR"

# ======================== Step 3: 运行跟踪 ========================
header "运行 IMM-UKF-JPDA 跟踪"

# 确定要处理的场景文件列表
if [ "$SCENE" = "all" ]; then
    SCENE_FILES=$(ls "$PREPROCESS_DIR"/scene-*.txt 2>/dev/null)
else
    SCENE_FILES="$PREPROCESS_DIR/${SCENE}.txt"
fi

if [ -z "$SCENE_FILES" ]; then
    error "未找到预处理文件"
fi

# ---- 可视化模式: 使用 roslaunch（一次只跑一个场景） ----
if [ "$WITH_VIZ" = true ]; then
    # 可视化模式只处理第一个场景
    SCENE_FILE=$(echo "$SCENE_FILES" | head -1)
    SCENE_NAME=$(basename "$SCENE_FILE" .txt)
    RESULT_FILE="$PREPROCESS_DIR/tracking_results_${SCENE_NAME}.txt"

    info "可视化模式: $SCENE_NAME (2 Hz 回放)"
    info "RViz 配置: 俯视图，已添加 Marker 显示"
    info ""
    info "在 RViz 中可以看到:"
    info "  - 蓝色半透明框 = nuScenes 检测框 (GT)"
    info "  - 绿色框       = 跟踪器输出的框"
    info "  - 彩色点       = 跟踪目标 (黄:初始化 / 绿:确认 / 蓝:静态)"
    info "  - 绿色箭头     = 速度方向"
    info "  - 红色箭头     = 自车位姿"
    echo ""

    # Set playback rate to match the effective frame rate for real-time playback
    if [ "$INTERP_HZ" -gt 0 ] 2>/dev/null; then
        VIZ_RATE="$INTERP_HZ"
    else
        VIZ_RATE=2
    fi
    info "播放速率: ${VIZ_RATE} Hz (实时速度)"
    info "预测参数: horizon=${PRED_HORIZON}s, step=${PRED_STEP}s"

    roslaunch object_tracking nuscenes_test.launch \
        data_file:="$SCENE_FILE" \
        result_file:="$RESULT_FILE" \
        playback_rate:="$VIZ_RATE" \
        nuscenes_dataroot:="$NUSCENES_DATAROOT" \
        prediction_horizon_s:="$PRED_HORIZON" \
        prediction_step_s:="$PRED_STEP" \
        viz:=true

    # 评估
    if [ "$SKIP_EVAL" = false ] && [ -f "$RESULT_FILE" ]; then
        header "评估结果"
        python3 "$SCRIPT_DIR/scripts/evaluate_tracking.py" \
            --result_file "$RESULT_FILE" \
            --gt_file "$SCENE_FILE" \
            --match_threshold 2.0 \
            --min_track_manage 5
    fi

    exit 0
fi

# ---- 批处理模式: 快速跑全部场景 ----

# 启动 roscore（后台）
roscore &
ROSCORE_PID=$!
sleep 2

cleanup() {
    kill $ROSCORE_PID 2>/dev/null
    wait $ROSCORE_PID 2>/dev/null
}
trap cleanup EXIT

RESULT_SUMMARY="$PREPROCESS_DIR/evaluation_summary.txt"
echo "# nuScenes IMM-UKF-JPDA Tracking Evaluation Summary" > "$RESULT_SUMMARY"
echo "# $(date)" >> "$RESULT_SUMMARY"
echo "" >> "$RESULT_SUMMARY"

for SCENE_FILE in $SCENE_FILES; do
    SCENE_NAME=$(basename "$SCENE_FILE" .txt)
    RESULT_FILE="$PREPROCESS_DIR/tracking_results_${SCENE_NAME}.txt"

    info "跟踪场景: $SCENE_NAME"

    rosrun object_tracking nuscenes_tracking \
        _data_file:="$SCENE_FILE" \
        _result_file:="$RESULT_FILE" \
        _playback_rate:=1000.0 \
        _prediction_horizon_s:="$PRED_HORIZON" \
        _prediction_step_s:="$PRED_STEP" \
        2>&1 | grep -E "Frame.*/|=== "

    if [ ! -f "$RESULT_FILE" ]; then
        warn "场景 $SCENE_NAME 跟踪结果未生成，跳过"
        continue
    fi

    info "结果保存: $RESULT_FILE"

    # 评估
    if [ "$SKIP_EVAL" = false ]; then
        echo "--- $SCENE_NAME ---" >> "$RESULT_SUMMARY"
        python3 "$SCRIPT_DIR/scripts/evaluate_tracking.py" \
            --result_file "$RESULT_FILE" \
            --gt_file "$SCENE_FILE" \
            --match_threshold 2.0 \
            --min_track_manage 5 \
            2>&1 | tee -a "$RESULT_SUMMARY"
        echo "" >> "$RESULT_SUMMARY"
    fi

    echo ""
done

# ======================== 汇总 ========================
header "完成"

info "预处理数据:   $PREPROCESS_DIR/scene-*.txt"
info "跟踪结果:     $PREPROCESS_DIR/tracking_results_*.txt"
if [ "$SKIP_EVAL" = false ]; then
    info "评估汇总:     $RESULT_SUMMARY"
fi

echo ""
info "可视化命令 (单场景慢速回放 + RViz):"
echo "  ./run_nuscenes.sh scene-0061 --viz"
echo ""
