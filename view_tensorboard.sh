LOG_DIR=${1:-"runs"}

echo "启动 TensorBoard..."
echo "日志目录: $LOG_DIR"
echo "访问地址: http://localhost:6006"
echo ""
echo "按 Ctrl+C 停止 TensorBoard"
echo ""

tensorboard --logdir="$LOG_DIR" --port=6006
