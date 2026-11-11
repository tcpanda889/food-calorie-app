#!/usr/bin/env bash
# 果蔬卡里后端一键启动脚本：自动加载 .env 环境变量
set -e
cd "$(dirname "$0")"

if [ -f .env ]; then
  set -a
  source .env
  set +a
fi

if [ -z "$BAIDU_SECRET_KEY" ]; then
  echo "⚠️  BAIDU_SECRET_KEY 未配置，将以演示模式运行（provider=demo）"
  echo "   补齐 .env 中的 BAIDU_SECRET_KEY 并重启即可切换真实识别"
fi

exec python3 -m uvicorn app:app --host 0.0.0.0 --port "${PORT:-8000}"
