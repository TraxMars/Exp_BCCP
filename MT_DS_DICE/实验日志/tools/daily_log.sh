#!/usr/bin/env bash
# 每日实验日志：crontab 每天 23:55（Asia/Shanghai）调用；也可手动运行：daily_log.sh [YYYY-MM-DD]
# 当天没有新提交、代码改动或结果更新时，只在 cron.log 里记一行，不调用 claude。
# 输出（包括 claude 的一行摘要）由 crontab 追加到 cron.log；手动运行时打印到终端。
set -uo pipefail

# cron 的环境变量很少，这里全部显式设置
export HOME=/aita_data/home/aita_duanxuelong
export PATH=/usr/local/bin:/usr/bin:/bin:/usr/local/sbin:/usr/sbin
export TZ=Asia/Shanghai
export LANG=C.UTF-8 LC_ALL=C.UTF-8
# 与 ~/.claude/settings.json 中的代理一致（claude 访问 API 需要它）
export HTTP_PROXY=http://127.0.0.1:17890 HTTPS_PROXY=http://127.0.0.1:17890

NEW=/aita_data/home/aita_duanxuelong/Exp_BCCP/MT_DS_DICE
OLD=/aita_data/home/aita_duanxuelong/Baseline/MT_DS_DICE
BASELINE_RESULTS=$OLD/Results_baseline_pl
LOGDIR=$NEW/实验日志
TOOLS=$LOGDIR/tools
PY=/usr/bin/python3

D=${1:-$(date +%F)}
date -d "$D" +%F >/dev/null 2>&1 || { echo "用法: $0 [YYYY-MM-DD]"; exit 2; }
NEXT=$(date -d "$D +1 day" +%F)
OUT=$LOGDIR/$D.md

say() { echo "[$(date '+%F %T')] [$D] $*"; }

exec 9>"$TOOLS/.lock"
flock -n 9 || { say "上一次运行尚未结束，本次跳过"; exit 0; }

# ---------- 1. 判断当天是否有新实验 ----------
# 统计起点：当天 00:00；若前一篇日志写于更早（如前一天 23:55），从它的写入时间算起，
# 以免漏掉 23:55–24:00 之间结束的 run。
PREV=$(ls "$LOGDIR"/????-??-??.md 2>/dev/null | awk -v out="$OUT" '$0 < out' | sort | tail -1)
SINCE="$D 00:00:00"
if [ -n "$PREV" ]; then
  p=$(date -r "$PREV" '+%F %T')
  [[ "$p" < "$SINCE" ]] && SINCE=$p
fi
n_commit=0; n_code=0; n_result=0
for repo in "$OLD" "$NEW"; do
  c=$(git -C "$repo" log --all --since="$SINCE" --until="$NEXT 00:00:00" --format=%h 2>/dev/null | wc -l)
  code=$(find "$repo" \( -path "$repo/.git" -o -path "$repo/Results*" -o -path "$LOGDIR" -o -name __pycache__ \) -prune \
           -o -type f -newermt "$SINCE" ! -newermt "$NEXT 00:00:00" -print 2>/dev/null | wc -l)
  res=0
  for r in "$repo"/Results*/; do
    [ -d "$r" ] || continue
    res=$((res + $(find "$r" -type f -newermt "$SINCE" ! -newermt "$NEXT 00:00:00" 2>/dev/null | wc -l)))
  done
  n_commit=$((n_commit + c)); n_code=$((n_code + code)); n_result=$((n_result + res))
done
if [ "$n_commit" -eq 0 ] && [ "$n_code" -eq 0 ] && [ "$n_result" -eq 0 ]; then
  say "无新实验（无新提交、无代码改动、无结果文件更新），未生成日志"
  exit 0
fi
say "开始（自 $SINCE 起）：新提交 $n_commit 个，代码文件改动 $n_code 个，结果文件更新 $n_result 个"

# ---------- 2. 找到 claude（VSCode 插件自带，路径随版本变化，取最新版本） ----------
CLAUDE=$(ls -d "$HOME"/.vscode-server/extensions/anthropic.claude-code-*/resources/native-binary/claude 2>/dev/null | sort -V | tail -1)
if [ -z "$CLAUDE" ] || [ ! -x "$CLAUDE" ]; then
  say "错误：找不到 claude 可执行文件"; exit 1
fi
if ! curl -s -o /dev/null -m 10 https://api.anthropic.com; then
  say "警告：通过代理 $HTTPS_PROXY 访问 api.anthropic.com 失败，仍尝试运行"
fi

# ---------- 3. 预先收集只读材料 ----------
gitinfo() {
  local repo=$1
  echo "### $repo"
  echo '```text'
  echo "\$ git log --all --since='$SINCE' --until='$NEXT 00:00' --stat"
  git -C "$repo" log --all --since="$SINCE" --until="$NEXT 00:00:00" --date=iso --stat 2>&1 | head -300
  echo "\$ git log -5 --oneline（最近 5 个提交，供参考）"
  git -C "$repo" log -5 --date=iso --format='%h %ad %s' 2>&1
  echo "\$ git status --short"
  git -C "$repo" status --short 2>&1 | head -100
  echo "\$ git diff --stat（未提交改动）"
  git -C "$repo" diff --stat 2>&1 | head -100
  echo '```'
}

CONTEXT=$(
  cat "$TOOLS/prompt.md"
  echo
  echo "# 本次运行参数"
  echo "- 日志日期：$D"
  echo "- 日志文件（只写这一个文件）：$OUT"
  echo "- 前一篇日志：${PREV:-（无，这是第一篇）}"
  echo "- tools 目录：$TOOLS"
  echo "- 基线 Results 目录：$BASELINE_RESULTS"
  echo "- 变化统计起点：$SINCE（当天 00:00，或前一篇日志的写入时间，取较早者）"
  echo "- 当天变化计数：新提交 $n_commit 个，代码文件改动 $n_code 个，结果文件更新 $n_result 个"
  echo "- 材料收集时间：$(date '+%F %T')"
  echo
  echo "# git 信息"
  gitinfo "$OLD"
  gitinfo "$NEW"
  echo
  echo "# 训练进程与 GPU"
  echo '```text'
  echo '$ ps -u '"$(id -un)"' -o pid,ppid,lstart,etime,args | grep -E "train.py|launch.py"'
  ps -u "$(id -un)" -o pid,ppid,lstart,etime,args | grep -E 'train.py|launch.py' | grep -v grep
  echo '$ nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv'
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv 2>&1
  echo '```'
  echo
  "$PY" -I "$TOOLS/summarize_runs.py" --date "$D" --baseline "$BASELINE_RESULTS" "$OLD" "$NEW" 2>&1
)

# ---------- 4. 调用 claude（无头模式，最小权限） ----------
# dontAsk：未在 --allowedTools 中放行的操作一律拒绝；--tools 只提供这 6 个内置工具；
# 只能读两个实验目录、只能写 实验日志/*.md（不含 tools/ 子目录；Edit 规则同时约束 Write）。
cd "$NEW" || exit 1
LOGPAT="/$LOGDIR/*.md"   # 权限规则中 //开头 表示绝对路径
printf '%s\n' "$CONTEXT" | timeout 40m "$CLAUDE" -p \
  --output-format text \
  --no-session-persistence \
  --setting-sources user \
  --strict-mcp-config \
  --disable-slash-commands \
  --permission-mode dontAsk \
  --permission-prompts none \
  --add-dir "$OLD" \
  --tools "Read,Glob,Grep,Bash,Write,Edit" \
  --allowedTools \
    "Read(/$NEW/**)" "Read(/$OLD/**)" "Glob(/$NEW/**)" "Glob(/$OLD/**)" "Grep(/$NEW/**)" "Grep(/$OLD/**)" \
    "Edit($LOGPAT)" \
    "Bash(git log:*)" "Bash(git show:*)" "Bash(git diff:*)" "Bash(git status:*)" \
    "Bash(ls:*)" "Bash(cat:*)" "Bash(head:*)" "Bash(tail:*)" "Bash(wc:*)" "Bash(grep:*)" "Bash(stat:*)" \
    "Bash(date:*)" "Bash(ps:*)" "Bash(nvidia-smi)" "Bash(nvidia-smi --query-gpu=*)" \
    "Bash(nvidia-smi --query-compute-apps=*)" "Bash($PY $TOOLS/summarize_runs.py:*)"
rc=$?

if [ "$rc" -ne 0 ]; then
  say "错误：claude 退出码 $rc（124 表示超时）"
elif [ -s "$OUT" ] && [ "$OUT" -nt "$TOOLS/.lock" ]; then
  say "完成：$OUT"
else
  say "警告：claude 正常退出，但 $OUT 本次没有写入"
fi
exit "$rc"
