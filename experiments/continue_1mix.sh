#!/bin/sh
# [AI 도움] 직렬 1혼합 10회 체크포인트에서 수업 학습기로 10회 더 학습하고 같은 특징을 평가한다.
set -eu
python_bin=${PYTHON:-/usr/bin/python3}
repo=$(cd "$(dirname "$0")/.." && pwd)
hmm="$repo/bjasr_sc/03gmm_hmm"
run=$(mktemp -d "$hmm/exp/iteration-control.XXXXXX")
echo "$run"
ln -s "$repo/bjasr_sc/01compute_features" "$run/01compute_features"
ln -s "$repo/bjasr_sc/data" "$run/data"
mkdir -p "$run/1mix20/exp/data" "$run/1mix20/exp/model_3state_1mix"
cd "$run/1mix20"
ln -s "$hmm/exp/model_3state_1mix/10.hmm" exp/model_3state_1mix/0.hmm
ln -s "$hmm/exp/data/train" exp/data/train
ln -s "$hmm/sc35.dic" sc35.dic
/usr/bin/time -p "$python_bin" -u "$hmm/03_train_sgmhmm_kr.py" > training.log 2> training-time.log
/usr/bin/time -p "$python_bin" -u "$hmm/06_recognize_kr_sc.py" > recognition.log 2> timing.log
"$python_bin" "$hmm/eval.py" > evaluation.log
cat evaluation.log timing.log training-time.log
