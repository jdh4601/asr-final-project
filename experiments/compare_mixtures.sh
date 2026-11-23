#!/bin/sh
# [AI 도움] 수업 인식기와 평가기를 같은 특징에 적용하고 모델별 결과와 시간을 보관한다.
set -eu
python_bin=${PYTHON:-python3}
repo=$(cd "$(dirname "$0")/.." && pwd)
hmm_dir="$repo/bjasr_sc/03gmm_hmm"
model_root=$(cd "${MODEL_ROOT:-$hmm_dir/exp}" && pwd)
run_dir=$(mktemp -d "$hmm_dir/exp/mixture-comparison.XXXXXX")
printf 'Results: %s\n' "$run_dir"

# 원본 인식기가 읽는 상대 경로에 비교할 모델만 연결한다.
ln -s "$repo/bjasr_sc/01compute_features" "$run_dir/01compute_features"
ln -s "$repo/bjasr_sc/data" "$run_dir/data"
for mixtures in 1 2; do
    model="$model_root/model_3state_${mixtures}mix/10.hmm"
    test -s "$model"
    work="$run_dir/${mixtures}mix"
    mkdir -p "$work/exp/data" "$work/exp/model_3state_1mix"
    ln -s "$model" "$work/exp/model_3state_1mix/10.hmm"
    ln -s "$hmm_dir/exp/data/train" "$work/exp/data/train"
    ln -s "$hmm_dir/sc35.dic" "$work/sc35.dic"
    (
        cd "$work"
        printf "%smix recognition: %s/recognition.log\n" "$mixtures" "$work"
        /usr/bin/time -p "$python_bin" -u "$hmm_dir/06_recognize_kr_sc.py" > recognition.log 2> timing.log
        "$python_bin" "$hmm_dir/eval.py" > evaluation.log
        cat evaluation.log timing.log
    )
done
