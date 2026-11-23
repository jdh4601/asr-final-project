"""Train the course 1- and 2-mixture HMMs with parallel Baum-Welch iterations."""
# [AI 도움] 제공 HMM의 발화별 기대 통계 계산을 프로세스로 분산하고 반복마다 한 번 갱신한다.
import argparse
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from pathlib import Path

import numpy as np

from hmmfunc import MonoPhoneHMM


def read_training_data(hmm):
    labels = {}
    for line in Path('exp/data/train/text_int').read_text().splitlines():
        utt, *phones = line.split()
        if utt in labels:
            raise ValueError(f'Duplicate label: {utt}')
        labels[utt] = np.asarray(phones, dtype=np.int64)

    utterances = {}
    listing = Path('../01compute_features/mfcc/train/feats.scp')
    for line in listing.read_text().splitlines():
        utt, filename, frames, dims = line.split()
        if utt in utterances or int(dims) != hmm.num_dims:
            raise ValueError(f'Invalid feature entry: {line}')
        utterances[utt] = (filename, int(frames), labels[utt])
    if not utterances or utterances.keys() != labels.keys():
        raise ValueError('Training features and labels must cover the same utterances')
    return list(utterances.values())


def accumulate_shard(hmm, utterances):
    # These four operations are the E-step from MonoPhoneHMM.train.
    hmm.reset_accumulators()
    total_likelihood = 0.0
    for filename, frames, phones in utterances:
        features = np.fromfile(filename, dtype=np.float32).reshape(frames, hmm.num_dims)
        hmm.calc_out_prob(features, phones)
        hmm.calc_alpha(phones)
        hmm.calc_beta(phones)
        hmm.update_accumulators(features, phones)
        total_likelihood += hmm.loglikelihood
    return hmm.pdf_accumulators, hmm.trans_accumulators, total_likelihood


def add_log_counts(left, right, zero):
    # The course code uses a finite sentinel for log(0). Keep two empty counts empty.
    return np.where((left == zero) & (right == zero), zero, np.logaddexp(left, right))


def merge_shard(hmm, local_pdfs, local_trans):
    for p in range(hmm.num_phones):
        for s in range(hmm.num_states):
            trans = hmm.trans_accumulators[p][s]
            for key in ('num', 'den'):
                trans[key] = add_log_counts(trans[key], local_trans[p][s][key], hmm.LZERO)
            for merged, part in zip(hmm.pdf_accumulators[p][s], local_pdfs[p][s]):
                # Mean and variance numerators are weighted sums in the linear domain.
                for key in ('mu', 'var'):
                    merged[key]['num'] += part[key]['num']
                occupancy = add_log_counts(merged['mu']['den'], part['mu']['den'], hmm.LZERO)
                merged['mu']['den'] = merged['var']['den'] = merged['weight']['num'] = occupancy
                merged['weight']['den'] = add_log_counts(
                    merged['weight']['den'], part['weight']['den'], hmm.LZERO)


def save_checkpoint(hmm, path):
    temporary = path.with_suffix('.tmp')
    hmm.save_hmm(temporary)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument('--iterations', type=int, default=10, help='Iterations per mixture count')
    parser.add_argument('--output', type=Path, required=True, help='New experiment directory')
    args = parser.parse_args()
    if args.workers < 1 or args.iterations < 1:
        parser.error('workers and iterations must be positive')

    hmm = MonoPhoneHMM()
    hmm.load_hmm('exp/model_3state_1mix/0.hmm')
    utterances = read_training_data(hmm)
    workers = min(args.workers, len(utterances))
    shards = [utterances[i::workers] for i in range(workers)]
    args.output.mkdir(parents=True, exist_ok=False)
    print(f'{len(utterances)} training utterances, {workers} workers', flush=True)
    with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        for mixtures in (1, 2):
            if mixtures == 2:
                hmm.mixup()
            model_dir = args.output / f'model_{hmm.num_states}state_{mixtures}mix'
            model_dir.mkdir()
            save_checkpoint(hmm, model_dir / '0.hmm')
            for iteration in range(1, args.iterations + 1):
                start = time.monotonic()
                # All shards use the same parameters. Merge in stable shard order.
                results = list(pool.map(accumulate_shard, repeat(hmm, workers), shards))
                hmm.reset_accumulators()
                likelihood = 0.0
                for pdf_stats, trans_stats, shard_likelihood in results:
                    merge_shard(hmm, pdf_stats, trans_stats)
                    likelihood += shard_likelihood
                hmm.update_parameters()
                hmm.pdf_accumulators = hmm.trans_accumulators = None
                save_checkpoint(hmm, model_dir / f'{iteration}.hmm')
                print(f'{mixtures}mix {iteration}/{args.iterations}: '
                      f'{len(utterances)} utterances, avg log likelihood '
                      f'{likelihood / len(utterances):.6f}, {time.monotonic() - start:.2f}s',
                      flush=True)


if __name__ == '__main__':
    main()
