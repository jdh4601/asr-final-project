# ASR Final Project

음성인식입문 기말 과제: 제공된 HMM 음성인식기를 개선하여 **영어 35단어의 인식 정확도**를 높이고, 제출한 코드·모델로 미공개 평가 결과를 재현합니다.

작업 전 [AGENTS.md](AGENTS.md)의 규칙과 [FINAL_PROJECT.md](FINAL_PROJECT.md)의 상세 요구사항을 읽어 주세요. 최신 공식 공지가 우선합니다.

**[현재 진행 상황과 다음 작업 보기](PROGRESS.md)**

## 프로젝트 구성

```text
.
├── AGENTS.md                 # 반드시 지킬 작업 규칙
├── FINAL_PROJECT.md          # 과제 요구사항·평가·제출 기준 정리
├── README.md                 # 팀원 시작 안내
├── bjasr_sc/                 # 기말 과제: 영어 35단어
│   ├── data/
│   │   ├── label/            # 음성 목록, 학습 음소열, 테스트 정답
│   │   ├── wav/              # 학습 WAV 1,050개 (Git 제외)
│   │   └── wav_test/         # 테스트 WAV 525개 (Git 제외)
│   ├── 01compute_features/   # MFCC 추출·학습 특징 통계 계산
│   └── 03gmm_hmm/            # HMM 학습·인식·평가, 영어 사전·음소 목록
└── bjasr_pbw20/              # 참고 실습: 한국어 20단어
```

처리 흐름: **WAV → MFCC → HMM 학습 → 사전 기반 단어 인식 → 정확도 평가**.
핵심 알고리즘은 `03gmm_hmm/hmmfunc.py`에 있습니다.

## 팀원 시작하기

### 1. 저장소 받기

```sh
git clone https://github.com/jdh4601/asr-final-project.git
cd asr-final-project
```

### 2. 데이터 준비

WAV·배포 ZIP·생성 특징·모델은 용량 관리를 위해 Git에 올리지 않습니다. 각 팀원은 수업에서 배포한 `bjasr_sc.zip`을 별도로 확보하세요.

1. ZIP을 프로젝트 밖의 임시 폴더에 압축 해제합니다.
2. 원본 패키지에서 **`data/wav/`와 `data/wav_test/`만** 이 저장소의 `bjasr_sc/data/` 아래로 복사합니다.
3. 저장소의 코드와 `data/label/` 파일은 유지합니다. 배포본으로 덮어쓰면 경로 수정이 사라집니다.

저장소의 `wav.scp`는 `../data/wav/...` 또는 `../data/wav_test/...` 형식입니다. `bjasr_sc/01compute_features/`에서 실행할 때 해석되는 경로이므로 clone 위치에 관계없이 사용할 수 있습니다. PBW20 실습도 같은 방식입니다.

### 3. 실행 환경 확인

Python 3와 NumPy를 사용하는 코드입니다. **과제 규정상 다른 라이브러리·패키지를 설치하지 않습니다.** 수업에서 허용·제공한 환경을 사용하세요. 로컬 실행 환경과 측정값은 [실험 기록](experiments/midterm-mixtures.md)에 있습니다.

```sh
python3 --version
python3 -c 'import numpy; print(numpy.__version__)'
```

NumPy가 없으면 먼저 수업 환경과 설치 허용 범위를 확인하세요. 새 패키지 설치로 임의 해결하지 않습니다.

## 기본 학습·평가 실행

아래 명령은 저장소 루트에서 시작합니다. 각 단계의 오류를 해결한 뒤 다음 단계로 진행하세요. 재실행하면 같은 경로의 특징·모델·결과를 덮어쓸 수 있으므로 이전 실험 결과를 먼저 보관하세요.

```sh
cd bjasr_sc/01compute_features
python3 01_compute_mfcc_kr_sc.py
python3 02_compute_mean_std_kr.py

cd ../03gmm_hmm
python3 00_make_label_kr.py
python3 01_make_proto_kr.py
python3 02_init_hmm_kr.py
python3 03_train_sgmhmm_kr.py
python3 06_recognize_kr_sc.py
python3 eval.py
```

- 기본 특징: 13차원 MFCC, 25ms 프레임, 10ms 이동.
- 단일 가우시안 학습: 1,050발화, 10회 반복. 인식은 기본적으로 `exp/model_3state_1mix/10.hmm`을 사용합니다.
- 결과: `bjasr_sc/03gmm_hmm/exp/data/test/result.txt`.
- `eval.py` 출력의 **Total이 525인지** 함께 확인하세요. 정확도만으로 누락 여부를 판단하지 않습니다.
- GMM 실험은 `03_train_gmmhmm_kr.py`의 발화 수·혼합 수·출력 경로를 먼저 확인하고, 인식에 사용할 모델도 맞춰야 합니다.
- 생성된 `feats.scp`에는 로컬 절대 경로가 포함되므로 다른 컴퓨터에서는 특징을 다시 추출합니다.

문서의 **34.10%는 공식 슬라이드 예시**입니다. 팀의 기본 성능은 직접 실행해 기록해야 합니다.

## 협업 방법

한 브랜치에서 한 가지 개선을 진행하고 PR에서 변경 이유와 검증 결과를 공유합니다.

```sh
git switch main
git pull --ff-only
git switch -c feature/experiment-name
# 작업 후 변경 파일과 대용량·민감 파일 포함 여부 확인
git status --short
git diff
git add <변경한-파일>
git diff --cached --stat
git commit -m "feat: describe the experiment"
git push -u origin feature/experiment-name
```

- GitHub에서 `main`을 대상으로 PR을 만들고 다른 팀원이 검토한 뒤 병합합니다.
- 기존 제공 코드는 보존하고 개선 코드는 새 스크립트·함수로 추가합니다. AI 도움 코드에는 규정된 주석과 출처를 남깁니다.
- 실험 기록은 `experiments/<실험명>.md`에 작성하세요. 폴더는 첫 기록을 추가할 때 생성합니다.
- 기록 항목: 변경점, 데이터·증강 출처와 수량, 특징·모델 설정, 정답 수/525, 정확도, 모델 바이트 크기, 학습·인식 시간, 환경, 실행 명령, 커밋 ID.
- 기본 성능을 먼저 기록하고 한 번에 한 가지 변경부터 비교합니다. 테스트 정답을 학습 데이터로 사용하지 않습니다.
- WAV·ZIP·특징·모델·로그·인증정보는 커밋하지 않습니다. 필요한 결과 모델은 팀에서 별도 전달하고 해당 커밋·설정과 함께 보관합니다.

## 제출 준비

제출 구성과 일정은 `FINAL_PROJECT.md` 및 최신 LMS 공지를 확인하세요. GitHub 저장소와 최종 제출 ZIP은 구성이 다릅니다. 제출에는 최종 모델·테스트 특징·인식 결과 등도 필요합니다.

미공개 평가 데이터는 당일 525개가 공개되며, **이미 제출한 코드·모델에서 입력만 교체**해야 합니다. 결과 누락·중복과 2시간 내 실행 가능 여부를 미리 확인하세요. 보고서에는 평가 캡처·모델 크기·출처·외부 데이터/증강 내역을 기록하고, 정확히 2쪽의 `.hwp`와 카피킬러 결과를 준비합니다.

기반 코드 출처: [비제이퍼블릭 『파이썬으로 배우는 음성인식』 공개 코드](https://github.com/bjpublic/python_speech_recognition). 실제 사용한 코드·자료의 출처는 최종 보고서에도 명기합니다.
