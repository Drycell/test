# 한국어 실행 가이드 (EXP-001)

이 저장소는 **C. elegans connectome 기반 CTRNN**과 **3D WingedWorm MuJoCo 환경**에서
구조 진화(Structure-only)를 우선 검증하기 위한 연구용 프로토타입입니다.

## 1) 빠른 실행 순서

```bash
make install
make smoke
make smoke-exp001
make train-exp001
make eval-latest
make report-latest
```

## 2) 주요 파일

- 실험 스펙: `AGENTS.md`
- 기본 실험 설정: `configs/experiment/exp001_mock.yaml`
- 긴 실행 설정: `configs/experiment/exp001_mock_long.yaml`
- mock connectome 데이터: `data/mock_connectome/`
- 실제 커넥톰 번들 데이터: `data/real_connectome/`
- 훈련 결과 디렉터리: `runs/latest/`

## 3) 산출물 확인

`make train-exp001` 이후 `runs/latest/`에 아래 핵심 파일이 생성됩니다.

- `config_resolved.yaml`
- `manifest.json`
- `metrics.csv`
- `generation_summary.csv`
- `best_genome.json`
- `best_graph_before_after.json`
- `best_graph_before.gml`
- `best_graph_after.gml`
- `best.ckpt`
- `last.ckpt`
- `trajectory_best.npz`

이후:
- `make eval-latest` → `eval_summary.json`
- `make report-latest` → `report.md`, `edit_frequency.csv`

## 4) 환경 이슈 대응

의존성 설치가 네트워크/프록시 정책으로 막히면 `make install`이 실패할 수 있습니다.
이 경우 코드 정합성은 아래로 최소 점검 가능합니다.

```bash
python -m compileall src scripts tests
```

다만 실제 시뮬레이션(`mujoco`, `gymnasium`, `numpy`) 실행은 패키지 설치가 필요합니다.


## 5) 실제 커넥톰 실행

```bash
PYTHONPATH=src python scripts/train.py --config configs/experiment/exp001_real.yaml
```
