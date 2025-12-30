# ImgMatching_v01 개발 핸드오프

## 1) 목표/현재 상태 요약
- 목표: `query`→`reference` 명칭 정리, 폴더 기반 데이터 선택, TopK/PCA 제거, manifest 정규화 후 배치 임베딩 파이프라인(run_manifest*, Test_Embedding*) 안정화.
- 현재: `manifest*.json`이 folder/indices 기반으로 정리되었고, TopK/PCA 로직과 설정이 제거됨. `variants.py`가 `project/imatch`로 이동/구현되어 런타임 variant 생성 지원. `run_manifest.py`의 스킵 버그 수정.

## 2) 변경된 파일 목록 및 핵심 변경점
- `project/json/manifestReference.json`, `project/json/manifest.json`, `project/json/manifestQuery.json`
  - image_groups를 `altitudes`→`folder`/`indices` 구조로 개편. dataset_key 정리(`jamshill_image` 등). rotation은 선택 필드.
- `project/json/data_key.json`
  - 데이터셋별 images에 folder 필드 기반 정의(01_01, 01_02, 01_03 경로 반영).
- `project/Test_Embedding.py`, `project/Test_Embedding4Query.py`
  - TopK/PCA 관련 파라미터, 처리 로직, 메타데이터 완전 제거. patch 후처리는 variant 정보와 keep_ratio만 유지.
- `project/run_manifest.py`, `project/run_manifestQuery.py`
  - folder 기반 이미지 그룹 확장으로 단순화(altitude 제거). `variants` 경로를 `imatch.variants`로 수정.
  - 예시/에러 메시지 및 stride 처리 업데이트. `run_manifest.py`의 잘못된 `continue` 들여쓰기 버그 수정(작업이 스킵되던 문제).
- `project/imatch/postprocess.py`
  - TopK 변형 및 관련 유효성 검사/라벨 생성/레지스트리 제거. 지원 변형은 raw/subsample만.
- `project/imatch/variants.py` (새 위치/파일)
  - `RuntimeVariant` dataclass와 `build_runtime_variant` 구현(raw/subsample 전용, stride 검증 포함).

## 3) 실행/테스트 방법
- 전제: `.env`에 DATASET_KEY, REFERENCE_ROOT/REFERENCE_PREFIX/REFERENCE_DATASET_PREFIX, EXPORT/REFERENCE/QUERY_EMBED_ROOT 등 경로를 실제 데이터에 맞게 설정.
- 참조 임베딩 (manifestReference.json 사용):
  ```bash
  python project/run_manifestQuery.py --manifest project/json/manifestReference.json
  ```
- 쿼리/이미지 세트 임베딩 (manifest.json 또는 manifestQuery.json):
  ```bash
  python project/run_manifest.py --manifest project/json/manifest.json
  # 또는
  python project/run_manifestQuery.py --manifest project/json/manifestQuery.json
  ```
- 단일 디버그 실행(옵션): `python project/Test_Embedding.py` / `python project/Test_Embedding4Query.py` (환경변수와 경로 세팅 필요).

## 4) 남은 TODO 우선순위
1. 실제 데이터 폴더/파일과 `data_key.json` 및 각 manifest의 `folder` 리스트 일치 여부 확인.
2. `.env` 경로(REFERENCE_ROOT, EXPORT/EMBED_ROOT 등) 검증 및 Docker/compose 환경 변수 동기화 필요 시 반영.
3. `Generate_DenseFT*.py` 등 후속 파이프라인이 folder 기반 구조와 TopK/PCA 제거에 문제 없는지 스모크 테스트.
4. 필요 시 `rotation` 필드(참조 manifest) 유지/제거 결정 및 파일명 규칙(`*_rot000*`) 확인.
5. 새 위치 `project/imatch/variants.py` 모듈을 사용하는 다른 스크립트가 있다면 임포트 경로 점검.

## 5) 발생한 에러/로그와 추정 원인
- `ModuleNotFoundError: No module named 'variants'`: variants.py가 imatch로 이동했으나 임포트 경로 미수정 → `from imatch.variants import build_runtime_variant`로 해결.
- `Embedding... 0/0`에서 멈춤: `run_manifest.py` 내 잘못된 들여쓰기(`continue`가 루프 밖으로 나가 모든 작업 스킵) → 들여쓰기 수정.
- TopK/PCA 관련 오류 가능성: 해당 기능을 완전 제거했으므로 이전 설정/환경변수(PCA_BASIS_PATH 등)는 더 이상 필요 없음.***
