# Beyond Mean ALPS

**Greater Absolute DTI-ALPS Asymmetry Is Associated with Higher Risk of Conversion to Alzheimer’s Disease** 논문의 분석 코드입니다.

ADNI 참가자 **1,131명, AD 전환 115건**을 대상으로 한 생존분석, 민감도 분석, 딥러닝 예측모형을 포함합니다. 주 노출은 **절대 좌우 ALPS 차이 `abs(ALPS_L - ALPS_R)`**입니다.

## 분석 구성

| 모듈 | 분석 | 논문 대응 |
|---|---|---|
| `primary` | 기저진단 층화 Cox 회귀, 비례위험 검정 | Table 2 |
| `directional_outliers` | 좌우 ALPS와 텐서 성분 비교, 꼬리값 제외, 영향도 분석 | Tables S2–S3, Figure 4A/C |
| `structural` | 구조 비대칭 보정, 검사일 매칭 민감도 분석 | Table S4, Figure S1D |
| `technical_risk` | 기술 공변량 보정, 대안적 비대칭 정의, 조정 절대위험 | Tables S7–S8, Figure 3B/C |
| `reorientation_scalar` | 텐서 재배향 비교, FA와 MD 비대칭 분석 | Tables S5A–S6, Figure 4B, Figure S1B/C |
| `competition` | ALPS와 association ROI의 Dzz 비대칭 공동 모형 | Tables S9A–S9B, Figure 4C |
| `spline` | 제한적 삼차 스플라인 | Figure 3A |
| `prediction` | DeepSurv, CoxTime, DeepHitSingle 반복 교차검증 | Tables S10–S12 |

세부 모형, 표준화, 부트스트랩과 평가 방법은 [분석 방법](docs/ANALYSIS_METHODS.md)에 정리되어 있습니다.

## 입력 자료

전체 분석에는 아래 여섯 파일이 필요합니다. 각 파일은 최종 분석 대상자의 임상 정보와 파생 MRI 측정값을 포함해야 합니다.

| 파일 | 내용 |
|---|---|
| `baseline_clinical_alps.xlsx` | 기저 임상 정보, 추적기간과 사건, 구조 공변량, 좌우 ALPS와 방향별 확산계수 |
| `reoriented_clinical_alps.xlsx` | 기본 분석 자료에 재배향 ALPS, FA, MD, 촬영 조건을 결합한 자료 |
| `structural_mri_volumes.csv` | 검사일별 좌우 구조 부피, 두개강내 부피, QC 정보 |
| `diffusion_motion_metrics.csv` | 촬영 중 변위와 이상치 슬라이스 측정값 |
| `head_orientation_angles.csv` | 머리의 pitch, yaw, roll 각도 |
| `acquisition_plane_angles.csv` | 촬영면 기울기 각도 |

파일은 `data/`에 두거나 `--data-dir`로 경로를 지정합니다. 두 Excel 파일의 시트명은 `survival_dataset_final_age`입니다. 정확한 열 이름, 단위와 코딩은 [입력 데이터 명세](docs/DATA_DICTIONARY.md)를 따릅니다.

개인별 자료와 연구에서 산출한 MRI 측정값은 별도로 준비해야 합니다. ADNI 자료는 [공식 접근 절차](https://adni.loni.usc.edu/data-samples/adni-data/)에 따라 이용할 수 있습니다. 코드는 최종 파생 데이터셋에서 시작하며, MRI 전처리는 논문과 인용된 [DTI-ALPS 파이프라인](https://github.com/gbarisano/alps)을 참고합니다. Figure 1과 Table S1의 대상자 선별 수치는 논문에 보고된 집계값입니다.

## 설치 및 실행

Python 3.12 가상환경에서 저장소 루트를 작업 폴더로 사용합니다. 운영체제별 가상환경 활성화 방법은 [영문 안내](README.md#installation)에 있습니다.

```bash
python -m pip install -r requirements.txt
python run_analysis.py --data-dir data --check-inputs
python run_analysis.py --data-dir data --output-dir results
python export_manuscript.py --data-dir data --results results
```

전체 예측 분석은 3개 모형, 2개 예측변수 구성, 5회 반복, 5개 fold, 3개 학습 seed를 사용하여 총 **450회 학습**합니다. 기본 장치는 CPU입니다. 소프트웨어 버전과 하드웨어에 따라 신경망 재학습 결과가 달라질 수 있습니다.

특정 분석만 실행하려면 `--sections`를 사용합니다.

```bash
python run_analysis.py --sections primary directional_outliers
```

논문 표 전체를 내보내려면 8개 모듈의 결과가 모두 필요합니다. VS Code와 Jupyter에서는 [Manuscript_Analyses.ipynb](Manuscript_Analyses.ipynb)를 사용할 수 있습니다.

## 결과와 검증

분석 결과는 `results/` 아래에, 논문 표와 그림 원천값은 `results/manuscript/`에 저장됩니다. 최종 그림의 조립과 MRI 예시 이미지 제작은 이 내보내기 작업에 포함되지 않습니다.

개인별 자료 없이 실행할 수 있는 테스트와 참조 결과 비교 명령은 다음과 같습니다.

```bash
python -m unittest discover -s tests -v
python validate_results.py --results results --skip-prediction
```

예측 분석의 참조값은 저장된 OOF 예측값에 대응합니다. 저장 예측값 재평가와 결과 비교 범위는 [재현성 안내](docs/REPRODUCIBILITY.md)에 설명되어 있습니다.

생성된 결과에는 대상자 식별자가 포함될 수 있으므로 비공개로 관리합니다. 기본 `data/`와 `results/` 경로에는 Git 제외 규칙이 적용됩니다.

## 인용과 라이선스

코드 사용 시 관련 논문을 인용해 주세요. 서지정보는 [CITATION.cff](CITATION.cff), 코드 이용 조건은 [MIT License](LICENSE)에 있습니다. ADNI 자료와 외부 소프트웨어에는 각각의 이용 조건이 적용됩니다.
