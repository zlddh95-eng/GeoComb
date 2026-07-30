# GeoComb 플랫폼 릴리스

GeoComb은 꿀벌 벌집 프레임 이미지에서 셀 중심을 검출하고 셀 내용을
8개 생물학적 범주로 분류하는 오픈소스 데스크톱 워크플로입니다. 공개
소프트웨어 버전은 `1.0.0`이며, 논문에서 평가한 동결 소스와 모델을
사용합니다.

## Windows CPU 패키지 빌드

```bat
install_cpu.bat
.venv_cpu\Scripts\python.exe check_environment.py
build_cpu.bat
```

생성 파일:

```text
release\GeoComb_Windows_CPU.zip
```

## Windows NVIDIA GPU 패키지 빌드

```bat
install_gpu.bat
.venv_gpu\Scripts\python.exe check_gpu_environment_v1_9.py
build_gpu.bat
```

생성 파일:

```text
release\GeoComb_Windows_GPU.zip
```

## 선택적 회귀검증

`validate_cpu.bat`과 `validate_gpu.bat`은 승인된 로컬
`regression_test` 자산이 있을 때 릴리스 관리자가 수행하는 단일 이미지
검사입니다. 이미지 기반 회귀검증 자산은 이 공개 소스 저장소에 포함되지
않습니다. 이 검사는 논문에 보고된 29개 이미지 플랫폼 벤치마크와는
구분됩니다.

## 모델과 데이터 배포

동결된 모델 가중치 4개는 해당 버전의 GitHub Release 자산 또는 DOI
아카이브로 배포합니다. 파일의 SHA-256은 `ASSET_MANIFEST.json`에서
검증할 수 있으며, 학습 출처와 사용 범위 및 한계는 `MODEL_CARD.md`에
기록합니다.

분류기 학습에는 보고된 DeepBee 평가 이미지 13장과 겹치지 않는 별도의
DeepBee 프레임 이미지 67장에서 유래한 수동 검토 crop과 label이
포함되었습니다. GeoComb은 DeepBee 원본 이미지, 추출 crop, 수정
annotation을 재배포하지 않습니다. FAIRHiveFrames-1K 원본 이미지도
포함하지 않으며 사용자는 각 공식 출처에서 받아야 합니다.

GeoComb 소스와 Ultralytics 기반 모델 산출물은 `AGPL-3.0-only` 경로로
배포합니다. AGPL과 충돌할 수 있는 별도의 비상업적 제한은 가중치에
추가하지 않습니다. 이는 DeepBee, FAIRHiveFrames-1K 또는 in-house
원본 이미지의 조건을 변경하거나 해당 이미지에 대한 권리를 부여한다는
뜻이 아닙니다.

## 분석 결과

프로젝트 폴더에는 다음 결과가 생성됩니다.

```text
project.geocomb
results\GeoComb_results.xlsx
results\cells\
results\overlays\frame_crop_5000\
results\overlays\original_image\
```

`project.geocomb`가 편집 가능한 원본 프로젝트입니다. 셀 위치나 class를
수정하면 SQLite 프로젝트에 저장되고 Excel과 overlay가 갱신됩니다.

## 편집 단축키

- 숫자 1–8: class 변경
- 더블클릭: cell 추가
- 드래그: center 이동
- Delete: cell 삭제
- 마우스 휠: 확대/축소
- 가운데 마우스 버튼: 화면 이동

상세한 사용법과 빌드·검증 범위는 `USER_GUIDE.md`와
`BUILD_AND_VALIDATION.md`를 참조하십시오.
