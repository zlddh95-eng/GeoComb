# GeoComb Windows 개발 패키지 v1

이 압축파일에는 Windows용 GeoComb 프로그램을 만들기 위한 소스, 최종
모델 4개, 회귀검증용 이미지와 Linux 기준 결과가 포함되어 있습니다.

## CPU 버전 시험

1. 압축을 Windows 폴더에 풉니다.
2. `install_cpu.bat`을 실행합니다.
3. `validate_cpu.bat`을 실행합니다.
4. 검증에 통과하면 `run_source_cpu.bat`으로 프로그램을 시험합니다.
5. `build_cpu.bat`을 실행하면 다음 파일이 생성됩니다.

```text
release\GeoComb_Windows_CPU.zip
```

## NVIDIA GPU 버전 시험

1. `install_gpu.bat`
2. `validate_gpu.bat`
3. `run_source_gpu.bat`
4. `build_gpu.bat`

결과:

```text
release\GeoComb_Windows_GPU.zip
```

## 분석 결과

프로젝트 폴더 안에 다음 결과가 생성됩니다.

```text
project.geocomb
results\GeoComb_results.xlsx
results\cells\
results\overlays\frame_crop_5000\
results\overlays\original_image\
```

`project.geocomb`가 편집 가능한 원본 프로젝트입니다. Cell 위치나 class를
수정하면 SQLite 프로젝트에 즉시 저장되고 Excel과 overlay가 자동으로
갱신됩니다.

## 편집 단축키

- 숫자 1–8: class 변경
- 더블클릭: cell 추가
- 드래그: center 이동
- Delete: cell 삭제
- 마우스 휠: 확대/축소
- 가운데 마우스 버튼: 화면 이동

## 주의

이 개발 패키지 자체에는 미리 빌드된 `.exe`가 없습니다. Windows 실행파일은
Windows 컴퓨터에서 `build_cpu.bat` 또는 `build_gpu.bat`을 실행하여 만들어야
합니다. 첫 배포 전에는 반드시 회귀검증을 통과해야 합니다.
