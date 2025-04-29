Pose-Coach/
├── main.py                # 메인 실행 파일
├── checkpoints/           # Pretrained models 저장 폴더
│   ├── vitpose_base.pth
│   ├── vitpose_small.pth
│   ├── vitpose_large.pth
│   └── vitpose_huge.pth
├── label_data/            # 운동 레이블, 메타 데이터
├── mmcv/                  # ViTPose에 필요한 mmcv 파일
├── user_data/             # 사용자 입력 데이터 (영상, 결과)
└── utils/                 # pose normalization, DTW 등 유틸리티 코드
