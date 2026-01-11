# Data Preparation

## Dataset Download

We use a curated version of the WLASL dataset provided by FiftyOne on Hugging Face.
Since the original script to download the dataset has too many broken links, we use the curated version from FiftyOne.

```bash
uvx hf download Voxel51/WLASL --repo-type=dataset --local-dir=./data/WLASL
```

Prepare the reduced WLASL dataset by running the script:

```bash
uv run ./src/sign_language_model/scripts/build_reduced_wlasl.py \
    --wlasl-root ./data/WLASL/ \
    --output-root ./data/wlasl_reduced \
    --glosses "before,cool,thin,go,drink,help,computer,cousin,who,bowling,trade,bed,accident,tall,thanksgiving,candy,short,pizza,man,no,wait,good,bad,son,like,doctor,now,find,you,thank you,please,hospital,bathroom,me,i" \
    --target-fps 24
```

## Dataset Structure

```bash
data/
├── WLASL/
│   ├── .cache/                 # Hugging Face cache
│   ├── data/
│   │   ├── data_0/
│   │   │   ├── *.mp4
│   │   │   └── ...
│   │   ├── data_1/
│   │   │   ├── *.mp4
│   │   │   └── ...
│   │   └── ...
│   ├── fiftyone.yml
│   ├── metadata.json
│   ├── frames.json
│   ├── samples.json
│   └── README.md
└── wlasl_reduced/
    ├── videos/
    │   ├── accident/
    │   │   ├── *.mp4
    │   │   └── ...
    │   ├── bathroom/
    │   │   ├── *.mp4
    │   │   └── ...
    │   └── ...
    ├── features_i3d/ # I3D features
    │   ├── accident/
    │   │   ├── *.npy  # shape: (2, C) where 2 repreesents flow & rgb, C is feature dim (e.g., 1024)
    │   │   └── ...
    │   ├── bathroom/
    │   │   ├── *.npy
    │   │   └── ...
    │   ├── ...
    │   └── README.md
    ├── features_kps/ # Keypoint features
    │   ├── accident/
    │   │   ├── *.npy  # shape: (C, T, V) = (3, T, 75), where T is number of frames, V is number of keypoints, C is coordinate dims (x,y,z)
    │   │   └── ...
    │   ├── bathroom/
    │   │   ├── *.npy
    │   │   └── ...
    │   ├── ...
    │   └── README.md
    ├── splits/
    │   ├── train.csv
    │   └── test.csv
    ├── gloss_map.json
    ├── metadata.csv
    └── README.md
```
