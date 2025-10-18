datasets/
├── FiFA-100k/
│   └── <parquet files>
│   └── data/
│       └── train/
│           ├── <image files: .jpg, etc>
│       └── latent/
│           ├── <image .npy, .npz>
├── manifest/
│   ├── fifa_10k.json
│   ├── fifa_50k.json
│   ├── fifa_100k.json
│   └── set1/   #   Parse from the set 10K
│       ├── good_from_10k.json
│       └── dub_from_10k.json
│   └── from_100k/   #   Parse from the set 10K
│       ├── good_from_10k.json
│       └── dub_from_10k.json
