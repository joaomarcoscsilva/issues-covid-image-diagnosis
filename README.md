# X-Ray Duplicates Detector

To prepare the dataset:

1. Download data:

        bash download_data.sh

2. Preprocess the downloaded data (for example to resolution 256x256):

        python3 process_data.py ./data/curated_xray_dataset 256