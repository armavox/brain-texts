python train/trainTorch.py -i /data/brain/rs-mhd-dataset-augmented -c /data/brain/checkpoints -e 15
python train/trainTorch.py -i /data/brain/rs-mhd-dataset-augmented -c /data/brain/checkpoints -e 15 -l 0.01
python train/trainTorch.py -i /data/brain/rs-mhd-dataset-augmented -c /data/brain/checkpoints -e 15 -l 0.01 -d 0.3
python train/trainTorch.py -i /data/brain/rs-mhd-dataset-augmented -c /data/brain/checkpoints -e 15 -l 0.0001
python train/trainTorch.py -i /data/brain/rs-mhd-dataset-augmented -c /data/brain/checkpoints -e 15 -l 0.0001 -bs 8
python train/trainTorch.py -i /data/brain/rs-mhd-dataset-augmented -c /data/brain/checkpoints -e 15 -l 0.0001 -d 0.3
python train/trainTorch.py -i /data/brain/rs-mhd-dataset-augmented -c /data/brain/checkpoints -e 15 -l 0.0001 -d 0.3 -bs 8
