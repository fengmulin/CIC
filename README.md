# CIC

Compensating for the Incomplete with the Complete: An Efficient Scene Text Detector

## Environment
The environment, datasets, and usage are based on: [DBNet](https://github.com/MhLiao/DB)
```bash
conda create -n CIC python==3.9
conda activate CIC

conda install pytorch torchvision cudatoolkit=11.3 -c pytorch

git clone https://github.com/fengmulin/CIC.git
cd CIC/

echo $CUDA_HOME
cd assets/ops/dcn/
python setup.py build_ext --inplace

```

## Evaluate the performance
```

CUDA_VISIBLE_DEVICES=0 python eval.py experiments/gap/td500/res50.yaml --box_thresh 0.45 --resume workspace/td500/td500_res50

```




## Training

```CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py path-to-yaml-file --num_gpus 4```

## Acknowledgement
Thanks to [DBNet](https://github.com/MhLiao/DB) for a standardized training and inference framework. 










    

