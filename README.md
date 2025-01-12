# SelectNAdaptICCVW
This is an official implementation of the paper "SelectNAdapt: Support Set Selection for Few-Shot Domain Adaptation". 

## Dependencies
The python environment is the same as  [Dassl.Pytorch](https://github.com/KaiyangZhou/Dassl.pytorch).

## Datasets
To download PACS dataset please visit this [link](https://www.kaggle.com/datasets/ma3ple/pacs-dataset) , office-31 can be downloaded from [here](https://www.kaggle.com/datasets/xixuhu/office31)

## Models
The source and self-supervised trained models can be downloaded from [here]().

## Run Code
To execute the selection process, you'll have to execute the following command, e.g.
´´´
python main.py --source --target --weights --method
´´´
## Citation 

## Acknowledge
This codebase is an extension of [LCCS](https://github.com/zwenyu/lccs) and also depends on [Dassl.Pytorch](https://github.com/KaiyangZhou/Dassl.pytorch) for pre-training using source datasets. Thanks to their implementation
