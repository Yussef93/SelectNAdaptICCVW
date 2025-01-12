# SelectNAdaptICCVW
This is an official implementation of the paper "SelectNAdapt: Support Set Selection for Few-Shot Domain Adaptation". 

## Abstract 🚀
![Alt Text](/selectnadapt/assets/Picture1.png "Logo of the Project")

Generalisation of deep neural networks becomes vulner￾able when distribution shifts are encountered between train
(source) and test (target) domain data. Few-shot domain adaptation mitigates this issue by adapting deep neural net￾works pre-trained on the source domain to the target do￾main using a randomly selected and annotated support set from the target domain. This paper argues that randomly selecting the support set can be further improved for effec￾tively adapting the pre-trained source models to the target
domain. Alternatively, we propose SelectNAdapt, an algo￾rithm to curate the selection of the target domain samples, which are then annotated and included in the support set. In particular, for the K-shot adaptation problem, we first leverage self-supervision to learn features of the target do￾main data. Then, we propose a per-class clustering scheme of the learned target domain features and select K rep￾resentative target samples using a distance-based scoring function. Finally, we bring our selection setup towards a practical ground by relying on pseudo-labels for cluster￾ing semantically similar target domain samples. Our ex￾periments show promising results on three few-shot domain adaptation benchmarks for image recognition compared to related approaches and the standard random selection.

## Dependencies
The python environment of this project is the same as  [Dassl.Pytorch](https://github.com/KaiyangZhou/Dassl.pytorch).

## Datasets
To download PACS dataset please visit this [link](https://www.kaggle.com/datasets/ma3ple/pacs-dataset) , office-31 can be downloaded from [here](https://www.kaggle.com/datasets/xixuhu/office31)

Datasets should be placed inside "./DATA/selectnadapt/imcls/data"

## Models
The source and self-supervised trained models can be downloaded from [here](https://faubox.rrze.uni-erlangen.de/getlink/fi2aftRT82WZcSrtw2CrTk/) and place them inside "output_source_models/" and "output_ss_models/" respectively.

## Run Code
To execute the selection process, you'll have to execute the following command, e.g.
```
python select_pacs.py --ouput_dir ""
```
## Citation 
```
@inproceedings{dawoud2023selectnadapt,
  title={SelectNAdapt: Support Set Selection for Few-Shot Domain Adaptation},
  author={Dawoud, Youssef and Carneiro, Gustavo and Belagiannis, Vasileios},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={973--982},
  year={2023}
}

```
## Acknowledge
This codebase is an extension of [LCCS](https://github.com/zwenyu/lccs) and also depends on [Dassl.Pytorch](https://github.com/KaiyangZhou/Dassl.pytorch) for pre-training using source datasets. Thanks to their implementation which made this work possible.
