# BrightFlow (WACV 2023)
This repository contains the official implementation of [BrightFlow: Brightness-Change-Aware Unsupervised Learning of Optical Flow](https://openaccess.thecvf.com/content/WACV2023/html/Marsal_BrightFlow_Brightness-Change-Aware_Unsupervised_Learning_of_Optical_Flow_WACV_2023_paper.html) that has been published to the **IEEE Winter Conference on Applications of Computer Vision (WACV) 2023**.

## Requirements

```
requirement.txt
```

## Datasets

To train/evaluate BrightFlow or the baseline without BrightFlow, please download the required datasets:
* [Sintel](http://sintel.is.tue.mpg.de/)
* [KITTI](https://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=flow)
* [HD1K](http://hci-benchmark.iwr.uni-heidelberg.de/)

## Training

#### Baseline



```
sh script/train_baseline.sh
```

#### BrightFlow

```
sh script/train_brightflow.sh
```

## Evaluation 

The checkpoints of trained models are available [here](https://drive.google.com/drive/folders/1r2LrW3svWW1kQ98u2D8CaapHXjrV43zD?usp=drive_link). 

```
sh script/eval.sh
```

## Acknowledgements

We thank authors of [RAFT](https://github.com/princeton-vl/RAFT/), [GMA](https://github.com/zacjiang/GMA), [SCV](https://github.com/zacjiang/SCV) and [SMURF](https://github.com/google-research/google-research/tree/master/smurf) for their great work and for sharing their code.

## Citation

```
@inproceedings{marsal2023brightflow,
  title={BrightFlow: Brightness-Change-Aware Unsupervised Learning of Optical Flow},
  author={Marsal, Remi and Chabot, Florian and Loesch, Angelique and Sahbi, Hichem},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={2061--2070},
  year={2023}
}
```

## License

This project is under the CeCILL license 2.1.