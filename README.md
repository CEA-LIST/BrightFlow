# BrightFlow (WACV 2023)
This repository contains the official implementation of [BrightFlow: Brightness-Change-Aware Unsupervised Learning of Optical Flow](https://openaccess.thecvf.com/content/WACV2023/html/Marsal_BrightFlow_Brightness-Change-Aware_Unsupervised_Learning_of_Optical_Flow_WACV_2023_paper.html) that has been accepted to **IEEE Winter Conference on Applications of Computer Vision (WACV) 2023**.

## Requirements

```
requirement.txt
```

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

```
sh script/eval.sh
```

## Checkpoints

Coming soon !

## Acknowledgements

We thank authors of [RAFT](https://github.com/princeton-vl/RAFT/), [GMA](https://github.com/zacjiang/GMA), [SCV](https://github.com/zacjiang/SCV) and [SMURF](https://github.com/google-research/google-research/tree/master/smurf) for their great work and for sharing their code.

## Citation

```
@inproceedings{marsal2023brightflow,
  title={BrightFlow: Brightness-Change-Aware Unsupervised Learning of Optical Flow},
  author={Marsal, R{\'e}mi and Chabot, Florian and Loesch, Ang{\'e}lique and Sahbi, Hichem},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={2061--2070},
  year={2023}
}
```

## License

This project is under the CeCILL license 2.1.