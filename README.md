

# PoNQ: a Neural QEM-based Mesh Representation
## CVPR 2024

[Nissim Maruani](https://nissmar.github.io)<sup>1</sup>, [Maks Ovsjanikov](https://www.lix.polytechnique.fr/~maks/)<sup>2</sup>, [Pierre Alliez](https://team.inria.fr/titane/pierre-alliez/)<sup>1</sup>, [Mathieu Desbrun](https://pages.saclay.inria.fr/mathieu.desbrun/)<sup>3</sup>.

<sup>1</sup> Inria, Université Côte d’Azur &emsp; <sup>2</sup> LIX, École Polytechnique, IP Paris &emsp; <sup>2</sup> Inria Saclay, École Polytechnique

<img src='data/banner.pdf' />

## Evaluation

Generate meshes on both Thingi32 and ABC for PoNQ and PoNQ-lite with our pre-trained network:

```
python src/generate_all_CNN.py configs/eval_cnn.yaml
python src/generate_all_CNN.py configs/eval_cnn.yaml -subd 1
```

Compute various metrics (CD, F1, NC, ECD, EF1):

```
python src/eval/eval_all.py configs/eval_cnn.yaml
python src/eval/eval_all.py configs/eval_cnn.yaml -subd 1
```

Check watertightness and count mesh elements: 

````
python src/eval/check_watertight.py FOLDER
````
## Model training

```
cd learning
python src/utils/train_cnn_multiple_quadrics_split.py configs/abc_cnn_multiple_quadrics_split_*.yaml
```

## Direct Optimization

Direct optimization on Thingi32:
````
python src/direct.py configs/direct_thingi.yaml -grid_n 32
python src/direct.py configs/direct_thingi.yaml -grid_n 64
python src/direct.py configs/direct_thingi.yaml -grid_n 128
````

Compute various metrics (CD, F1, NC, ECD, EF1):

````
python src/eval/eval_THINGI.py FOLDER
````


