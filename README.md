

# PoNQ: a Neural QEM-based Mesh Representation
[Nissim Maruani](https://nissmar.github.io), [Maks Ovsjanikov](https://www.lix.polytechnique.fr/~maks/), [Pierre Alliez](https://team.inria.fr/titane/pierre-alliez/), [Mathieu Desbrun](https://pages.saclay.inria.fr/mathieu.desbrun/).


### Direct Optimization

Direct optimization on Thingi32:
````
python src/direct.py configs/direct_thingi.yaml -grid_n 32
python src/direct.py configs/direct_thingi.yaml -grid_n 64
python src/direct.py configs/direct_thingi.yaml -grid_n 128
````

Compute various metrics (CD, F1, NC, ECD, EF1):

````
python src/eval/eval_THINGI.py quadrics_thingi_32/
python src/eval/eval_THINGI.py quadrics_thingi_64/
python src/eval/eval_THINGI.py quadrics_thingi_128/
````

Check watertightness and count mesh elements: 

````
python src/eval/check_watertight.py quadrics_thingi_32/
python src/eval/check_watertight.py quadrics_thingi_64/
python src/eval/check_watertight.py quadrics_thingi_128/
````

### Model training

```
cd learning
python train_cnn.py ../configs/abc_cnn.yaml
```



### Evaluation

## PoNQ
Evaluate network on both Thingi32 and ABC:

```
python src/generate_all_CNN.py configs/eval_cnn.yaml
```
Compute various metrics (CD, F1, NC, ECD, EF1):

```
python src/eval/eval_all.py configs/eval_cnn.yaml
```

## PoNQ-lite
Evaluate network on both Thingi32 and ABC:

```
python src/generate_all_CNN.py configs/eval_cnn.yaml -subd 1
```
Compute various metrics (CD, F1, NC, ECD, EF1):

```
python src/eval/eval_all.py configs/eval_cnn.yaml -subd 1
```