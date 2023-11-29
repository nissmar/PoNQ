

# PoNQ

### Direct Optimization

Direct Optimization

````
python src/direct.py configs/direct_thingi.yaml -grid_n 32
python src/direct.py configs/direct_thingi.yaml -grid_n 64
python src/direct.py configs/direct_thingi.yaml -grid_n 128
````

Metrics 

````
python src/eval/eval_THINGI.py quadrics_thingi_32/
python src/eval/eval_THINGI.py quadrics_thingi_64/
python src/eval/eval_THINGI.py quadrics_thingi_128/
````

Watertightness and triangle count

````
python src/eval/check_watertight.py quadrics_thingi_32
python src/eval/check_watertight.py quadrics_thingi_64
python src/eval/check_watertight.py quadrics_thingi_128
````

### Model training

```
cd learning
python train_cnn.py ../configs/abc_cnn.yaml
```



### Evaluation

## PoNQ
```
python src/generate_all_CNN.py configs/eval_cnn.yaml
```

```
python src/eval/eval_all.py configs/eval_cnn.yaml
```

## PoNQ-lite

```
python src/generate_all_CNN.py configs/eval_cnn.yaml -subd 1
```

```
python src/eval/eval_all.py configs/eval_cnn.yaml -subd 1
```