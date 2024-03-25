# Pytorch Lab Framework Ver 2.0

---

## New Feature:

- ### New config yaml file

  Please use ```configs/default/classification.yaml``` or others default config file.

- ### New Project Structure

    ```bash
    classification 
    ├── runs
    │   └── 20240321_171939   (Experiment time)
    │       ├── classification.yml  (Backup config file)
    │       ├── cls_id_to_label.txt
    │       └── weights (save in every epoch)
    │           ├── 20240321_183441_Epoch1_Top1#99.5703.pth
    │           └── 20240321_183636_Epoch2_Top1#98.4375.pth
    └── temp  (Test output dir)
        ├── classification
        │   ├── label1
        │   ├── label2
        │   ├── ...
        │   └── classification_output.txt
        └── segmentation
    ```

- ### Add MLFlow UI Flow

  Using code: ```mlflow ui``` in your Terminal

    ```bash
    [Terminal]
    >>> cd pytorch-lab
    >>> mlflow ui
    >>> INFO:waitress:Serving on http://127.0.0.1:5000
    ```

- ### New Code Structure

  The biggest change is in the ```src/trainer.py``` file

---

## Delete:

Del pytorch DDP mode

---

## Using

### Train:

- Configure your config files e.g. ```mnist.yml```
- Select your config file path in ```train.py```
- Run the ```train.py```

### Test:

- Configure the ```test_config``` keyword in your config file
- Select your config file path in ```test.py```
- Run the ```test.py```
- See the output in ```project/xxx```