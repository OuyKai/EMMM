# EMMM
Implementations of model **EMMM**  described in our paper.

## Environment Requirement

The project is implemented using python 3.6 and tested in Linux environment. We use ``anaconda`` to manage our experiment environment.

Our system environment and CUDA version as follows:

```bash
Ubuntu 16.04.7
GeForce GTX 2080ti CUDA Version: 10.2
```

Our python version and requirements as follows:

- Python 3.6.13
- PyTorch 1.9.0
- NumPy 1.19.5

## Usage

1. Install all the requirements.

2. If there is no folder `datasets`, you can download it from https://drive.google.com/file/d/14QmaHZkcxhmeVmc7gzHPWg4xf2Ufnf3S/view?usp=sharing.

3. Train the model using the Python script `run.py` .

   You can run the following command to train the model **EMMM** on the Tmall dataset:

   ```bash
   python run.py --dataset-name Tmall
   ```

3. If you want to reproduce the best results in the main experiment of our paper, you can click the following link for the best model https://drive.google.com/file/d/1vGHJ1Q5qBRwUMZQig72kL3f4khwb_x8Q/view?usp=sharing.

4. The best models are saved in directory BestModel, and we can evaluate the best model using the Python script `evaluate.py`.

   To load the best model for testing using the following command:
    
   ```bash
   python evaluate.py
   ```
   After running this command, you will see the reproduce results.
