# IARA
This project aims to disseminate acoustic data through the implementation of scientific work.
The repository contains PyTorch code for reading the archive in different configurations,
    separating standard folds, and training structures for the results presented in the article.

## Authors
- **Main author**: Evandro Rocha
- **Advisor**: Natanael Junior

## Repository Structure

- **data/:** This directory serves as the default location to store raw and processed data.
- **notebooks/:** Explore demonstrations and tutorials in Jupyter notebooks.
- **src/:** Contains the code organized as a pip package. It includes functionality for reading data, separating folds, and training models
- **test_scripts/:** Utilize scripts for development and testing of features within the pip package.
- **training_scripts/:** Access scripts used for training, generating plots, and creating tables presented in the final article.

## Dataset

The IARA dataset is publicly available on Zenodo:

> **DOI:** [10.5281/zenodo.15777429](https://doi.org/10.5281/zenodo.15777429)

Download the archive and extract the contents into the `data/iara/` directory. The dataset contains ~50 GB of audio files organized in collections A–H:

| Collection | Description |
|---|---|
| A | OS Near CPA In (~11 GB) |
| B | OS Near CPA Out (~5 GB) |
| C | OS Far CPA In (~9 GB) |
| D | OS Far CPA Out (~11 GB) |
| E | OS Background (~4 GB) |
| F | Glider CPA In (~2 GB) |
| G | Glider CPA Out (~2 GB) |
| H | Glider Background (~2 GB) |

Expected structure after extraction:
```
data/iara/
├── A/
├── B/
├── C/
├── D/
├── E/
├── F/
├── G/
└── H/
```

## Installation

### Using Docker (Recommended)

This project includes a multi-platform Docker setup that works on Windows/Linux (with or without NVIDIA GPU) and macOS (including Apple Silicon M1/M2).

Shortcut commands are available via `make` (Linux/Mac) or `make.bat` (Windows):

| Command | Linux/Mac | Windows | Description |
|---|---|---|---|
| Start container (CPU / Mac M1) | `make up` | `.\make up` | Build and start without GPU |
| Start container (NVIDIA GPU) | `make up-gpu` | `.\make up-gpu` | Build and start with GPU support |
| Stop container | `make down` | `.\make down` | Stop and remove container |
| Open interactive shell | `make bash` | `.\make bash` | Enter the container terminal |
| Run CNN test | `make test-cnn` | `.\make test-cnn` | Run `test_scripts/cnn.py` |
| Run all trainings | `make run-all` | `.\make run-all` | Run `training_scripts/run_all.sh` |

**1. For Mac M1/M2 or Machines with CPU only:**
```bash
make up
make bash
```

**2. For Machines with NVIDIA GPU:**
```bash
make up-gpu
make bash
```

> **Note:** Your local project directory is mapped into the container. Any changes you make to the code locally will be reflected inside the container immediately.

### Development Mode
To install the IARA library in development mode, navigate to the `src` folder and run the following command
```bash
pip install -e . --user
```

### Deployment Mode
To install the IARA library in deploy mode, navigate to the `src` folder and run the following command
```bash
pip install .
```

## Usage

After entering the container with `make bash` (or `.\make bash` on Windows), the following scripts are available:

### Test Scripts (inside container)

```bash
# Test the CNN model pipeline
python test_scripts/cnn.py

# Inspect dataset collections and fold splits
python test_scripts/collection.py

# Test dataset access and loading
python test_scripts/dataset_acess.py

# Test the audio processor (mel-spectrogram generation)
python test_scripts/processor.py

# Test the MLP model pipeline
python test_scripts/mlp.py
```

### Training Scripts (inside container)

```bash
# Run CNN grid search for a specific group (1–6)
python training_scripts/grid_search.py -c cnn -t multiclass -f lofar -g 1

# Run all trainings sequentially
bash training_scripts/run_all.sh
```

> **Note:** Audio files are pre-processed into mel-spectrograms on first run and cached in `data/iara_processed/`. Subsequent runs skip this step. Training also saves a checkpoint after each epoch, allowing interrupted runs to resume automatically.

Refer to the `notebooks` directory for a detailed guide on accessing and using IARA.
