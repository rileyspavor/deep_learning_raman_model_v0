        C────C
      /        \
    C    1-D     C      Riley Spavor
    |  Graphene  |   riley.spavor@gmail.com
    C    CNN     C      Itera-x.com
      \        /     
        C────C

A deep learning framework for classifying graphene materials from Raman spectroscopy data using 1D Convolutional Neural Networks.





====================== ADDITIONAL INFORMATION ==================================
(bottom of page)
================================================================================

Basic Intructions for Re-Creating Results

========================== STEP 1: DOWNLOAD DATA ===============================

Google Drive Links
'''

See file directory for folder hierarchy 

data.zip/

https://drive.google.com/file/d/1AoXoSrj0nOZyZ-3ODmLskPb8Km71ib3s/view?usp=sharing

Download: full project.zip (if not working)

deep_learning_raman_model_v0.zip/

https://drive.google.com/file/d/1wUqeHGZlZFG1h2Ue5Xw7PBDB5L3j1vpk/view?usp=drive_link

'''

========================= STEP 2: INSTALLATION =================================


```bash
pip install -r requirements.txt

```
=========================== STEP 3: TRAIN MODEL ================================


**Note:** This step can be skipped as the model is already trained. Retraining can take 1+ hours.

```bash
python scripts/train_v3.py --data "data/processed/v3_data/synthetic_graphene_parametric_9class_v2.npz"
```

**Model Versioning:** Each training run automatically creates a new versioned folder (e.g., `model_v2`, `model_v3`, `model_vn`) to prevent overwriting previous models. The latest version is automatically used when loading models.

Model structure:
```
models/saved_models_v3/
├── model_v3/          # Latest model version that we will use

```

**Options: please ignore**
- `--data`: Path to training data .npz file
- `--output`: Base output directory (default: `models/saved_models_v3`)
- `--epochs`: Number of epochs (default: 100)
- `--batch-size`: Batch size (default: 32)
- `--lr`: Learning rate (default: 0.001)
- `--pretrained-model`: Directory containing pretrained model (default: same as --output)
- `--pretrained-version`: Specific model version to load (e.g., 3). If not specified and --pretrained-model is set, uses latest version
- `--from-checkpoint VERSION`: Shortcut to load checkpoint from --output directory (e.g., `--from-checkpoint 3`)

**Examples:**
```bash
# Train from scratch (default) control + c to quit
python scripts/train_v3.py --data "data/processed/v3_data/synthetic_graphene_parametric_9class_v2.npz"


#Advances setting for training
    # Continue training from latest model version ex 4
    python scripts/train_v3.py --data "data/processed/v3_data/synthetic_graphene_parametric_9class_v2.npz" --pretrained-version 4

    # Continue training from specific model version (shortcut)
    python scripts/train_v3.py --data "data/processed/v3_data/synthetic_graphene_parametric_9class_v2.npz" --from-checkpoint 1

    # Load pretrained model from different directory
    python scripts/train_v3.py --data "data/processed/v3_data/synthetic_graphene_parametric_9class_v2.npz" --pretrained-model "models/saved_models_v3" --pretrained-version 2
```
'''

========================= STEP 4: TEST ON REAL DATA ============================


Test the trained model on real Raman spectrum files. The script automatically preprocesses .txt files and aligns them to the model's wavenumber grid. Results are saved to a text file.

```bash
# Test all files in data/test/testing_real_data directory
python scripts/test_real_data_v3.py

#Advanced testing setting
    # Test all files with specific model directory
    python scripts/test_real_data_v3.py --model-dir models/saved_models_v3

    # Test a single file
    python scripts/test_real_data_v3.py --file "G-1 With Peaks.txt"

    # Save results as CSV file
    python scripts/test_real_data_v3.py --csv --output prediction_results.csv

    # Custom output file name
    python scripts/test_real_data_v3.py --output my_results.txt
```

**Options: Please Ignore**
- `--data-dir`: Directory with .txt spectrum files (default: `data/test`)
- `--file`: Specific file to test (tests all files if not specified)
- `--model-dir`: Base directory containing models (default: `models/saved_models_v3`). Automatically uses latest version.
- `--output`: Output file for results (default: `prediction_results.txt`)
- `--csv`: Save results as CSV file instead of text file

**Note:** The test script automatically finds and loads the latest model version. To use a specific version, you can specify the full path: `--model-dir models/saved_models_v3/model_v2`

====================== STEP 5: VISUALIZE TRAINING DATA =========================


Plot one random Raman spectrum for each class in the training dataset to visualize what each class looks like.

```bash
# Plot spectra from training dataset (used to train model_v3)
python scripts/plot_training_spectra.py --data-file "data/processed/v3_data/synthetic_graphene_parametric_9class_v2.npz"

# Plot spectra from training dataset (pick a dataset and n of spectra per class)
python scripts/plot_training_spectra.py
```

======================== STEP 6: VISUALIZE TEST DATA ===========================


Plot all real Raman spectra from the test dataset on a single plot to visualize test data characteristics.

```bash
# Plot all test data spectra (default: data/test/testing_real_data)
python scripts/plot_test_data.py

```

**Options:**
- `--data-dir`: Directory with .txt spectrum files (default: `data/test/testing_real_data`)
- `--output`: Output path for the plot (default: `results/test_data.png`)
- `--no-show`: Don't display the plot (only save)
- `--alpha`: Transparency of spectrum lines (0-1, default: 0.7)
- `--linewidth`: Width of spectrum lines (default: 1.0)

=================== STEP 7: COMPARE TEST VS TRAINING DATA ======================


Pick a test spectrum, run it through the model, and visualize it against training spectra from the predicted class to see how close it matches.

```bash
# Interactive mode - will prompt you to select a test file
python scripts/compare_test_to_training.py

# Use specific test file
python scripts/compare_test_to_training.py --file "G-1 With Peaks.txt"
```

**Options:**
- `--data-dir`: Directory with test .txt files (default: `data/test/testing_real_data`)
- `--training-data`: Path to training .npz file (default: `data/processed/v3_data/synthetic_graphene_parametric_9class_v2.npz`)
- `--model-dir`: Model directory (default: `models/saved_models_v3`)
- `--n-samples`: Number of training spectra to show (default: 10)
- `--output`: Output path for plot (default: auto-generated as `results/test_vs_training_{filename}.png`)
- `--no-show`: Don't display the plot (only save)
- `--file`: Specific test file to use (skips selection prompt)

=================== Additional Information =====================================

(more detailed instructions in docs)

```
project_root/
├── data/                        # All data folders
│   ├── processed/               # Sythetic based datasets after preprocessing
│       ├── v3_data/             # FINALIZED dataset for training
│   ├── test/                    # Real Raman Spectroscopy Test data
│   └── class_labels.json        # Mappings and metadata
├── docs/                        # Documentation (MD and TXT files)
│   ├── MODEL_ARCHITECTURE.txt   # Model architecture details
│   ├── s...data_generation.txt  # Data generation explanation/methods
│   └── ...                      # Other documentation files
├── models/                      # Saved model weights and artifacts
│   └── saved_models_v3/         # Versioned model directories
│       ├── model_v1/            # Model version 1
│       ├── model_v2/            # Model version 2
│       └── model_v3/            # Model version 3 (and newer) #FINAL MODEL
│           ├── model_state_v*.pth      # Model weights
│           ├── model_checkpoint_v*.pth # Training checkpoints
│           ├── class_names_v*.json    # Class name mappings
│           ├── target_grid_v*.npy      # Wavenumber grid
│           └── training_history_v*.png # Training plots
├── results/                     # Output files and visualizations
│   ├── visualizations/          # CNN visualization outputs
│   ├── training_data_visual_*.png  # Training data visualizations
│   └── prediction_results_*.txt    # Prediction results on real_test_data
├── src/                         # Core source code
│   ├── model.py                 # CNN model definition
│   ├── training.py              # Training functions
│   ├── data_ingestion.py        # Data loading utilities
│   ├── preprocessing.py         # Data preprocessing
│   ├── inference.py             # Inference utilities
│   └── utils.py                 # Helper functions
├── scripts/                     # Standalone scripts
│   ├── train_v3.py              # Main training script
│   ├── test_real_data_v3.py     # Test script for real data
│   ├── plot_training_spectra.py # Visualize training data
│   ├── plot_test_data.py        # Visualize test real data
│   ├── compare_test_to_training.py  # Compare test spectrum to training data
│   ├── visualize_cnn_classification.py  # CNN visualization
│   └── ...                      # Other utility scripts
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```
