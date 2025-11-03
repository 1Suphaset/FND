# FakeNewsDetection

This repository contains notebooks and artifacts for a Thai fake-news detection pipeline.

Contents
- `Core/InitData` - data cleaning and undersampling notebooks and results
- `Core/Features` - tokenization and TF-IDF extraction
- `Core/Models` - training (neural network) and saved model artifacts
- `Core/RawData` - original dataset (CSV)

Quick notes
- Notebooks use relative paths (e.g. `../RawData/NewsData10200records.csv` and `result/*`). To run notebooks without path errors, open Jupyter with the working directory set to `Core` or run each notebook from its folder.
- Large artifacts (saved models, result pickles) are stored under `Core/**/result/`.

Dependencies
See `requirements.txt` for a minimal list of Python packages needed to reproduce the notebooks.

Running locally (Windows cmd.exe)
1. Create and activate a virtual environment:

	python -m venv venv
	venv\Scripts\activate

2. Install dependencies:

	pip install -r requirements.txt

3. Start Jupyter in the `Core` folder (so relative paths match):

	cd Core
	jupyter notebook

Then open the notebooks you want to run: `InitData/PrepareData.ipynb`, `Features/TFIDF.ipynb`, `Models/NeuralNetwork.ipynb`.

Notes on reproducibility and next steps
- Add exact package versions if you need deterministic installs (pin versions in `requirements.txt`).
- Consider adding a small `run_all.py` or a notebook that sequentially executes preprocessing → features → training.
- Add a `.gitignore` to avoid committing large result files (one is included in the repo now).

If you'd like, I can add a runnable script to execute the pipeline end-to-end and pin package versions.