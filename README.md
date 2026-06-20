# Predicting on the TUEG Dataset using VLMs and Associated Text Rationales

The purpose of this research project is to help neurologists with predicting various conditions on the brain, like epilepsy, through spectrograms of EEG readings.
Our focus is to provide meaningful and accurate text rationales in tandem with the predictions. This would make such predictions far more useful in a clinical
setting as neurologists could verify results via the generated text rationale.

## Getting Started

This repository does not come with the data. To get started, download the dataset you are planning to use via the `download.sh` script, and move it to its
respective folder. Then run `generate.py` to generate the preprocessed `data` directory and `labels.csv` file. `data` is one directory containing the generated
spectrogram `.png` images, each corresponding to one `.edf` file. `labels.csv` is a spreadsheet that maps input (the `.png` image) to the output, which are boolean
columns representing whether or not a given artifact exists in the associated spectrogram `.png` image.

Before running the evaluation via `eval.py`, customize `config.yml` to specify the model you would like to use, which dataset you would like to evaluate,
and some other relevant parameters like `limit` which specifies how many images to evaluate.

## Additional Resources

Each dataset comes with a `generate-annotated.py` file which generates an equivalent set of `.png` images, except that the artifacts are visually annotated
with a colour. This may be useful for manual review of findings.

