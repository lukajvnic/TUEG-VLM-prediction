# Predicting on the TUEG Dataset using VLMs and Associated Text Rationales

The purpose of this research project is to help neurologists with predicting various conditions on the brain, like epilepsy, through spectrograms of EEG readings.
Our focus is to provide meaningful and accurate text rationales in tandem with the predictions. This would make such predictions far more useful in a clinical
setting as neurologists could verify results via the generated text rationale.

## Getting Started

This repository does not come with the data. To get started, download the dataset you are planning to use via the `scripts/download.sh` script, and move it to its
respective folder under `datasets/`. Then run that dataset's `generate.py` to generate the preprocessed `data` directory and `labels.csv` file. `data` is one directory containing the generated
spectrogram `.png` images, each corresponding to one `.edf` file. `labels.csv` is a spreadsheet that maps input (the `.png` image) to the output, which are boolean
columns representing whether or not a given artifact exists in the associated spectrogram `.png` image.

Before running the evaluation via `scripts/eval.py`, customize `config.yml` to specify the model you would like to use, which dataset you would like to evaluate,
and some other relevant parameters like `limit` which specifies how many images to evaluate.

Install Python dependencies with:

```bash
pip install -r requirements.txt
```

### Using Ollama Models

This project can run local Ollama vision models through LangChain. First install and start Ollama, then pull a multimodal/vision model:

```bash
ollama pull qwen2.5vl:7b
```

If Ollama is not already running, start the server in another terminal:

```bash
ollama serve
```

If you see `bind: address already in use`, Ollama is already running and you can continue.

Configure `config.yml` like this:

```yaml
dataset: TUEV
model: qwen2.5vl:7b
provider: ollama

model-kwargs:
  base_url: http://localhost:11434
  temperature: 0

structured-output:
  method: json_schema

limit: 10
```

Then run:

```bash
python3 scripts/eval.py
```

Because `scripts/eval.py` sends spectrogram images, use a vision-capable Ollama model such as `qwen2.5vl:7b`, `llama3.2-vision`, or `llava`. Text-only models will not work with the current image input format.


## Additional Resources

Each dataset comes with a `generate-annotated.py` file which generates an equivalent set of `.png` images, except that the artifacts are visually annotated
with a colour. This may be useful for manual review of findings.

## Implementation Details

### config.yml

This config script outlines some fundamental parameters that can be changed to influence how evaluation runs.

#### Top level Parameters

 - `dataset` - Select the dataset to run evaluation on. One of: `TUEP`, `TUAB`, `TUEV`, `TUAR`, `TUSZ`, `TUSL`
 - `model` - Which model to run evaluation through.
 - `provider` - The provider of the above model. Find the list here: https://docs.langchain.com/oss/python/integrations/providers/all_providers
 - `model-kwargs` - Optional keyword arguments forwarded to LangChain model initialization, such as Ollama's `base_url` or `temperature`.
 - `structured-output` - Optional keyword arguments forwarded to LangChain's `.with_structured_output()`, such as `method: json_schema` for Ollama.

#### Dataset-specific Parameters 

These parameters exist under each specific dataset heading.

 - `prompt` - Which prompt to give the model to assist with evaluation.
 - `data-directory` - In which subdirectory the relevant preprocessed data can be found.

### scripts/structure.py 

A script that outlines the relevant output structure for the VLM, customized for each dataset. Specifically, it provides a `text_rationale` attribute
along with a boolean attribute for each possible artifact. Is used by `scripts/eval.py`

### scripts/eval.py

The program runs with the following high-level workflow:
 1. API secrets are loaded from the `.env` file, and config parameters are loaded from `config.yml`
 2. The output structure is fetched from `scripts/structure.py`
 3. The model is initialized, `.csv` file is created, and prompt is loaded.
 4. A list of all image paths to evaluate is loaded into the program.
 5. A batch is built, which combines spectrogram images with the prompts.
 6. The batch is sent to the model for evaluation. This step takes the longest, and data is recorded as the batch is processed.
 7. The results of the model are saved to the `results` directory

## Running with a GPU

Simply update the `slurm` section in `config.yml` to select GPU, memory, and time allocations, then run `python run.py`