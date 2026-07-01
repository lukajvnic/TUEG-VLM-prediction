# Quickbook for VLM fine-tuning

## Organizing the Data

In order to fine-tune a VLM, the data must first be organized into examples. Each example contains:
 1) The EEG spectrogram image (`.png`)
 2) The prompt (string provided in tandem with the image)
 3) The target text (the neurologist report associated with the image)

One important thing to watch out for here is including the same patient in train and test splits, because the model could learn patterns
specific to that patient rather than more general patterns.

## Choosing a VLM and fine-tuning method

There should exist parameters in a `config.yml` file like `model` and `method`, where a user could specify a model like `LLaVA` or `Qwen2-VL`, etc.
The fine-tuning method will be something like `LoRA` or `QLoRA`.

To load an existing pretrained open-source model, in python, we can use the `transformers` library. Similarly, we can use the `bitsandbytes` library to implement
`LoRA` and `QLoRA`, so a large model can fit on less (or even one) GPU.

### Which fine-tuning method to use?

 1) `LoRA` - Ignore most weights, only train a relevant subsection.
 2) `QLoRA` - Like `LoRA`, but reorganizes the model to fit in 4-bits instead of 16, uses less memory/gpus, and has a *slightly* worse performance.
 3) Neither - Very computationally expensive, trains all the weights.

## Formatting the data for the VLM

VLMs tend to have their own input structure. We will likely need another `structure.py` type program to format the data specifically for a given VLM.

## Supervised fine-tuning

The python `trl` library comes with the `SFTTrainer` class, which can be used to fine-tune VLM models.
It's as easy as `SFTTrainer(model, dataset, output_dir, epochs, batch_size).train()`

This library does the whole forward pass → loss → backprop → optimizer loop for us.

