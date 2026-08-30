# vision

This file is what i want you to implement. It's based on the high-level vision of how I want this redesign of the TUEG-VLM-prediction project to work. I want you to reference the ~/dev/TUEG-VLM-predicition repo for code and implementation. A lot of the code already exists.

## writing and organizing software

I want this code to be clean and modular. Split everything down into functions. Dont write comments, unless its extremely necessary, then put one inline comment that's short and to the point. Break everything into clean and simple functions. Use a library to minimize code if it exists. Don't write smoketests and validators, just the pipeline. Don't use any sort of file arguments. If there is something to be customized, make it a constant at the top of the file. But use this sparsely. It is rarely needed to customize something, there really should only be one way for a file to be run. Keep code as clean and minimal as possible. Think deeply about how to structure a file before writing it, and consider multiple approaches before settling on one. Plan ahead to make the code as reusable as possible. Don't repeat yourself in code, write a function and reuse it. I listed out the files here. we are going to go step by step, but I really don't think we need more files than this, except maybe sbatch files but even that might not be necessary, it could run through python.

## pipeline

The vision for this version is that, essentially, this project is just a pipeline. You can get more context on the project from the reference repo. But basically it's just a pipeline. First, you download the data. Then you preprocess it, Then you generate ground truth rationales. Then you eval it against all the models. Then you judge the rationales. (This is all we need for now, finetuning comes later). But basically at each of these steps, some images on some models fail. Maybe the model couldn't read it correctly, or the model produced garbage results. As each image goes through the pipeline, some images drop out and can't be run on the next step in the pipeline. The idea here is that everything is centralized. There are only a fixed amount of images and models to be run. So we will keep track of each image in pipeline.csv. And each image will have N copies, one for each model (I think 45). Each row will have a simple list of booleans, whether or not it passed a given stage in the pipeline. This way we can easily track what is completed and what still needs to be done. The idea is that now we no longer depend on a messy config.yml that we need to comment each time we run. We decide what to run based on this pipeline.csv, and we run what has failed. Which we know based on the boolean.

## data organization

Each dataset has its own directory in datasets. Each of these has a test and train directory for images, and an `eval-baseline.csv`, `judge-baseline.csv`, and `labels.csv` files.

`labels.csv` will contain columns `path,<one column here for each artifact/classification>,ground_truth_rationale`
`eval-baseline.csv` will contain columns `path,model,<one column here for each artifact/classification>,rationale`
`judge-baseline.csv` will contain columns `path,model,correct_predictions,correct_rationale,correct_rationale_reason`
 - here correct_predictions will just compare if it accurately predicted which artifacts appear or correctly classified it
 - correct_rationale will be by the judge LLM deciding if the rationale is saying the same thing as the ground truth rationale