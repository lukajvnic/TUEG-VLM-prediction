Write me a professional looking report in latex on the state of the project today. I want you to use the outline below as structure but feel free to add more detail that you think is necessary. Reference the codebase to ensure what you’re writing is correct. This is meant to be given to my PI to understand what I’ve done so far and how it works technically and architecturally.

* Ollama setup  
1) ollama-install.py in $SCRATCH/ollama  
* Hugging face setup  
1) hf-install.py  
* Dataset generation  
1) Download each individual dataset via download.sh  
2) Provide structure of datasets/ folder  
3) Run generate-all.sh script which runs respective generate.py scripts  
   1) generate.py does a 70/30 split on the dataset ensuring no patient overlap  
4) Some recordings are over 30 minutes long, would not be feasible to generate 20 second windows of each  
   1) We still need a balanced dataset, so we record a variety of images, some in empty parts and some in active parts of the recording.  
   2) Go into technical implementation, MNE, ensuring they fit in their rows  
5) Additionally generates labels.csv for each image, with image name, booleans of artifacts, and whether its test or train  
* Ground truth rationale generation  
1) Fetch ground truth values and image path from labels.csv in the dataset  
2) Use gemma:22b or something to generate rationales and save in rationales.csv  
* Eval  
  * For each test image identify which artifacts appear and generate a rationale for each model on each dataset  
  * Similar loop to rationale generation.  
  * Backup and retry and logging, like infrastructure to rerun failed runs etc, and to check in on runs as its going  
  * Config.yml for cleanliness