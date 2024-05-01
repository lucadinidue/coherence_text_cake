Code for the experiments of <img src="https://github.com/lucadinidue/coherence/assets/28627385/6b7db2e7-7526-404b-80e1-2893f5bd40e2" alt="drawing" width="30"/> **Text-Cake**: Challenging Language Models on Local Text Coherence 

The **dataset** directory contains the dataset used for the experiments. It contains one directory for each text domain, each containing the training and test files for each language. 
The dataset files are `\tab` separated files with three fields:
  - passage_id: the id of the text passage
  - text: the text of the passage
  - label: the perturbation applied to the passage

The **scripts** directory contains the code used to perform the fine-tunings and the probing experiments.
