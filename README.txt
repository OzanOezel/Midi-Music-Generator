The files within this folder are:

- Presentation.ppsx : The presentation of the project.

- run.py : Asks for the user to enter a musical note (A = la, B = si, C = do... etc.), then outputs a generated audio named "playme.wav" in that key.

- data_and_tokens.py : The code where the vocabulary and dataset is generated from MIDI input files.

- dataset.pth : The generated data.

- vocab_list.pty : The generated vocabulary.

- network.py : The code where the model is trained.

- Mido : This folder contains the source code of the python library "mido". It is essential to run the code run.py. All credit goes to https://github.com/mido/mido/graphs/contributors

-Ver0 : This folder contains the version of the code where hidden states are not passed through batches. This is only for showing. Not used in the final version.

