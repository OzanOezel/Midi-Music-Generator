"""
This script takes MIDI files as input, tokenizes the files and creates a vocabulary.
Then saves the tokenized data and the vocabulary.
"""

""""------------------------- Importing Required Libraries -------------------------"""
import torch
import mido #This library is used to process MIDI files
import numpy as np
import math
import os
import glob


"""------------------------- Acquiring the Dataset -------------------------"""
!git clone https://github.com/jukedeck/nottingham-dataset.git #This is the cleaned up version of the Notthingham Dataset


"""------------------------- Preprocessing the MIDI Files -------------------------"""

#A class that takes the directory of a MIDI file as input and outputs the stucture of the song as a tokens.
def midi_to_token_sequence(midi_path):
    """
    This function turns a single MIDI file into a token seqeuence such as [TIME_SHIFT_1, NOTE_ON_60, NOTE_OFF_60, ...].
    """

    mid = mido.MidiFile(midi_path) # Load the MIDI file


    ticks_per_step = mid.ticks_per_beat / 4 # Compute ticks for 16th of a note (1/4 of a beat)
                                            # ticks_per_beat is the resolution of the file (Could vary between MIDI files)

    #Quantizing the MIDI files
    events = []
    current_tick = 0

    for msg in mid:

        current_tick += msg.time * mid.ticks_per_beat # msg.time is the time between events (beats)
                                                      # beats*ticks/beats = ticks

        if msg.type == 'note_on' and msg.velocity > 0: #If the note is on
            pitch = msg.note #The note that is playing
            step = int(math.floor(current_tick / ticks_per_step)) #converts ticks into 16th note steps
            events.append((step, 'NOTE_ON', pitch)) #indicating that the note starts at that time
        elif (msg.type == 'note_off') or (msg.type == 'note_on' and msg.velocity == 0): #If the note is off
            pitch = msg.note #The note that is stopping
            step = int(math.floor(current_tick / ticks_per_step)) #converts ticks into 16th note steps
            events.append((step, 'NOTE_OFF', pitch)) #indicating that the note ends at that time

    # Sort events by the time step
    events.sort(key=lambda x: x[0])

    #Turning the dataset into a format that is easier to turn into a vocabulary.
    #Converting (step, event_type, pitch) into a token sequence consisting of:
    #TIME_SHIFT_1 (move time forward by 1 “unit”)
    #NOTE_ON_x (start playing pitch x (x = 60 equals Middle C note on a piano))
    #NOTE_OFF_x (stop playing pitch x)

    token_sequence = []
    prev_step = 0
    for (step, etype, pitch) in events:
        time_diff = step - prev_step
        # Insert TIME_SHIFT tokens
        while time_diff > 0:
            token_sequence.append("TIME_SHIFT_1")
            time_diff -= 1
        # Insert NOTE_ON_pitch or NOTE_OFF_pitch tokens
        if etype == 'NOTE_ON':
            token_sequence.append(f"NOTE_ON_{pitch}")
        else:
            token_sequence.append(f"NOTE_OFF_{pitch}")
        prev_step = step

    return token_sequence

##Saving the paths for all the files that are in MIDI format to a list, to be used in the function "Midi_to_token_sequence":
midi_dir = "nottingham-dataset/MIDI"
midi_paths = glob.glob(os.path.join(midi_dir, "*.mid"))

#Downsampling the dataset into 100 files for easier experimentation:
midi_paths_subset = midi_paths[:100]
#midi_paths_subset = midi_paths

#Parsing all midi files into token sequences using the function created before.(Creating the tokens to be used in the dataset):
all_token_sequences = []
for path in midi_paths_subset:
    tokens = midi_to_token_sequence(path)
    all_token_sequences.append(tokens)
print(f"{len(all_token_sequences)} MIDI files into token sequences.")

#Creating a vocabulary of all available tokens
token_set = set()
for seq in all_token_sequences:
    token_set.update(seq)

vocab_list = sorted(list(token_set)) #Sorts the vocabulary
token2idx = {token: i for i, token in enumerate(vocab_list)} #Converting word to numerical index
idx2token = {i: token for i, token in enumerate(vocab_list)} #Numerical index to word list, to be used in generating music

vocab_size = len(vocab_list)


# A dataset of 50 + 50 tokens
seq_length=50 #length of the context
samples = []

for seq in all_token_sequences:
    # Converting tokens to integer IDs
    int_seq = [token2idx[t] for t in seq]

    # Creating sliding windows of [50] ---> [50] (x ---> y) (x ----> x+1)
    for i in range(len(int_seq) - seq_length):
        x = int_seq[i : i + seq_length]          # input
        y = int_seq[i + 1 : i + 1 + seq_length]  # target (shifted by 1)
        samples.append((x, y))

#Converting the data into Torch Tensor:
def convert_samples_to_tensors(samples):
    """
    This function converts a list of x, y into Tensor objects.
    """

    X_list = []
    Y_list = []
    for x, y in samples:
        X_list.append(torch.tensor(x, dtype=torch.long))
        Y_list.append(torch.tensor(y, dtype=torch.long))

    # Stacking all samples into a single tensor object
    X = torch.stack(X_list)
    Y = torch.stack(Y_list)

    return X, Y

#Creating the final dataset to be used in the network
X, Y = convert_samples_to_tensors(samples)

from torch.utils.data import TensorDataset

# Creating a TensorDataset to later use in Dataloader
dataset = TensorDataset(X, Y)

# Saving the dataset
torch.save(dataset, 'dataset.pth')
#Saving the vocabulary
torch.save(vocab_list, 'vocab_list.pth')