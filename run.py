"""
This script loads the trained model created in network.py, loads the vocabulary that is created in data_and_tokens.py to
generate a sequence of tokens given a starting sequence determined by the user. Then, using the library mido
(https://github.com/mido/mido), turns the generated sequence into a MIDI file. After that, turns the midi file into a
wav file called "play_me.wav" for easier playability.
"""

import torch
import torch.nn as nn
import numpy as np
import mido
from mido import Message, MidiFile, MidiTrack
from scipy.io.wavfile import write


# The class for the LSTM model:
class MusicLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)  # Maps tokens to vectors
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True)  # LSTM layer
        self.fc = nn.Linear(hidden_dim, vocab_size)  # Maps hidden states to vocabulary size

    def forward(self, x):
        embedded = self.embedding(x)  # Embedding layer
        out, (h, c) = self.lstm(embedded)  # LSTM layer
        logits = self.fc(out)  # Fully connected layer
        return logits

#Loading the trained model:
model = torch.load('trained_model.pth')

# Loading the vocabulary to be used in converting tokens into numericals and vice versa
vocab_list = torch.load('vocab_list.pth')
token2idx = {token: i for i, token in enumerate(vocab_list)} #Converting word to numerical index
idx2token = {i: token for i, token in enumerate(vocab_list)} #Numerical index to word list, to be used in generating music

# Making sure that the model is using the right device:
if torch.backends.mps.is_available():
    device = torch.device("mps") #mps for macbook
elif torch.cuda.is_available():
    device = torch.device("cuda") #cuda for google colab
else:
    device = torch.device("cpu")
print("Using device:", device)


""" Generating Music"""

# Starting sequence. This is given by the user. 'NOTE_ON_x' means to start playing a note where x is a number that corresponds to
# a note. 'NOTE_OFF_x' stops the note from playing. 'TIME_SHIFT_1' is 16th of a note wait between actions. For example
# ['NOTE_ON_60'], ['TIME_SHIFT_1'], ['NOTE_OFF_60'] corresponds to a C (do) note playing for 16th of a note.
#tokens = ['NOTE_ON_60', 'TIME_SHIFT_1', 'NOTE_OFF_60'] #starts with a C note that is playing for 16th of a note
#tokens = ['NOTE_ON_60', 'TIME_SHIFT_1', 'NOTE_ON_63', 'TIME_SHIFT_1', 'NOTE_ON_67'] #starts with a C major chord
#tokens = ["NOTE_ON_67", "NOTE_ON_71", "NOTE_ON_74", "NOTE_ON_77"] #G7 chord which is fairly more complex

#For a better demo, the following code asks the user to select a key and starts the sequence on that key:
# Defining the starting sequences for each key
key_to_sequence = {
    "A": ['NOTE_ON_69', 'TIME_SHIFT_1', 'NOTE_OFF_69'],  # A
    "B": ['NOTE_ON_71', 'TIME_SHIFT_1', 'NOTE_OFF_71'],  # B
    "C": ['NOTE_ON_60', 'TIME_SHIFT_1', 'NOTE_OFF_60'],  # C (Middle C)
    "D": ['NOTE_ON_62', 'TIME_SHIFT_1', 'NOTE_OFF_62'],  # D
    "E": ['NOTE_ON_64', 'TIME_SHIFT_1', 'NOTE_OFF_64'],  # E
    "F": ['NOTE_ON_65', 'TIME_SHIFT_1', 'NOTE_OFF_65'],  # F
    "G": ['NOTE_ON_67', 'TIME_SHIFT_1', 'NOTE_OFF_67'],  # G
}

# Ask the user for a key selection
print("Please select a key to generate music (Available options: A, B, C, D, E, F, G):")
selected_key = input("Enter your key: ").strip().upper()

# Validate the user's input
if selected_key not in key_to_sequence:
    print(f"Invalid key: {selected_key}. Please restart and choose a valid key.")
else:
    # Generate the starting sequence based on the selected key
    tokens = key_to_sequence[selected_key]
    print(f"Selected key: {selected_key}")

#Converting tokens to numerical index
def tokens_to_indices(tokens, token2idx):
    return [token2idx[token] for token in tokens]
start_sequence = tokens_to_indices(tokens, token2idx)

# Convert to tensor and move to device
generated_sequence = torch.tensor(start_sequence, dtype=torch.long).unsqueeze(0).to(device)

model.eval() #eval mode so that the model uses correct settings
with torch.no_grad():
    for _ in range(50):  # Generate 50 tokens
        logits = model(generated_sequence)
        next_token = torch.argmax(logits[:, -1, :], dim=-1)  # Predict next token
        generated_sequence = torch.cat((generated_sequence, next_token.unsqueeze(0)), dim=1) # Appends the generated token into the end of tensor

# Decode the generated token numbers back into a format readable for MIDI. (['NOTE_ON_x', 'NOTE_OFF_x', 'TIME_SHIFT_x'])
decoded_sequence = [idx2token[idx] for idx in generated_sequence[0].tolist()]
print("Generated Sequence:", decoded_sequence)

#Function for generating the midi file from given sequence:
def generate_midi_from_sequence(sequence, output_file="generated.mid"):
    """
    Converts a sequence of tokens (e.g., ['NOTE_ON_x', 'NOTE_OFF_x', 'TIME_SHIFT_x']) into a MIDI file and saves it.
    """

    # Create a new MIDI file and track
    midi = MidiFile()
    track = MidiTrack()
    midi.tracks.append(track)

    # Current time (in ticks)
    current_time = 0

    for token in sequence:
        if token.startswith("NOTE_ON_"):
            # Extract pitch and add NOTE_ON message
            pitch = int(token.split("_")[2])  # Extracting the pitch value
            track.append(Message('note_on', note=pitch, velocity=64, time=current_time))
            current_time = 0  # Reset time since we append it immediately
        elif token.startswith("NOTE_OFF_"):
            # Extract pitch and add NOTE_OFF message
            pitch = int(token.split("_")[2])  # Extracting the pitch value
            track.append(Message('note_off', note=pitch, velocity=64, time=current_time))
            current_time = 0  # Reset time
        elif token.startswith("TIME_SHIFT_"):
            # Extract time shift (assuming 16th note resolution)
            time_shift = int(token.split("_")[2])  # Correct segment for time shift
            current_time += time_shift * 240  # Adjust multiplier based on ticks_per_beat.
                                              # The multiplier affects the speed of the generated audio. (lower number = faster audio)

    # Save the MIDI file
    midi.save(output_file)
    print(f"MIDI file saved as: {output_file}")

generate_midi_from_sequence(decoded_sequence) #Generating the MIDI file

# Converting the midi file into a wav file so that a simple media player can play the sound. Listening to the MIDI file
# directly from a MIDI player would result in better audio quality but for demo purposes, this generated wav file should
# be good enough. I did not want to import any extra libraries for this purpose.
def midi_to_sine_wave(midi_path, output_wav="output.wav", sample_rate=44100):
    """
    Converts a MIDI file to a sine wave audio and save it as a wav file.
    """
    # Load MIDI
    mid = mido.MidiFile(midi_path)

    # Prepare an empty audio buffer (float32) for the entire duration
    audio_buffer = np.zeros(int((mid.length + 1) * sample_rate), dtype=np.float32)  # Add extra time to buffer

    current_time = 0.0  # Tracks 'seconds' in the MIDI timeline
    for msg in mid:
        current_time += msg.time
        # Handle 'note_on' messages
        if msg.type == 'note_on' and msg.velocity > 0:
            freq = 440.0 * (2.0 ** ((msg.note - 69) / 12.0))  # Convert MIDI note to frequency
            duration = 0.5  # Default duration
            start_idx = int(current_time * sample_rate)
            end_idx = start_idx + int(duration * sample_rate)
            if end_idx > len(audio_buffer):
                end_idx = len(audio_buffer)
            t = np.linspace(0, duration, end_idx - start_idx, endpoint=False)

            # Generate a sine wave and apply an envelope for smooth decay
            wave = 0.3 * np.sin(2 * np.pi * freq * t)
            envelope = np.linspace(1, 0, len(t))  # Linear fade-out envelope
            wave = wave * envelope

            # Add the wave to the buffer
            audio_buffer[start_idx:end_idx] += wave

    # Normalize the audio to prevent clipping
    if np.max(np.abs(audio_buffer)) > 0:
        audio_buffer /= np.max(np.abs(audio_buffer))

    # Convert to int16 for WAV
    audio_data = (audio_buffer * 32767).astype(np.int16)

    # Save WAV
    write(output_wav, sample_rate, audio_data)
    print(f"WAV file saved as: {output_wav}")
    return output_wav

# Generating the wav file.
midi_file = "generated.mid"
output_file = "play_me.wav"
midi_to_sine_wave(midi_file, output_file)
