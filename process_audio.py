import librosa
import os
import numpy as np
import pandas as pd
import scipy.signal as signal


class ProcessAudio:
    def __init__(self, mfcc_coefficients):
        self.mfcc_coefficients = mfcc_coefficients
        self.audio_labels = [
            "chroma",
            "centroid",
            "bandwidth",
            "zcr",
            "energy",
            "contrast",
            "rolloff",
            "pitch",
        ]
        for coefficient in range(1, (self.mfcc_coefficients + 1)):
            self.audio_labels.append("mfcc_" + str(coefficient))
        self.audio_labels.append("label")

    def _process_audio(self, file_path, label):
        data = {}
        for feature in self.audio_labels:
            data[feature] = None
        # Load audio file
        audio, sr = librosa.load(
            file_path, sr=None
        )  # sr=None to retain original sampling rate

        # Define the frequency range for human voice
        low_cutoff = 85.0  # Hz
        high_cutoff = 255.0  # Hz

        # Design a bandpass filter
        nyquist = 0.5 * sr
        low = low_cutoff / nyquist
        high = high_cutoff / nyquist

        # Using Butterworth filter
        b, a = signal.butter(4, [low, high], btype="band")

        # Apply the bandpass filter
        audio = signal.filtfilt(b, a, audio)

        # Compute Mel-Frequency Cepstral Coefficients (MFCCs)
        mfccs = librosa.feature.mfcc(
            y=audio, sr=sr, n_mfcc=self.mfcc_coefficients
        )  # Choose desired number of coefficients
        # Flatten MFCCs into a feature vector
        mfcc_flat = np.mean(mfccs.T, axis=0)
        for index, mfcc in enumerate(mfcc_flat):
            key = "mfcc_" + str(index + 1)
            data[key] = [float(mfcc)]

        # Compute Chroma features
        data["chroma"] = [
            float(np.mean(np.mean(librosa.feature.chroma_stft(y=audio, sr=sr), axis=1)))
        ]  # Consolidates chroma into one number

        # Compute Spectral Centroid
        data["centroid"] = [np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))]

        # Compute Spectral Bandwidth
        data["bandwidth"] = [
            np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sr))
        ]

        # Compute Zero-Crossing Rate
        data["zcr"] = [np.mean(librosa.feature.zero_crossing_rate(audio))]

        # Compute Energy
        data["energy"] = [float(np.sum(np.square(audio)))]

        # Compute Spectral Contrast
        data["contrast"] = [np.mean(librosa.feature.spectral_contrast(y=audio, sr=sr))]

        # Compute Spectral Roll-off
        data["rolloff"] = [np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sr))]

        # Compute Pitch (using librosa's YIN algorithm)
        data["pitch"] = [
            np.mean(
                librosa.yin(
                    y=audio,
                    fmin=librosa.note_to_hz("C2"),
                    fmax=librosa.note_to_hz("C7"),
                )
            )
        ]  # Fmin and Fmax should cover most human voice frequencies

        data["label"] = label
        return pd.DataFrame.from_dict(data)

    def load_and_export_common_voice_data(
        self, data, audio_file_directory, output_location=None
    ):
        if output_location is None:
            output_location = audio_file_directory

        # Get all audio files in given directory and export features to one solid csv
        audio_df = pd.DataFrame(columns=self.audio_labels)

        for index in data.index:
            audio_path = os.path.join(audio_file_directory, data["path"][index])
            if data["gender"][index] == "male_masculine":
                label = "male"
            elif data["gender"][index] == "female_feminine":
                label = "female"

            features = self._process_audio(audio_path, label)
            audio_df = pd.concat([audio_df, features], ignore_index=True, sort=False)

        audio_df.to_csv(output_location, index=False)

    def get_common_voice_audio_file_data(self, input_file, condense_size=1000):
        if input_file.endswith(".tsv"):
            file_data = pd.read_csv(input_file, sep="\t")
        else:
            file_data = pd.read_csv(input_file)

        male_data = file_data.loc[file_data["gender"] == "male_masculine"]
        female_data = file_data.loc[file_data["gender"] == "female_feminine"]
        # Shuffle Rows
        male_data = male_data.sample(frac=1)
        female_data = female_data.sample(frac=1)

        # Simplify to condensed table
        male_data = male_data[["path", "gender"]].head(condense_size // 2)
        female_data = female_data[["path", "gender"]].head(condense_size // 2)

        return pd.concat([male_data, female_data], ignore_index=True, sort=False)

    def get_deepfake_audio_file_data(self, input_directory, condense_size=1000):
        real_file_list = os.listdir(os.path.join(input_directory, "REAL"))
        fake_file_list = os.listdir(os.path.join(input_directory, "FAKE"))

        real_data = pd.DataFrame(columns=["path", "label"])
        fake_data = pd.DataFrame(columns=["path", "label"])

        for item in real_file_list:
            file_path = os.path.join("REAL", item)
            real_data = real_data._append(
                {"path": file_path, "label": "real"}, ignore_index=True
            )

        for item in fake_file_list:
            file_path = os.path.join("FAKE", item)
            fake_data = fake_data._append(
                {"path": file_path, "label": "fake"}, ignore_index=True
            )

        real_data = real_data.sample(frac=1)
        fake_data = fake_data.sample(frac=1)

        real_data = real_data.head(condense_size // 2)
        fake_data = fake_data.head(condense_size // 2)

        return pd.concat([real_data, fake_data], ignore_index=True, sort=False)

    def load_and_export_voice_data(
        self, data, audio_file_directory, output_location=None
    ):
        if output_location is None:
            output_location = audio_file_directory

        # Get all audio files in given directory and export features to one solid csv
        audio_df = pd.DataFrame(columns=self.audio_labels)

        for index in data.index:
            audio_path = os.path.join(audio_file_directory, data["path"][index])
            label = data["label"][index]

            features = self._process_audio(audio_path, label)
            audio_df = pd.concat([audio_df, features], ignore_index=True, sort=False)

        audio_df.to_csv(output_location, index=False)

    def get_in_the_wild_audio_file_data(
        self, input_file, condense_size=1000, input_select=0
    ):
        if input_file.endswith(".tsv"):
            file_data = pd.read_csv(input_file, sep="\t")
        else:
            file_data = pd.read_csv(input_file)

        unique_names = file_data[
            "speaker"
        ].value_counts()  # Grabs the unique celebrity names and how many audio files there are per speaker
        speaker_name = unique_names.index[input_select]
        speaker_name = speaker_name.replace(" ", "_").lower()

        # Using the most common celebrity voice in the dataset
        real_data = file_data.loc[
            (file_data["label"] == "bona-fide")
            & (file_data["speaker"] == unique_names.index[input_select])
        ]
        fake_data = file_data.loc[
            (file_data["label"] == "spoof")
            & (file_data["speaker"] == unique_names.index[input_select])
        ]
        # Shuffle Rows
        real_data = real_data.sample(frac=1)
        fake_data = fake_data.sample(frac=1)

        # Simplify to condensed table
        real_data = real_data[["file", "label"]].head(condense_size // 2)
        fake_data = fake_data[["file", "label"]].head(condense_size // 2)

        return (
            pd.concat([real_data, fake_data], ignore_index=True, sort=False),
            speaker_name,
        )

    def load_and_export_in_the_wild_data(
        self, data, audio_file_directory, output_location=None
    ):
        if output_location is None:
            output_location = audio_file_directory

        # Get all audio files in given directory and export features to one solid csv
        audio_df = pd.DataFrame(columns=self.audio_labels)

        for index in data.index:
            audio_path = os.path.join(audio_file_directory, data["file"][index])

            features = self._process_audio(audio_path, data["label"][index])
            audio_df = pd.concat([audio_df, features], ignore_index=True, sort=False)

        audio_df.to_csv(output_location, index=False)
