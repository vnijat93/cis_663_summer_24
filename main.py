import os
from process_audio import ProcessAudio
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# File paths for exports and imports of data
current_dir = os.path.dirname(os.path.abspath(__file__)).strip('voice_recognition')
common_voice_data = os.path.join(current_dir, 'Sources and References/Data Sets/cv-corpus-17.0-2024-03-15/en/validated.tsv')
deepfake_data = os.path.join(current_dir, 'Sources and References/Data Sets/release_in_the_wild/meta.csv')
deepfake_data_dir = os.path.join(current_dir, 'Sources and References/Data Sets/release_in_the_wild')
common_voice_data_dir = os.path.join(current_dir, 'Sources and References/Data Sets/cv-corpus-17.0-2024-03-15/en/clips')
# export_csv = os.path.join(current_dir, 'voice_recognition/barack_obama_voice_export.csv')

# Init audio processing class
audio_data = ProcessAudio(mfcc_coefficients=13)

# These functions grab audio data from the csv and then calculate features and export feature data to a csv
# common_voice_files = audio_data.get_common_voice_audio_file_data(common_voice_data, condense_size=50000)
# audio_data.load_and_export_common_voice_data(data=common_voice_files, audio_file_directory=common_voice_data_dir, output_location=export_csv)


# deepfake_audio_files = audio_data.get_deepfake_audio_file_data(deepfake_data, condense_size=100)
# audio_data.load_and_export_voice_data(data=deepfake_audio_files, audio_file_directory=deepfake_data, output_location=export_csv)

deepfake_audio_files, speaker_name = audio_data.get_in_the_wild_audio_file_data(deepfake_data, condense_size=1000, input_select=0)
output_location = 'voice_recognition/' + speaker_name + '_deepfake_voice_export_filtered.csv'
export_csv = os.path.join(current_dir, output_location)
audio_data.load_and_export_in_the_wild_data(data=deepfake_audio_files, audio_file_directory=deepfake_data_dir, output_location=export_csv)

# Directory structure where each subdirectory corresponds to a class (e.g., /data/class1 or /data/class2
# class_names = ['class1', 'class2', ...]




# X = []
# Y = []
# for i, class_name in enumerate(class_names):
#     class_dir = os.path.join(data_dir, class_name)
#     for filename in os.listdir(class_dir):
#         if filename.endswith('.wav'):
#             file_path = os.path.join(class_dir, filename)
#             features = load_and_preprocess_audio(file_path)
#             X.append(features)
#             Y.append(i)

# X = np.array(X)
# Y = np.array(Y)

# # Step 2: Split data into train and test sets
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# # Step 3: Model training
# model = RandomForestClassifier(n_estimators=100)
# model.fit(X_train, Y_train)

# # Step 4: Validation and testing
# accuracy = model.score(X_test, Y_test)
# print("Accuracy:", accuracy)

# Step 5: Integration with Speech Recognition
# Here, you would implement the mechanism to record audio from the user, preprocess it and
# then use the trained model to classify it into one of the classes.