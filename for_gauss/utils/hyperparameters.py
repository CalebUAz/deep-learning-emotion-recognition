from typing_extensions import Self


class Config(Self):
    def __init__(self):
        batch_size = 50
        emotion_categories = 5

        epochs = 70
        eeg_input_dim = 310
        eye_input_dim = 33
        output_dim = 12
        learning_rate = 5 * 1e-4
        batch_size = 50