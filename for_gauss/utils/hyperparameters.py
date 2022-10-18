class Config():
    def __init__(self):
        self.batch_size = 50
        self.emotion_categories = 5

        self.epochs = 70
        self.eeg_input_dim = 310
        self.eye_input_dim = 33
        self.output_dim = 12
        self.learning_rate = 5 * 1e-4
        self.batch_size = 50