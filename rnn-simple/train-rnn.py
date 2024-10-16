import numpy as np
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense, Embedding
from keras.preprocessing.sequence import pad_sequences
# from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.text import Tokenizer

from dataset import load_data  # Import từ dataset.py

# Load dữ liệu từ file dataset.py
sentences, labels = load_data()

# Chuẩn hóa dữ liệu
tokenizer = Tokenizer(num_words=10000)  # Lấy 10,000 từ phổ biến nhất
tokenizer.fit_on_texts(sentences)
X = tokenizer.texts_to_sequences(sentences)
X = pad_sequences(X, maxlen=10)  # Đảm bảo các chuỗi có độ dài bằng 10

# Mô hình RNN
model = Sequential()
model.add(Embedding(input_dim=10000, output_dim=64, input_length=10))  # Tầng Embedding
model.add(SimpleRNN(64, activation="relu"))  # Tầng SimpleRNN
model.add(Dense(1, activation="sigmoid"))  # Tầng đầu ra cho phân loại nhị phân

# Biên dịch mô hình
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Huấn luyện mô hình
model.fit(X, np.array(labels), epochs=10)

# Dự đoán trên dữ liệu mới
new_sentence = ["This is a bad product"]
new_X = tokenizer.texts_to_sequences(new_sentence)
new_X = pad_sequences(new_X, maxlen=10)
prediction = model.predict(new_X)

# In kết quả
print("Dự đoán cảm xúc: ", "Tích cực" if prediction > 0.5 else "Tiêu cực")
