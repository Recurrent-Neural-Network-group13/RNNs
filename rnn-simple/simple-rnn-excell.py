# import pandas as pd
# from keras.models import Sequential
# from keras.layers import RNN, Dense, Embedding
# from keras.layers import SimpleRNNCell  # Sử dụng SimpleRNNCell cho RNN
# from tensorflow.keras.preprocessing.text import Tokenizer
# from keras.preprocessing.sequence import pad_sequences

# # Bước 1: Đọc dữ liệu từ file Excel
# dataset = pd.read_excel('D:\\Documents For Study\\Source Code\\RNN_ver2\\RNNs\\rnn-simple\\sentiment_dataset.xlsx')
# texts = dataset['Text'].values
# labels = dataset['Sentiment'].values

# # Bước 2: Tokenization và Padding
# tokenizer = Tokenizer(num_words=5000)  # Sử dụng 5000 từ thông dụng nhất
# tokenizer.fit_on_texts(texts)
# sequences = tokenizer.texts_to_sequences(texts)
# maxlen = 100  # Độ dài tối đa của chuỗi, có thể thay đổi
# X = pad_sequences(sequences, maxlen=maxlen)
# y = labels

# # Bước 3: Xây dựng mô hình RNN
# model = Sequential()
# model.add(Embedding(input_dim=5000, output_dim=32, input_length=maxlen))

# # Sử dụng RNN với SimpleRNNCell
# rnn_cell = SimpleRNNCell(32)  # Tạo cell RNN với 32 đơn vị
# model.add(RNN(rnn_cell))  # Thay thế SimpleRNN bằng RNN

# model.add(Dense(1, activation='sigmoid'))

# # Bước 4: Compile mô hình
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# # Bước 5: Huấn luyện mô hình
# model.fit(X, y, epochs=50, batch_size=32, validation_split=0.2)

# # Bước 6: Dự đoán
# sample_text = ["The efficient and quick service was great."]
# sample_seq = tokenizer.texts_to_sequences(sample_text)
# sample_pad = pad_sequences(sample_seq, maxlen=maxlen)
# prediction = model.predict(sample_pad)
# print(f"Prediction:", "Good" if prediction > 0.5 else "Bad")

import pandas as pd
from keras.models import Sequential
from keras.layers import RNN, Dense, Embedding
from keras.layers import SimpleRNNCell  # Sử dụng SimpleRNNCell cho RNN
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# Bước 1: Đọc dữ liệu từ file Excel
dataset = pd.read_excel('D:\\Documents For Study\\Source Code\\RNN_ver2\\RNNs\\rnn-simple\\sentiment_dataset.xlsx')
texts = dataset['Text'].values
labels = dataset['Sentiment'].values

# Kiểm tra và chuyển đổi các giá trị không phải chuỗi trong cột 'Text' thành chuỗi
texts = [str(text) if isinstance(text, (str, float)) else '' for text in texts]

# Bước 2: Tokenization và Padding
tokenizer = Tokenizer(num_words=5000)  # Sử dụng 5000 từ thông dụng nhất
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
maxlen = 100  # Độ dài tối đa của chuỗi, có thể thay đổi
X = pad_sequences(sequences, maxlen=maxlen)
y = labels

# Bước 3: Xây dựng mô hình RNN
model = Sequential()
model.add(Embedding(input_dim=5000, output_dim=32, input_length=maxlen))

# Sử dụng RNN với SimpleRNNCell
rnn_cell = SimpleRNNCell(32)  # Tạo cell RNN với 32 đơn vị
model.add(RNN(rnn_cell))  # Thay thế SimpleRNN bằng RNN

model.add(Dense(1, activation='sigmoid'))

# Bước 4: Compile mô hình
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Bước 5: Huấn luyện mô hình
model.fit(X, y, epochs=100, batch_size=32, validation_split=0.2)

# Bước 6: Dự đoán
sample_text = ["The customer support not satisfied with helpfulness"]
sample_seq = tokenizer.texts_to_sequences(sample_text)
sample_pad = pad_sequences(sample_seq, maxlen=maxlen)
prediction = model.predict(sample_pad)
print(f"Prediction:", "Good" if prediction > 0.5 else "Bad")
