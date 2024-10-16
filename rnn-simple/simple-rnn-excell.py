import pandas as pd
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense, Embedding
# from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# Bước 1: Đọc dữ liệu từ file Excel
# dataset = pd.read_excel("sentiment_dataset.xlsx")  # Đảm bảo bạn thay thế bằng đường dẫn tệp của bạn
dataset = pd.read_excel(r"C:\recurrent-neuron-netwwork\rnn-simple\sentiment_dataset.xlsx")
texts = dataset['Text'].values
labels = dataset['Sentiment'].values

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
model.add(SimpleRNN(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Bước 4: Compile mô hình
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Bước 5: Huấn luyện mô hình
model.fit(X, y, epochs=50, batch_size=32, validation_split=0.2)

# Bước 6: Dự đoán
sample_text = ["This product is not amazing!"]
sample_seq = tokenizer.texts_to_sequences(sample_text)
sample_pad = pad_sequences(sample_seq, maxlen=maxlen)
prediction = model.predict(sample_pad)
print(f"Prediction:", "Tích cực" if prediction > 0.5 else "Tiêu cực")
