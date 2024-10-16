from tensorflow.keras.models import load_model

# Tải mô hình đã lưu
model = load_model('save_models.keras')

# Hoặc nếu bạn dùng định dạng HDF5
# model = load_model('model_name.h5')
#test