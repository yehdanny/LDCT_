import tensorflow as tf
print("TensorFlow 版本:", tf.__version__)


gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"TensorFlow 已偵測到 {len(gpus)} 個 GPU：{gpus}")
else:
    print("TensorFlow 未偵測到 GPU，模型將運行在 CPU 上。")