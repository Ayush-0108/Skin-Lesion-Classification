import tensorflow as tf
from tensorflow import keras
ds = keras.utils.image_dataset_from_directory(
    directory = r"C:\Users\kumar\Skin_lesion_Classification\data\raw\ISIC2018",
    labels = "inferred",
    label_mode = "categorical",
    batch_size= 32,
    image_size = (256,256),
    seed = 123,
    validation_split = 0.2,
    subset = "both"
)
train_ds = ds[0]
validation_ds = ds[1]