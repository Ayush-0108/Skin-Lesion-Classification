from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale = 1./255,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True,
    validation_split = 0.2
)

train_generator = train_datagen.flow_from_directory(
    directory = r"C:\Users\kumar\Skin_lesion_Classification\data\raw\ISIC2018",
    target_size = (256,256),
    batch_size = 32,
    class_mode = "categorical",
    subset = "training",
    seed = 123
)

validation_generator = train_datagen.flow_from_directory(
    directory = r"C:\Users\kumar\Skin_lesion_Classification\data\raw\ISIC2018",
    target_size = (256,256),
    batch_size = 32,
    class_mode = "categorical",
    subset = "validation",
    seed = 123
)