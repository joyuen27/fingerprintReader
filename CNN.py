import tensorflow as tf
from tensorflow import keras
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

training_data = keras.utils.image_dataset_from_directory("data/training", color_mode="grayscale", label_mode="binary", shuffle=True, batch_size=32, image_size=(194, 90))
testing_data = keras.utils.image_dataset_from_directory("data/validation", color_mode="grayscale", label_mode="binary", shuffle=True, batch_size=32, image_size=(194, 90))

def cnn_model():
    # Input layer
    input = keras.Input(shape=(194, 90))

    # Slice input
    pic1 = keras.layers.Lambda(lambda x: tf.expand_dims(tf.split(x, [97, 97], axis=1)[0], axis=-1), output_shape=(97, 90, 1))(input)
    pic2 = keras.layers.Lambda(lambda x: tf.expand_dims(tf.split(x, [97, 97], axis=1)[1], axis=-1), output_shape=(97, 90, 1))(input)

    # CNN layers (channels_last is the default)
    pic1 = keras.layers.Conv2D(32, (3, 3), activation='relu')(pic1)
    pic1 = keras.layers.MaxPooling2D((2, 2))(pic1)
    pic1 = keras.layers.Conv2D(64, (3, 3), activation='relu')(pic1)
    pic1 = keras.layers.MaxPooling2D((2, 2))(pic1)

    pic2 = keras.layers.Conv2D(32, (3, 3), activation='relu')(pic2)
    pic2 = keras.layers.MaxPooling2D((2, 2))(pic2)
    pic2 = keras.layers.Conv2D(64, (3, 3), activation='relu')(pic2)
    pic2 = keras.layers.MaxPooling2D((2, 2))(pic2)

    # Flatten layer
    pic1 = keras.layers.Flatten()(pic1)
    pic2 = keras.layers.Flatten()(pic2)

    # Combine channels
    out = keras.layers.Concatenate()([pic1, pic2])

    # Second layer
    out = keras.layers.Dense(128, activation=tf.nn.relu)(out)

    # Final layer
    out = keras.layers.Dense(1, activation=tf.nn.sigmoid)(out)

    model = keras.Model(inputs=input, outputs=out)
    model.compile(optimizer="adam", loss="binary_crossentropy")

    model.summary()
    
    return model


cnn = cnn_model()
cnn.fit(training_data, epochs=10)
cnn.evaluate(testing_data)
