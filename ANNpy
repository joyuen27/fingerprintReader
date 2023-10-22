import tensorflow as tf
from tensorflow import keras
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

training_data = keras.utils.image_dataset_from_directory("data/training", color_mode="grayscale", label_mode="binary", shuffle=True, batch_size=32, image_size=(194, 90))
testing_data = keras.utils.image_dataset_from_directory("data/testing", color_mode="grayscale", label_mode="binary", shuffle=True, batch_size=32, image_size=(194, 90))

def nn_model():
    # Input layer
    input = keras.Input(shape=(194, 90))

    # Slice input
    pic1 = keras.layers.Lambda(lambda x: tf.expand_dims(tf.split(x, [97, 97], axis=1)[0], axis=-1), output_shape=(97, 90, 1))(input)
    pic2 = keras.layers.Lambda(lambda x: tf.expand_dims(tf.split(x, [97, 97], axis=1)[1], axis=-1), output_shape=(97, 90, 1))(input)

    # Flatten input
    pic1 = keras.layers.Flatten()(pic1)
    pic2 = keras.layers.Flatten()(pic2)

    # First layer
    pic1 = keras.layers.Dense(128, activation=tf.nn.relu)(pic1)
    pic2 = keras.layers.Dense(128, activation=tf.nn.relu)(pic2)

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


nn = nn_model()
nn.fit(training_data, epochs=15)
nn.evaluate(testing_data)
