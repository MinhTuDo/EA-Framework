from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, Activation, BatchNormalization

from keras_flops import get_flops

# build model
inp = Input((32, 32, 3))
x = Conv2D(32, kernel_size=(3, 3), activation="relu", padding='same')(inp)
x = BatchNormalization(axis=3)(x)
x = Conv2D(64, (3, 3), activation="relu", padding='same')(x)
x = BatchNormalization(axis=3)(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Dropout(0.25)(x)
x = Flatten()(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.5)(x)
out = Dense(10, activation="softmax")(x)
model = Model(inp, out)
model.summary()

print(len(model.layers[-3].get_weights()[0]))

# Calculae FLOPS
flops = get_flops(model, batch_size=1)
print(f"FLOPS: {flops / 10 ** 9:.03} G")
# >>> FLOPS: 0.0338 G 2,118,026