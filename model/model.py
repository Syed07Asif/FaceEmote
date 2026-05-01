from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, BatchNormalization, Activation,
    SeparableConv2D, MaxPooling2D, Add,
    GlobalAveragePooling2D, Dense
)

def mini_xception(input_shape, num_classes):
    inputs = Input(shape=input_shape)

    # Initial Conv Layers
    x = Conv2D(32, (3,3), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(64, (3,3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # -------- Block 1 --------
    residual = Conv2D(128, (1,1), strides=(2,2), padding='same')(x)

    x = SeparableConv2D(128, (3,3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = SeparableConv2D(128, (3,3), padding='same')(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D((3,3), strides=(2,2), padding='same')(x)
    x = Add()([x, residual])

    # -------- Block 2 --------
    residual = Conv2D(256, (1,1), strides=(2,2), padding='same')(x)

    x = SeparableConv2D(256, (3,3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = SeparableConv2D(256, (3,3), padding='same')(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D((3,3), strides=(2,2), padding='same')(x)
    x = Add()([x, residual])

    # -------- Block 3 --------
    residual = Conv2D(512, (1,1), strides=(2,2), padding='same')(x)

    x = SeparableConv2D(512, (3,3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = SeparableConv2D(512, (3,3), padding='same')(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D((3,3), strides=(2,2), padding='same')(x)
    x = Add()([x, residual])

    # Output
    x = GlobalAveragePooling2D()(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    return Model(inputs, outputs)