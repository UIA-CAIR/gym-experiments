from tensorflow.python.keras import Input
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.layers.convolutional import Conv2D

from util.loss_functions import huber_loss_1


def model(state_size, action_size, learning_rate):
    n_routing = 3

    x = Input(shape=state_size)
    conv1 = Conv2D(filters=256, kernel_size=1, strides=1, padding='valid', activation='relu', name='conv1')(x)
    primarycaps = PrimaryCap(conv1, dim_vector=8, n_channels=32, kernel_size=9, strides=2, padding='valid')
    digitcaps = CapsuleLayer(num_capsule=action_size, dim_vector=16, num_routing=n_routing, name='digitcaps')(primarycaps)
    out_caps = Length(name='out_caps')(digitcaps)

    model = Model(inputs=[x], outputs=[out_caps])
    model.compile(optimizer=Adam(lr=learning_rate), loss=huber_loss_1)
    return model, "capsule1"