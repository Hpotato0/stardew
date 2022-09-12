from tensorflow import keras
from keras import layers
from keras.callbacks import EarlyStopping, ModelCheckpoint

from lists import without_imexport
import os

class Transformer(keras.Model):
    def __init__(self, x_train, df_number, epoch, batch, learning_rate=0.001):
        super().__init__()
        self.model = build_model(
        x_train.shape[1:],
        head_size=256,
        num_heads=4,
        ff_dim=4,
        num_transformer_blocks=4,
        mlp_units=[128],
        mlp_dropout=0.4,
        dropout=0.25,
        )

        self.model.compile(
            loss="mean_squared_error",
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate)
        )

        self.loaded = False
        self.load_model(df_number, epoch, batch)
    
    def load_model(self, df_number, epoch, batch):
        if os.path.exists(f'./model/{epoch}') == False:
            os.makedirs(f'./model/{epoch}')
            
        model_path = f'./model/{epoch}/transformer-{df_number}-{epoch}-{batch}.h5'
        general_model_path = f'./model/{epoch}/transformer-general-{epoch}-{batch}.h5'
        general_without_model_path = f'./model/{epoch}/transformer-general-without-{epoch}-{batch}.h5'

        if os.path.exists(model_path) == True:
            self.model.load_weights(model_path)
            print(f"successfully loaded model {model_path}")
            self.loaded = True
        elif 'general' in df_number:
            return
        elif int(df_number) not in without_imexport:
            if os.path.exists(general_model_path) == True:
                self.model.load_weights(general_model_path)
                print(f"successfully loaded general model {general_model_path}")
        elif int(df_number) in without_imexport:
            if os.path.exists(general_without_model_path) == True:
                self.model.load_weights(general_without_model_path)
                print(f"successfully loaded general model without imexport {general_without_model_path}")

    def save_model(self, df_number, epoch, batch):
        # 모델 저장
        self.model.save(f'./model/{epoch}/transformer-{df_number}-{epoch}-{batch}.h5')

## Transformer 정의
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):

    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs

    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res

def build_model(input_shape, head_size, num_heads, ff_dim, num_transformer_blocks, mlp_units, dropout=0, mlp_dropout=0):
    inputs = keras.Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    outputs = layers.Dense(28)(x) # 4주 예측
    return keras.Model(inputs, outputs)

## keras eraly stop, chekpoint 정의
def call_back_set(name, epoch, batch_size):
    early_stopping = EarlyStopping(monitor='val_loss', patience=10)

    if os.path.exists(f'./check') == False:
        os.makedirs(f'./check')

    filename = f'./check/{name}-{epoch}-{batch_size}.h5'

    checkpoint = ModelCheckpoint(filename,
                                 monitor='val_loss',
                                 verbose=1,
                                 save_best_only=True,
                                 save_weights_only=True,
                                 mode='auto'
                                 )
    return [early_stopping, checkpoint]
