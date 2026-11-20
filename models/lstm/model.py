import re

from keras.layers import LSTM, BatchNormalization, Dense, Activation, Input, Concatenate, Lambda
from keras.models import Model

from models.common import (f1, embed_dim_for, _gather_fn, CategoricalEmbedding, build_embed_model,
                            build_explain_model, build_logits_model, build_raw_logits_model, train_model)


def build_model(max_len, vocab_sizes, categorical_columns, numerical_columns,
                 n_neurons = 100, n_layers = 8, output_dim = 1, task = "regression", event_level = True,
                 model_name = None):
    """Builds the stacked-LSTM model from the loader's vocab/column info."""
    width = max(list(categorical_columns.values()) + numerical_columns) + 1
    inputs = Input(shape = (max_len, width), name = "prefix")

    embeds = [CategoricalEmbedding(col, vocab_sizes[name], embed_dim_for(vocab_sizes[name] - 2),
                                    name = f"cat_embed_{re.sub(r'[^A-Za-z0-9_.-]', '_', name)}")(inputs)
              for name, col in categorical_columns.items()]
    numeric = Lambda(_gather_fn, arguments = {"columns": numerical_columns},
                      output_shape = (max_len, len(numerical_columns)),
                      name = "numerical_slice")(inputs)

    x = Concatenate(axis = -1, name = "concat_features")(embeds + [numeric])

    if n_layers == 1:
        x = LSTM(n_neurons, implementation = 2, recurrent_dropout = 0.2)(x)
        x = BatchNormalization()(x)
    else:
        x = LSTM(n_neurons, implementation = 2, recurrent_dropout = 0.2, return_sequences = True)(x)
        x = BatchNormalization()(x)
        for _ in range(n_layers - 2):
            x = LSTM(n_neurons, implementation = 2, recurrent_dropout = 0.2, return_sequences = True)(x)
            x = BatchNormalization()(x)
        x = LSTM(n_neurons, implementation = 2, recurrent_dropout = 0.2)(x)
        x = BatchNormalization()(x)

    if task == "regression":
        outputs = Dense(1)(x)
        model = Model(inputs, outputs, name = model_name)
        model.compile(loss = "mae", optimizer = "Nadam", metrics = ["mean_squared_error", "mae"])
    else:
        x = Dense(output_dim, name = "logits")(x)
        if event_level:
            outputs = Activation("sigmoid")(x)
            model = Model(inputs, outputs, name = model_name)
            model.compile(loss = "binary_crossentropy", optimizer = "Nadam", metrics = ["accuracy", "binary_accuracy", f1])
        else:
            outputs = Activation("softmax")(x)
            model = Model(inputs, outputs, name = model_name)
            model.compile(loss = "categorical_crossentropy", optimizer = "Nadam", metrics = ["accuracy"])
    return model
