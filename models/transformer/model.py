import re

import tensorflow as tf
from keras.layers import (Activation, Concatenate, Dense, Dropout, Embedding, Input, Lambda, Layer,
                           LayerNormalization, MultiHeadAttention)
from keras.models import Model
from keras.saving import register_keras_serializable

from models.common import (f1, embed_dim_for, CategoricalEmbedding, _gather_fn, build_embed_model,
                            build_explain_model, build_logits_model, build_raw_logits_model, train_model)


@register_keras_serializable(package = "v3transformer")
class PositionalEncoding(Layer):
    """Learned positional embedding added to the concatenated embeddings, zeroed at padding positions."""
    def __init__(self, maxlen, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.maxlen = maxlen
        self.embed_dim = embed_dim
        self.pos_emb = Embedding(maxlen, embed_dim, name = (kwargs.get("name") or "pos") + "_table")

    def build(self, input_shape):
        self.pos_emb.build((None, self.maxlen))
        super().build(input_shape)

    def call(self, inputs):
        length = tf.shape(inputs)[1]
        positions = self.pos_emb(tf.range(length))
        is_real = tf.reduce_any(tf.not_equal(inputs, 0.0), axis = -1, keepdims = True)
        return (inputs + positions) * tf.cast(is_real, inputs.dtype)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update({"maxlen": self.maxlen, "embed_dim": self.embed_dim})
        return config


@register_keras_serializable(package = "v3transformer")
class MaskedTransformerBlock(Layer):
    """MultiHeadAttention + FFN block (residual + layernorm around each), with a padding mask in the attention."""
    def __init__(self, embed_dim, num_heads, ff_dim, rate = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim, self.num_heads, self.ff_dim, self.rate = embed_dim, num_heads, ff_dim, rate
        self.att = MultiHeadAttention(num_heads = num_heads, key_dim = embed_dim)
        self.ffn = tf.keras.Sequential([Dense(ff_dim, activation = "relu"), Dense(embed_dim)])
        self.layernorm_a = LayerNormalization(epsilon = 1e-6)
        self.layernorm_b = LayerNormalization(epsilon = 1e-6)
        self.dropout_a = Dropout(rate)
        self.dropout_b = Dropout(rate)

    def build(self, input_shape):
        self.att.build(input_shape, input_shape)
        self.ffn.build(input_shape)
        self.layernorm_a.build(input_shape)
        self.layernorm_b.build(input_shape)
        super().build(input_shape)

    def call(self, inputs, training = None):
        is_real = tf.reduce_any(tf.not_equal(inputs, 0.0), axis = -1)
        real_mask = tf.cast(is_real, inputs.dtype)[:, :, None]
        attention_mask = is_real[:, tf.newaxis, :]

        attn_output = self.att(inputs, inputs, attention_mask = attention_mask)
        attn_output = self.dropout_a(attn_output, training = training)
        out_a = self.layernorm_a(inputs + attn_output)
        ffn_output = self.ffn(out_a)
        ffn_output = self.dropout_b(ffn_output, training = training)
        out_b = self.layernorm_b(out_a + ffn_output)
        return out_b * real_mask

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update({"embed_dim": self.embed_dim, "num_heads": self.num_heads,
                       "ff_dim": self.ff_dim, "rate": self.rate})
        return config


@register_keras_serializable(package = "v3transformer")
class ExtractLastToken(Layer):
    """Picks the representation of the last (real) position of the prefix."""
    def call(self, inputs):
        return inputs[:, -1, :]

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])


def build_model(max_len, vocab_sizes, categorical_columns, numerical_columns,
                 n_blocks = 1, num_heads = 4, ff_dim = 64, dropout_rate = 0.1,
                 output_dim = 1, task = "regression", event_level = True, model_name = None):
    """Builds the Transformer model from the loader's vocab/column info."""
    width = max(list(categorical_columns.values()) + numerical_columns) + 1
    inputs = Input(shape = (max_len, width), name = "prefix")

    embeds = [CategoricalEmbedding(col, vocab_sizes[name], embed_dim_for(vocab_sizes[name] - 2),
                                    name = f"cat_embed_{re.sub(r'[^A-Za-z0-9_.-]', '_', name)}")(inputs)
              for name, col in categorical_columns.items()]
    numeric = Lambda(_gather_fn, arguments = {"columns": numerical_columns},
                      output_shape = (max_len, len(numerical_columns)),
                      name = "numerical_slice")(inputs)

    x = Concatenate(axis = -1, name = "concat_features")(embeds + [numeric])
    embed_dim = x.shape[-1]

    x = PositionalEncoding(max_len, embed_dim, name = "positional_encoding")(x)
    for i in range(n_blocks):
        x = MaskedTransformerBlock(embed_dim, num_heads, ff_dim, rate = dropout_rate, name = f"transformer_block_{i}")(x)
    x = ExtractLastToken(name = "extract_last_token")(x)

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
