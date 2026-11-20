"""Shared helpers used by more than one explainability method."""


def build_column_blocks(model_module, categorical_columns, numerical_columns, vocab_sizes, feature_names):
    """Slice of each variable's own columns in the concatenated embedding-space vector (build_embed_model's
    output): categoricals in categorical_columns' own order, each as wide as its embedding dimension, then
    numericals in numerical_columns' order, each width 1.
    Args:
        model_module: the model's own module (models/lstm/model.py or models/transformer/model.py), for its
            embed_dim_for
        categorical_columns: load_dataset's own "categorical_columns"
        numerical_columns: load_dataset's own "numerical_columns"
        vocab_sizes: load_dataset's own "vocab_sizes"
        feature_names: load_dataset's own "feature_names"

    Returns a dict, variable name -> its column slice in embedding space. Keyed by variable name, not in
    feature_names order (categoricals and numericals are interleaved in feature_names, but concatenated as two
    separate blocks in the model) -- callers look each name up by key to put their result back in
    feature_names order."""
    column_blocks, cum = {}, 0
    for name, col in categorical_columns.items():
        d = model_module.embed_dim_for(vocab_sizes[name] - 2)
        column_blocks[name] = slice(cum, cum + d)
        cum += d
    for col in numerical_columns:
        column_blocks[feature_names[col]] = slice(cum, cum + 1)
        cum += 1
    return column_blocks
