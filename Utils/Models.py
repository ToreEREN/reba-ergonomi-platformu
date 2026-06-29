import sys
if 'Utils.Imports' in sys.modules:
    del sys.modules['Utils.Imports']
from Utils.Imports import *

# =========================================================
# 1) CUSTOM LAYERS
# =========================================================
class FeatureTokenizer(layers.Layer):
    """
    Her feature'ı bir token'a dönüştürür.
    Girdi : (batch, n_features)
    Çıktı : (batch, n_features, d_token)
    """
    def __init__(self, n_features, d_token, **kwargs):
        super().__init__(**kwargs)
        self.n_features = n_features
        self.d_token = d_token

    def build(self, input_shape):
        self.feature_weight = self.add_weight(
            name="feature_weight",
            shape=(self.n_features, self.d_token),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.feature_bias = self.add_weight(
            name="feature_bias",
            shape=(self.n_features, self.d_token),
            initializer="zeros",
            trainable=True,
        )

    def call(self, inputs):
        # inputs: (B, F)
        x = tf.expand_dims(inputs, axis=-1)              # (B, F, 1)
        x = x * self.feature_weight + self.feature_bias  # (B, F, d_token)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "n_features": self.n_features,
            "d_token": self.d_token
        })
        return config


class FeaturePositionalEmbedding(layers.Layer):
    """
    Feature index bazlı learnable positional embedding.
    """
    def __init__(self, n_tokens, d_token, **kwargs):
        super().__init__(**kwargs)
        self.n_tokens = n_tokens
        self.d_token = d_token
        self.embedding = layers.Embedding(input_dim=n_tokens, output_dim=d_token)

    def call(self, inputs):
        # inputs: (B, T, d)
        positions = tf.range(start=0, limit=self.n_tokens, delta=1)
        pos = self.embedding(positions)      # (T, d)
        return inputs + pos

    def get_config(self):
        config = super().get_config()
        config.update({
            "n_tokens": self.n_tokens,
            "d_token": self.d_token
        })
        return config


class AttentionPooling(layers.Layer):
    """
    Token'lar üzerinde attention-weighted pooling.
    CLS yerine daha esnek bir global özet çıkarır.
    Girdi : (B, T, d)
    Çıktı : (B, d)
    """
    def __init__(self, d_token, **kwargs):
        super().__init__(**kwargs)
        self.d_token = d_token
        self.score_dense = layers.Dense(1, name="attn_pool_score")

    def call(self, inputs):
        # inputs: (B, T, d)
        scores = self.score_dense(inputs)                 # (B, T, 1)
        weights = tf.nn.softmax(scores, axis=1)          # (B, T, 1)
        pooled = tf.reduce_sum(inputs * weights, axis=1) # (B, d)
        return pooled

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_token": self.d_token
        })
        return config


class ResidualMLPBlock(layers.Layer):
    """
    Residual MLP block:
    Dense -> Dropout -> Dense -> Residual -> LayerNorm
    """
    def __init__(self, units, dropout=0.1, activation="gelu", **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.dropout = dropout
        self.activation = activation

        self.proj_in = None
        self.dense1 = layers.Dense(units, activation=activation)
        self.drop1 = layers.Dropout(dropout)
        self.dense2 = layers.Dense(units)
        self.add = layers.Add()
        self.norm = layers.LayerNormalization()

    def build(self, input_shape):
        in_dim = int(input_shape[-1])
        if in_dim != self.units:
            self.proj_in = layers.Dense(self.units, name=f"{self.name}_proj_in")
        super().build(input_shape)

    def call(self, inputs, training=None):
        shortcut = inputs
        if self.proj_in is not None:
            shortcut = self.proj_in(shortcut)

        x = self.dense1(inputs)
        x = self.drop1(x, training=training)
        x = self.dense2(x)
        x = self.add([shortcut, x])
        x = self.norm(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "units": self.units,
            "dropout": self.dropout,
            "activation": self.activation
        })
        return config


# =========================================================
# 2) TRANSFORMER BLOCK
# =========================================================
def transformer_block(x, d_token, n_heads, dropout=0.1, ff_mult=4, block_id=0):
    # PreNorm Attention
    x_norm = layers.LayerNormalization(name=f"ln1_{block_id}")(x)
    attn_out = layers.MultiHeadAttention(
        num_heads=n_heads,
        key_dim=max(1, d_token // n_heads),
        dropout=dropout,
        name=f"mha_{block_id}"
    )(x_norm, x_norm)
    x = layers.Add(name=f"attn_add_{block_id}")([x, attn_out])

    # PreNorm FFN
    x_norm2 = layers.LayerNormalization(name=f"ln2_{block_id}")(x)
    ff = layers.Dense(d_token * ff_mult, activation="gelu", name=f"ff1_{block_id}")(x_norm2)
    ff = layers.Dropout(dropout, name=f"ff_drop1_{block_id}")(ff)
    ff = layers.Dense(d_token, name=f"ff2_{block_id}")(ff)
    ff = layers.Dropout(dropout, name=f"ff_drop2_{block_id}")(ff)
    x = layers.Add(name=f"ff_add_{block_id}")([x, ff])

    return x


# =========================================================
# 3) ADVANCED FT-TRANSFORMER MULTI-TASK REGRESSION
# =========================================================
def build_advanced_ft_transformer_regression(
    n_features,
    n_targets,
    d_token=96,
    n_blocks=6,
    n_heads=8,
    dropout=0.20,
    ff_mult=4,
    shared_mlp_units=(256, 128),
    per_target_hidden=64,
    use_uncertainty=False,
    name="AdvancedFTTransformer_MultiTask_Regression"
):
    """
    Advanced multi-output regression model.

    Yapı:
    Input
      -> FeatureTokenizer
      -> Positional Embedding
      -> Transformer Blocks
      -> Attention Pooling
      -> Shared Residual MLP trunk
      -> Target-specific heads
      -> Concatenate outputs

    use_uncertainty=False:
        çıktı şekli = (batch, n_targets)

    use_uncertainty=True:
        çıktı şekli = (batch, 2*n_targets)
        ilk n_targets = mean
        ikinci n_targets = log_var
    """
    inputs = keras.Input(shape=(n_features,), name="tabular_input")

    # -----------------------------------------------------
    # Feature tokenization
    # -----------------------------------------------------
    x = FeatureTokenizer(
        n_features=n_features,
        d_token=d_token,
        name="feature_tokenizer"
    )(inputs)  # (B, F, d)

    # -----------------------------------------------------
    # Positional embedding
    # -----------------------------------------------------
    x = FeaturePositionalEmbedding(
        n_tokens=n_features,
        d_token=d_token,
        name="feature_pos_embedding"
    )(x)

    # -----------------------------------------------------
    # Transformer backbone
    # -----------------------------------------------------
    for i in range(n_blocks):
        x = transformer_block(
            x,
            d_token=d_token,
            n_heads=n_heads,
            dropout=dropout,
            ff_mult=ff_mult,
            block_id=i
        )

    x = layers.LayerNormalization(name="backbone_final_ln")(x)

    # -----------------------------------------------------
    # Attention pooling
    # -----------------------------------------------------
    shared = AttentionPooling(d_token=d_token, name="attention_pooling")(x)

    # -----------------------------------------------------
    # Shared trunk
    # -----------------------------------------------------
    for i, units in enumerate(shared_mlp_units):
        shared = ResidualMLPBlock(
            units=units,
            dropout=dropout,
            activation="gelu",
            name=f"shared_resblock_{i}"
        )(shared)

    # -----------------------------------------------------
    # Optional shared projection
    # -----------------------------------------------------
    shared = layers.Dense(
        shared_mlp_units[-1],
        activation="gelu",
        name="shared_projection"
    )(shared)
    shared = layers.Dropout(dropout, name="shared_projection_dropout")(shared)

    # -----------------------------------------------------
    # Target-specific heads
    # -----------------------------------------------------
    target_outputs = []
    for t in range(n_targets):
        h = layers.Dense(
            per_target_hidden,
            activation="gelu",
            name=f"target_{t}_dense_0"
        )(shared)
        h = layers.Dropout(dropout, name=f"target_{t}_drop_0")(h)

        h = ResidualMLPBlock(
            units=per_target_hidden,
            dropout=dropout,
            activation="gelu",
            name=f"target_{t}_resblock"
        )(h)

        if use_uncertainty:
            mean_out = layers.Dense(1, activation="linear", name=f"target_{t}_mean")(h)
            log_var_out = layers.Dense(1, activation="linear", name=f"target_{t}_log_var")(h)
            target_outputs.extend([mean_out, log_var_out])
        else:
            out = layers.Dense(1, activation="linear", name=f"target_{t}_out")(h)
            target_outputs.append(out)

    outputs = layers.Concatenate(name="regression_output")(target_outputs)

    model = keras.Model(inputs=inputs, outputs=outputs, name=name)
    return model


# =========================================================
# 4) LOSS FUNCTIONS
# =========================================================
def heteroscedastic_multioutput_loss(n_targets):
    """
    use_uncertainty=True durumunda:
    y_pred = [mean_0, log_var_0, mean_1, log_var_1, ..., mean_n, log_var_n]
    veya concatenation sırasına göre 2*n_targets boyutlu vektör.

    Burada yukarıdaki modelde sıra:
    [target_0_mean, target_0_log_var, target_1_mean, target_1_log_var, ...]

    Bu yüzden slicing ona göre yapılır.
    """
    def loss_fn(y_true, y_pred):
        mean_list = []
        log_var_list = []

        for i in range(n_targets):
            mean_list.append(y_pred[:, 2 * i : 2 * i + 1])
            log_var_list.append(y_pred[:, 2 * i + 1 : 2 * i + 2])

        mean = tf.concat(mean_list, axis=1)      # (B, n_targets)
        log_var = tf.concat(log_var_list, axis=1)

        precision = tf.exp(-log_var)
        loss = precision * tf.square(y_true - mean) + log_var
        return tf.reduce_mean(loss)

    return loss_fn


# =========================================================
# 5) COMPILE HELPER
# =========================================================
def compile_advanced_ft_transformer(
    model,
    n_targets,
    lr=1e-3,
    weight_decay=1e-5,
    use_uncertainty=False
):
    """
    Model compile yardımcı fonksiyonu.
    """
    try:
        optimizer = keras.optimizers.AdamW(
            learning_rate=lr,
            weight_decay=weight_decay
        )
    except:
        optimizer = keras.optimizers.Adam(learning_rate=lr)

    if use_uncertainty:
        loss_fn = heteroscedastic_multioutput_loss(n_targets=n_targets)
    else:
        loss_fn = "mse"

    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=["mae"]
    )
    return model


# =========================================================
# 6) METRİKLER
# =========================================================
def regression_metrics_multioutput(y_true, y_pred, target_names=None, prefix=""):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.ndim != 2:
        raise ValueError("y_true shape must be (n_samples, n_targets)")
    if y_pred.ndim != 2:
        raise ValueError("y_pred shape must be (n_samples, n_targets)")

    n_out = y_true.shape[1]
    rows = []

    for i in range(n_out):
        tname = target_names[i] if target_names is not None else f"target_{i}"
        yt = y_true[:, i]
        yp = y_pred[:, i]

        mae = mean_absolute_error(yt, yp)
        mse = mean_squared_error(yt, yp)
        rmse = np.sqrt(mse)

        try:
            r2 = r2_score(yt, yp)
        except:
            r2 = np.nan

        rows.append({
            "set": prefix,
            "target": tname,
            "mae": mae,
            "mse": mse,
            "rmse": rmse,
            "r2": r2
        })

    metric_df = pd.DataFrame(rows)

    overall_row = {
        "set": prefix,
        "target": "OVERALL_MEAN",
        "mae": metric_df["mae"].mean(),
        "mse": metric_df["mse"].mean(),
        "rmse": metric_df["rmse"].mean(),
        "r2": metric_df["r2"].mean()
    }

    metric_df = pd.concat([metric_df, pd.DataFrame([overall_row])], ignore_index=True)
    return metric_df


# =========================================================
# 7) TAHMİN YARDIMCISI
# =========================================================
def predict_advanced_multioutput(model, X, use_uncertainty=False, n_targets=None, batch_size=256):
    """
    Tahmin helper.

    use_uncertainty=False:
        return y_pred

    use_uncertainty=True:
        return mean_pred, log_var_pred
    """
    pred = model.predict(X, batch_size=batch_size, verbose=0)

    if not use_uncertainty:
        return pred

    if n_targets is None:
        raise ValueError("n_targets must be provided when use_uncertainty=True")

    mean_cols = []
    log_var_cols = []
    for i in range(n_targets):
        mean_cols.append(pred[:, 2 * i : 2 * i + 1])
        log_var_cols.append(pred[:, 2 * i + 1 : 2 * i + 2])

    mean_pred = np.concatenate(mean_cols, axis=1)
    log_var_pred = np.concatenate(log_var_cols, axis=1)
    return mean_pred, log_var_pred


# =========================================================
# 8) CALLBACK HELPER
# =========================================================
def get_default_callbacks(
    monitor="val_loss",
    patience_es=25,
    patience_rlr=10,
    factor=0.5,
    min_lr=1e-6,
    restore_best_weights=True
):
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor=monitor,
            patience=patience_es,
            restore_best_weights=restore_best_weights,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor=monitor,
            factor=factor,
            patience=patience_rlr,
            min_lr=min_lr,
            verbose=1
        )
    ]
    return callbacks

# =========================================================
# 6) INPUT HAZIRLAMA FONKSİYONU
# =========================================================
def prepare_input_dataframe(df_input, input_cols):
    df_input = df_input.copy()

    if not isinstance(df_input, pd.DataFrame):
        raise TypeError("df_input bir pandas DataFrame olmalıdır.")

    # Eksik kolonlar
    missing_cols = [col for col in input_cols if col not in df_input.columns]
    extra_cols = [col for col in df_input.columns if col not in input_cols]

    if len(missing_cols) > 0:
        raise ValueError(f"Eksik input kolonları var: {missing_cols}")

    if len(extra_cols) > 0:
        print("Uyarı - fazladan kolonlar yok sayılacak:", extra_cols)

    # Sıralamayı eğitimdeki sıraya göre düzelt
    df_input = df_input[input_cols].copy()

    # Numeric dönüşüm
    for col in input_cols:
        df_input[col] = pd.to_numeric(df_input[col], errors="coerce")

    # NaN kontrolü
    nan_cols = df_input.columns[df_input.isna().any()].tolist()
    if len(nan_cols) > 0:
        raise ValueError(f"Input verisinde NaN oluştu. Problemli kolonlar: {nan_cols}")

    return df_input


# =========================================================
# 7) TAHMİN FONKSİYONU
# =========================================================
def predict_from_df_input(df_input, model, scaler, input_cols, target_cols,
                          use_uncertainty=False, batch_size=256):
    df_prepared = prepare_input_dataframe(df_input, input_cols)

    X_input = df_prepared.values.astype("float32")
    X_input_s = scaler.transform(X_input)

    if use_uncertainty:
        y_pred_mean, y_pred_logvar = predict_advanced_multioutput(
            model=model,
            X=X_input_s,
            use_uncertainty=True,
            n_targets=len(target_cols),
            batch_size=batch_size
        )

        df_pred_mean = pd.DataFrame(y_pred_mean, columns=target_cols)
        df_pred_logvar = pd.DataFrame(
            y_pred_logvar,
            columns=[f"{c}_logvar" for c in target_cols]
        )

        df_out = pd.concat([df_prepared.reset_index(drop=True), df_pred_mean, df_pred_logvar], axis=1)
        return df_out, df_pred_mean, df_pred_logvar

    else:
        y_pred = predict_advanced_multioutput(
            model=model,
            X=X_input_s,
            use_uncertainty=False,
            batch_size=batch_size
        )

        df_pred = pd.DataFrame(y_pred, columns=target_cols)
        df_out = pd.concat([df_prepared.reset_index(drop=True), df_pred], axis=1)
        return df_out, df_pred

