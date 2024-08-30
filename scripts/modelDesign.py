# Imports
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, GRU, BatchNormalization, Flatten
from tensorflow.keras.optimizers import Adam

def build_model_gru(
    input_shape,
    units=64,
    activation="tanh",
    recurrent_activation="sigmoid",
    use_bias=True,
    kernel_initializer="glorot_uniform",
    recurrent_initializer="orthogonal",
    bias_initializer="zeros",
    kernel_regularizer=None,
    recurrent_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    recurrent_constraint=None,
    bias_constraint=None,
    dropout=0.0,
    recurrent_dropout=0.0,
    seed=None,
    return_sequences=False,
    return_state=False,
    go_backwards=False,
    stateful=False,
    unroll=False,
    reset_after=True,
    **kwargs
):
    model = Sequential()
    
    # Add GRU layer
    model.add(GRU(
        units=units,
        activation=activation,
        recurrent_activation=recurrent_activation,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        recurrent_initializer=recurrent_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=kernel_regularizer,
        recurrent_regularizer=recurrent_regularizer,
        bias_regularizer=bias_regularizer,
        activity_regularizer=activity_regularizer,
        kernel_constraint=kernel_constraint,
        recurrent_constraint=recurrent_constraint,
        bias_constraint=bias_constraint,
        dropout=dropout,
        recurrent_dropout=recurrent_dropout,
        return_sequences=return_sequences,
        return_state=return_state,
        go_backwards=go_backwards,
        stateful=stateful,
        unroll=unroll,
        reset_after=reset_after,
        input_shape=input_shape,
        **kwargs
    ))
    
    # Example: Add another GRU layer if return_sequences=True
    if return_sequences:
        model.add(GRU(units=units // 2, return_sequences=False))
    
    # Example: Add a Dense layer
    model.add(Dense(units=64, activation='relu'))
    
    # Example: Add a Dropout layer
    model.add(Dropout(rate=0.5))
    
    # Example: Add Batch Normalization
    model.add(BatchNormalization())
    
    # Example: Add a final output layer (regression case)
    model.add(Dense(units=1, activation='linear'))

    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

    return model


def build_model_lstm(input_shape, units=50, num_lstm_layers=1, activation='tanh', recurrent_activation='sigmoid',
                     use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal',
                     bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None,
                     recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None,
                     kernel_constraint=None, recurrent_constraint=None, bias_constraint=None,
                     dropout=0.0, recurrent_dropout=0.0, return_sequences=False, return_state=False,
                     go_backwards=False, stateful=False, unroll=False):
    model = Sequential()
    
    for i in range(num_lstm_layers):
        # For the first layer, use input_shape
        if i == 0:
            model.add(LSTM(
                units=units,
                activation=activation,
                recurrent_activation=recurrent_activation,
                use_bias=use_bias,
                kernel_initializer=kernel_initializer,
                recurrent_initializer=recurrent_initializer,
                bias_initializer=bias_initializer,
                unit_forget_bias=unit_forget_bias,
                kernel_regularizer=kernel_regularizer,
                recurrent_regularizer=recurrent_regularizer,
                bias_regularizer=bias_regularizer,
                activity_regularizer=activity_regularizer,
                kernel_constraint=kernel_constraint,
                recurrent_constraint=recurrent_constraint,
                bias_constraint=bias_constraint,
                dropout=dropout,
                recurrent_dropout=recurrent_dropout,
                return_sequences=True,  # Typically return sequences for intermediate layers
                return_state=return_state,
                go_backwards=go_backwards,
                stateful=stateful,
                unroll=unroll,
                input_shape=input_shape
            ))
        else:
            model.add(LSTM(
                units=units,
                activation=activation,
                recurrent_activation=recurrent_activation,
                use_bias=use_bias,
                kernel_initializer=kernel_initializer,
                recurrent_initializer=recurrent_initializer,
                bias_initializer=bias_initializer,
                unit_forget_bias=unit_forget_bias,
                kernel_regularizer=kernel_regularizer,
                recurrent_regularizer=recurrent_regularizer,
                bias_regularizer=bias_regularizer,
                activity_regularizer=activity_regularizer,
                kernel_constraint=kernel_constraint,
                recurrent_constraint=recurrent_constraint,
                bias_constraint=bias_constraint,
                dropout=dropout,
                recurrent_dropout=recurrent_dropout,
                return_sequences=True,  # Return sequences for intermediate layers
                return_state=return_state,
                go_backwards=go_backwards,
                stateful=stateful,
                unroll=unroll
            ))

        # Dropout layer after each LSTM layer
        model.add(Dropout(dropout))
    
    # Final LSTM layer (return_sequences should be False for the last layer)
    model.add(LSTM(
        units=units,
        activation=activation,
        recurrent_activation=recurrent_activation,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        recurrent_initializer=recurrent_initializer,
        bias_initializer=bias_initializer,
        unit_forget_bias=unit_forget_bias,
        kernel_regularizer=kernel_regularizer,
        recurrent_regularizer=recurrent_regularizer,
        bias_regularizer=bias_regularizer,
        activity_regularizer=activity_regularizer,
        kernel_constraint=kernel_constraint,
        recurrent_constraint=recurrent_constraint,
        bias_constraint=bias_constraint,
        dropout=dropout,
        recurrent_dropout=recurrent_dropout,
        return_sequences=False,  # Do not return sequences for the last LSTM layer
        return_state=return_state,
        go_backwards=go_backwards,
        stateful=stateful,
        unroll=unroll
    ))

    # Output layer for regression
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    return model