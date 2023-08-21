import numpy as np
import tensorflow as tf
from tensorflow import keras as keras
import tf2onnx
import idx2numpy
import vehicle_lang as vcl


def custom_train(
    model,
    train_dataset,  # datasets are type of `tf.data.Dataset` (before shuffling and batching)
    test_dataset,
    num_epochs,
    alpha=1,
    constraint_loss=None,
):
    # set optimizer and loss functions for the model
    # default to Adam and CategoricalCrossentropy
    optimizer = keras.optimizers.Adam()
    standard_loss = keras.losses.CategoricalCrossentropy(from_logits=True) # the last activation is 'relu'

    # set accuracy / loss metrics for train/test sets
    train_acc_metric = keras.metrics.CategoricalAccuracy()
    test_acc_metric = keras.metrics.CategoricalAccuracy()
    train_loss_metric = keras.metrics.CategoricalCrossentropy()
    test_loss_metric = keras.metrics.CategoricalCrossentropy()

    # defines training and test routine for an epoch
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}")
        # defines forward/backward propagation using auto-differentiation
        for x_batch_train, y_batch_train in train_dataset:
            # open a GradientTape to record the forward pass
            with tf.GradientTape() as tape:
                # outputs for this batch
                outputs = model(x_batch_train, training=True)
                # calculate each losses
                standard_batch_loss = standard_loss(y_batch_train, outputs)
                constraint_batch_loss = constraint_loss(y_batch_train, outputs)
                # calculate weighted loss
                weighted_loss = standard_batch_loss * alpha + constraint_batch_loss * (
                    1 - alpha
                )
            # automatically retieve the gradients of the trainable variables with respect to the loss
            grads = tape.gradient(weighted_loss, model.trainable_weights)
            # take one step of gradient descent
            # by updating the value of the variables to minimize the loss
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

        # evaluate train dataset at the end of each epoch
        for x_batch_train, y_batch_train in train_dataset:
            train_outputs = model(x_batch_train, training=False)
            train_acc_metric.update_state(y_batch_train, train_outputs)
            train_loss_metric.update_state(y_batch_train, train_outputs)

        # evaluate test dataset at the end of each epoch
        for x_batch_test, y_batch_test in test_dataset:
            test_outputs = model(x_batch_test, training=False)
            test_acc_metric.update_state(y_batch_test, test_outputs)
            test_loss_metric.update_state(y_batch_test, test_outputs)

        train_acc = train_acc_metric.result()
        test_acc = test_acc_metric.result()
        train_loss = train_loss_metric.result()
        test_loss = test_loss_metric.result()

        train_acc_metric.reset_states()
        test_acc_metric.reset_states()
        train_loss_metric.reset_states()
        test_loss_metric.reset_states()

        print(
            f"""Train acc: {float(train_acc):.4f}, Train loss: {float(train_loss):.4f}, Test acc: {float(test_acc):.4f}, Test loss: {float(test_loss):.4f}
            """
        )
    return model

def mnist_custom_train(buffer_size=1024, batch_size=200):
    ## set different alphas
    alphas = [1., 0.75, 0.5, 0.25, 0.]

    ## prepare datasets
    # get the dataset to be used, format to be prepared for NN training
    from keras.datasets import mnist

    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # scale images to the [0, 1] range
    X_train = X_train.astype("float32") / 255
    X_test = X_test.astype("float32") / 255

    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    # process tensors to dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))

    # preprocess datasets
    # shuffle train data to avoid learning from the ordering and to prevent overfit 
    train_dataset = train_dataset.shuffle(buffer_size).batch(batch_size)
    test_dataset = test_dataset.batch(batch_size)

    ## custom train the model with different alphas
    for alpha in alphas:
        # struct model
        model = keras.Sequential(
            [
                keras.layers.Input(shape=(28, 28)),
                keras.layers.Flatten(),
                keras.layers.Dense(30, activation="relu"),
                keras.layers.Dense(10, activation="relu"),
            ]
        )
        model = custom_train(
            model,  # pass the prepared model and datasets
            train_dataset,
            test_dataset,
            num_epochs=5,  
            alpha=alpha,
            constraint_loss=keras.losses.mae  # random choice for now
        )

        # save the model
        model.save(f"mnist_custom_train_with_alpha_{alpha}")
        # convert model to onnx
        onnx_model, _ = tf2onnx.convert.from_keras(
            model, output_path=f"mnist_custom_train_with_alpha_{alpha}.onnx"
        )

    # convert data to idx
    idx2numpy.convert_to_file("mnist_test_data.idx", X_test)
    idx2numpy.convert_to_file("mnist_test_label.idx", y_test)

def cr_infit_constraint_loss(expected_outputs, actual_outputs, weight=0.0000001):
    # (y_train_batch, outputs)
    # logical expression: f(x, theta) != f(x', theta) where f is the network
    # translation
    result = np.where(expected_outputs == actual_outputs, 0, weight)
    return np.sum(result)

