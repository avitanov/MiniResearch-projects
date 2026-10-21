"""Mini-batch SGD logistic regression implemented from scratch.

This module contains a small binary logistic regression classifier trained
with mini-batch stochastic gradient descent. It is used in the HAR notebook
to classify activities as STATIC or DYNAMIC.

The implementation expects NumPy arrays:
- Features ``X``: numeric array with shape ``(n_samples, n_features)``.
  Recommended dtype is ``float64`` or ``float32``.
- Labels ``y``: binary numeric array with shape ``(n_samples,)``.
  Values should be encoded as ``0`` and ``1``. Integer, boolean, or float
  dtypes work as long as values represent binary labels.
"""

import numpy as np


class MiniBatchSGDLogisticRegression:
    """Binary logistic regression trained with mini-batch SGD.

    The model learns weights ``w`` and bias ``b`` for:

    ``sigmoid(X @ w + b)``

    Training minimizes binary cross-entropy. At the beginning of each epoch,
    the training rows are shuffled, then split into mini-batches. After all
    batches are processed, full-training-set loss is recorded in
    ``loss_history``.

    Parameters
    ----------
    learning_rate : float, default=0.01
        Step size used for gradient descent updates.
    n_epochs : int, default=100
        Number of full passes over the training data.
    batch_size : int, default=32
        Number of samples per mini-batch. If equal to ``n_samples``, training
        becomes full-batch gradient descent.
    random_state : int, default=42
        Seed used for reproducible shuffling.

    Attributes
    ----------
    weights : np.ndarray or None
        Learned feature weights with shape ``(n_features,)`` and dtype float.
        It is ``None`` before ``fit`` is called.
    bias : float or None
        Learned intercept term. It is ``None`` before ``fit`` is called.
    loss_history : list[float]
        Binary cross-entropy loss after each epoch.
    """

    def __init__(
        self,
        learning_rate=0.01,
        n_epochs=100,
        batch_size=32,
        random_state=42,
    ):
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.random_state = random_state

        self.weights = None
        self.bias = None
        self.loss_history = []

    def _sigmoid(self, z):
        """Apply the sigmoid function element-wise.

        Parameters
        ----------
        z : np.ndarray or float
            Linear model output. Usually shape ``(n_samples,)`` with float
            dtype.

        Returns
        -------
        np.ndarray or float
            Values in ``[0, 1]`` with the same shape as ``z``.
        """
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def _binary_cross_entropy(self, y_true, y_pred):
        """Compute mean binary cross-entropy loss.

        Parameters
        ----------
        y_true : np.ndarray
            True binary labels with shape ``(n_samples,)``. Values should be
            ``0`` or ``1``.
        y_pred : np.ndarray
            Predicted probabilities for class ``1`` with shape
            ``(n_samples,)`` and float dtype.

        Returns
        -------
        float
            Mean binary cross-entropy over all samples.
        """
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)

        return -np.mean(
            y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)
        )

    def fit(self, X, y):
        """Fit the model on binary-labeled training data.

        Parameters
        ----------
        X : np.ndarray
            Training feature matrix with shape ``(n_samples, n_features)``.
            Values should be numeric, preferably scaled floats.
        y : np.ndarray
            Binary target array with shape ``(n_samples,)``. Values should be
            encoded as ``0`` and ``1``.

        Returns
        -------
        MiniBatchSGDLogisticRegression
            The fitted model instance.

        Notes
        -----
        After fitting:
        - ``weights`` has shape ``(n_features,)`` and dtype float.
        - ``bias`` is a float.
        - ``loss_history`` contains one float per epoch.
        """
        n_samples, n_features = X.shape

        rng = np.random.default_rng(self.random_state)

        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.loss_history = []

        for epoch in range(self.n_epochs):
            shuffled_indices = rng.permutation(n_samples)
            X_shuffled = X[shuffled_indices]
            y_shuffled = y[shuffled_indices]

            for start_idx in range(0, n_samples, self.batch_size):
                end_idx = start_idx + self.batch_size

                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]

                batch_size_actual = len(y_batch)
                linear_output = np.dot(X_batch, self.weights) + self.bias
                y_pred = self._sigmoid(linear_output)

                error = y_pred - y_batch

                dW = np.dot(X_batch.T, error) / batch_size_actual
                db = np.mean(error)

                self.weights -= self.learning_rate * dW
                self.bias -= self.learning_rate * db

            full_train_probs = self._sigmoid(
                np.dot(X, self.weights) + self.bias
            )

            epoch_loss = self._binary_cross_entropy(
                y,
                full_train_probs,
            )

            self.loss_history.append(epoch_loss)

        return self

    def predict_proba(self, X):
        """Predict probabilities for the positive class.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix with shape ``(n_samples, n_features)`` and numeric
            dtype.

        Returns
        -------
        np.ndarray
            Predicted probabilities for class ``1`` with shape
            ``(n_samples,)`` and float dtype.
        """
        return self._sigmoid(
            np.dot(X, self.weights) + self.bias
        )

    def predict(self, X):
        """Predict binary class labels.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix with shape ``(n_samples, n_features)`` and numeric
            dtype.

        Returns
        -------
        np.ndarray
            Predicted labels with shape ``(n_samples,)`` and integer dtype.
            Values are ``0`` when probability is below ``0.5`` and ``1`` when
            probability is at least ``0.5``.
        """
        probabilities = self.predict_proba(X)
        return (probabilities >= 0.5).astype(int)
