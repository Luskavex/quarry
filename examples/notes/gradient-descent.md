# Gradient descent

Gradient descent minimizes a loss function by repeatedly stepping in the direction of the negative
gradient. Each update is `w = w - learning_rate * gradient`. The gradient points uphill, so moving
against it lowers the loss, at least for a small enough step.

## Learning rate

The learning rate is the size of each step. If it is too large, the loss oscillates or diverges
because every step overshoots the minimum. If it is too small, training is stable but very slow.
A common practice is to start with a value such as 0.1 or 0.01 and watch the loss curve.

Learning rate schedules reduce the step size during training. Step decay, cosine annealing and
warm-up followed by decay are popular choices for deep networks.

## Feature scaling

Gradient descent converges much faster when features have similar scales. With one feature in the
thousands and another between 0 and 1, the loss surface becomes a long narrow valley and the
updates zig-zag across it. Standardizing every feature to zero mean and unit variance fixes this.

## Variants

Batch gradient descent uses the whole dataset for every update. Stochastic gradient descent (SGD)
uses a single example, which is noisy but cheap. Mini-batch gradient descent sits in between and is
what most deep learning frameworks use. Momentum and Adam add a running average of past gradients
so the optimizer keeps moving through flat regions and small bumps.
