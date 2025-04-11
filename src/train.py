import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from ffnn import FFNN

# Fungsi untuk menghitung akurasi
def calculate_accuracy(y_true, y_pred):
    y_true_labels = np.argmax(y_true, axis=1)
    y_pred_labels = np.argmax(y_pred, axis=1)
    accuracy = np.mean(y_true_labels == y_pred_labels)
    return accuracy

# Load dataset
print("Loading MNIST dataset...")
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X = mnist['data']
y = mnist['target'].astype(int)

# Pakai sebagian kecil data biar cepat training
X = X[:5000] / 255.0
y = y[:5000]

encoder = OneHotEncoder(sparse_output=False)
y = encoder.fit_transform(y.reshape(-1, 1))

# Split data 
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Model FFNN
model = FFNN(
    layer_sizes=[784, 64, 10],
    activations=['relu', 'softmax'],
    loss_function='mse',
    init_method='normal',
    regularization=None,
    seed=42
)

epochs = 10
learning_rate = 0.01
batch_size = 32

print("Start training...")
for epoch in range(epochs):
    indices = np.arange(X_train.shape[0])
    np.random.shuffle(indices)
    X_train = X_train[indices]
    y_train = y_train[indices]

    for start in range(0, X_train.shape[0], batch_size):
        end = start + batch_size
        X_batch = X_train[start:end]
        y_batch = y_train[start:end]
        
        output = model.forward(X_batch)
        grads_w, grads_b = model.backward(X_batch, y_batch)
        for i in range(len(model.weights)):
            if model.regularization == 'l2':
                model.weights[i] -= learning_rate * (grads_w[i] + model.reg_lambda * model.weights[i])
            elif model.regularization == 'l1':
                model.weights[i] -= learning_rate * (grads_w[i] + model.reg_lambda * np.sign(model.weights[i]))
            else:
                model.weights[i] -= learning_rate * grads_w[i]
                
            model.biases[i] -= learning_rate * grads_b[i]

    train_pred = model.forward(X_train)
    val_pred = model.forward(X_val)

    # Hitung loss
    train_loss = model.compute_loss(y_train, train_pred)
    val_loss = model.compute_loss(y_val, val_pred)

    # Hitung akurasi
    train_acc = calculate_accuracy(y_train, train_pred)
    val_acc = calculate_accuracy(y_val, val_pred)

    print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Train Acc: {train_acc:.4f} - Val Acc: {val_acc:.4f}")
