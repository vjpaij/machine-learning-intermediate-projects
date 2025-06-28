### Description:

A Variational Autoencoder (VAE) is a type of generative neural network that learns to encode input data into a latent space and then decode it back to reconstruct the original input. VAEs are powerful for dimensionality reduction, image generation, and anomaly detection. In this project, we build a simple VAE using Keras and apply it to the MNIST dataset.

- Builds a Variational Autoencoder for MNIST data
- Learns probabilistic latent variables for generative modeling
- Visualizes the latent space to show clustering of digits

# Variational Autoencoder (VAE) on MNIST - Explanation and Results

This project demonstrates a Variational Autoencoder (VAE) implemented in PyTorch, trained on the MNIST dataset of handwritten digits. Below is a detailed explanation of the code, the reasoning behind its components, and the interpretation of the results.

## 📦 Dependencies and Dataset

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
```

### Key Points:

* **Torch/TorchVision**: Used for building the neural network and loading the MNIST dataset.
* **Matplotlib**: For visualizing the latent space.

## ⚙️ Device Configuration

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

* Uses GPU if available for faster computation.

## 📊 Data Preprocessing

```python
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1))  # Flatten 28x28 to 784
])
```

* Transforms each image to a 784-dimensional vector.

```python
train_dataset = datasets.MNIST(...)
test_dataset = datasets.MNIST(...)
train_loader = DataLoader(...)
test_loader = DataLoader(...)
```

* Loads and batches the MNIST dataset.

## 🧠 VAE Model Architecture

```python
class VAE(nn.Module):
```

### Encoder:

```python
self.fc1 = nn.Linear(input_dim, hidden_dim)
self.fc_mu = nn.Linear(hidden_dim, latent_dim)
self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
```

* Projects input to latent space.
* **mu** and **logvar** define the Gaussian distribution.

### Decoder:

```python
self.fc2 = nn.Linear(latent_dim, hidden_dim)
self.fc3 = nn.Linear(hidden_dim, input_dim)
```

* Reconstructs the image from a sampled latent vector.

### Key Methods:

* **encode**: Maps input to mean and log-variance.
* **reparameterize**: Samples z using the reparameterization trick.
* **decode**: Generates output from z.
* **forward**: Integrates the above.

## 📉 Loss Function

```python
def vae_loss(x, x_recon, mu, logvar):
    BCE = F.binary_cross_entropy(x_recon, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD
```

### Components:

* **BCE (Reconstruction Loss)**: Measures pixel-wise error.
* **KLD (KL Divergence)**: Regularizes latent space toward standard normal distribution.

## 🔁 Training the VAE

```python
for epoch in range(1, 31):
    ...
    loss.backward()
    optimizer.step()
```

* Trains for 30 epochs.
* Prints average loss per data point per epoch.

## 🌌 Visualizing the Latent Space

```python
model.eval()
...
plt.scatter(z_all[:, 0], z_all[:, 1])
```

* Projects test images into the 2D latent space.
* Shows how the model organizes digits in a meaningful way.

## 📈 Result Interpretation

* **Loss**: Combines reconstruction accuracy and regularization.
* **Latent Space Plot**: Each dot represents a digit’s encoded location. Clear clustering indicates that the VAE learned a useful representation.

## 🧪 Use Cases and Predictions

* The model can generate new digits by sampling z \~ N(0, 1) and passing through the decoder.
* Can be extended for anomaly detection or data generation.

---

## ✅ Summary

This VAE implementation trains a probabilistic model that learns to encode MNIST digits into a structured 2D latent space. The decoder can generate similar images from sampled latent vectors, making this useful for generative tasks and unsupervised representation learning.
