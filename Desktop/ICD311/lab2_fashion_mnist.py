import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.utils.data as data

# 1. Define Transform (Requirement: Use ToTensor)
# ToTensor converts the PIL image to a PyTorch tensor and scales pixels to [0, 1]
transform = transforms.Compose([transforms.ToTensor()])

# 2. Load the Dataset (Requirement: Use torchvision.datasets.FashionMNIST)
# download=True will download the data to a folder named 'data' if not present
full_train_dataset = torchvision.datasets.FashionMNIST(
    root='./data', 
    train=True, 
    download=True, 
    transform=transform
)

# 3. Split into Training and Validation (Requirement: Split training data)
# We will use 80% for training and 20% for validation
train_size = int(0.8 * len(full_train_dataset))
val_size = len(full_train_dataset) - train_size

train_dataset, val_dataset = data.random_split(full_train_dataset, [train_size, val_size])

print(f"Training set size: {len(train_dataset)}")
print(f"Validation set size: {len(val_dataset)}")

# 4. Define Class Names (Standard Fashion-MNIST classes)
class_names = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

# 5. Display Examples (Requirement: Display examples with correct class names)
def display_images(dataset, num_images=5):
    plt.figure(figsize=(15, 3))
    for i in range(num_images):
        # Get the image and label from the dataset
        image, label = dataset[i]
        
        # Matplotlib expects the channel dimension last, PyTorch puts it first
        # Change shape from (1, 28, 28) to (28, 28)
        image = image.squeeze()
        
        plt.subplot(1, num_images, i + 1)
        plt.imshow(image, cmap='gray')
        plt.title(class_names[label])
        plt.axis('off')
    plt.show()

print("Displaying examples from the Validation Set...")
display_images(val_dataset, num_images=5)
