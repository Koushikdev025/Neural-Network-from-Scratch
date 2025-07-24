import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Simple Neural Network for Image Reconstruction


class ImageReconstructor(nn.Module):
    def __init__(self):
        super(ImageReconstructor, self).__init__()
        # Very simple architecture - just fully connected layers
        self.network = nn.Sequential(
            nn.Linear(1, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, 64*64*3),  # Output 64x64x3 RGB image
            nn.Sigmoid()  # Output values between 0 and 1
        )

    def forward(self, x):
        return self.network(x)

# Load and preprocess your image


def load_image(image_path):
    """Load and preprocess a 64x64 RGB image"""
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])

    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image)
    return image_tensor

# Convert tensor to displayable image


def tensor_to_image(tensor):
    """Convert tensor back to displayable image"""
    tensor = tensor.clamp(0, 1)
    tensor = tensor.view(3, 64, 64)
    tensor = tensor.permute(1, 2, 0)
    return tensor.cpu().detach().numpy()

# Training function for dual images


def train_dual_model(image1_path, image2_path, epochs=10, checkpoint_interval=1):
    # Load both target images
    target_image1 = load_image(image1_path).flatten().to(
        device)  # Flatten to 1D
    target_image2 = load_image(image2_path).flatten().to(
        device)  # Flatten to 1D

    # Create model
    model = ImageReconstructor().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # Dummy input (same input for both images)
    dummy_input = torch.tensor([[1.0]], dtype=torch.float32).to(device)

    losses = []
    losses_img1 = []
    losses_img2 = []

    print("Starting dual image training...")
    print("The network will try to find a compromise between both images with the same input!")

    for epoch in range(epochs):
        optimizer.zero_grad()

        # Forward pass
        output = model(dummy_input).flatten()

        # Calculate loss for both images (the network tries to satisfy both)
        loss1 = criterion(output, target_image1)
        loss2 = criterion(output, target_image2)

        # Combined loss - you could experiment with different weightings
        total_loss = loss1 + loss2  # Equal weight to both images

        # Backward pass
        total_loss.backward()
        optimizer.step()

        losses.append(total_loss.item())
        losses_img1.append(loss1.item())
        losses_img2.append(loss2.item())

        # Print progress and show images at checkpoints
        if (epoch + 1) % checkpoint_interval == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"  Total Loss: {total_loss.item():.6f}")
            print(f"  Loss vs Image 1: {loss1.item():.6f}")
            print(f"  Loss vs Image 2: {loss2.item():.6f}")

            # Generate and display current reconstruction
            with torch.no_grad():
                reconstructed = model(dummy_input).flatten()
                reconstructed_image = tensor_to_image(reconstructed)

                # Display all three images side by side
                fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))

                # Original image 1
                original_image1 = tensor_to_image(target_image1)
                ax1.imshow(original_image1)
                ax1.set_title('Original Image 1')
                ax1.axis('off')

                # Original image 2
                original_image2 = tensor_to_image(target_image2)
                ax2.imshow(original_image2)
                ax2.set_title('Original Image 2')
                ax2.axis('off')

                # Reconstructed image (compromise)
                ax3.imshow(reconstructed_image)
                ax3.set_title(f'Network Output (Epoch {epoch+1})')
                ax3.axis('off')

                plt.tight_layout()
                plt.show()

    # Plot loss curves
    plt.figure(figsize=(12, 5))

    # Combined loss plot
    plt.subplot(1, 2, 1)
    plt.plot(losses, label='Total Loss', linewidth=2)
    plt.plot(losses_img1, label='Loss vs Image 1', alpha=0.7)
    plt.plot(losses_img2, label='Loss vs Image 2', alpha=0.7)
    plt.title('Training Losses Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)

    # Difference in losses
    plt.subplot(1, 2, 2)
    loss_diff = [abs(l1 - l2) for l1, l2 in zip(losses_img1, losses_img2)]
    plt.plot(loss_diff, color='red', linewidth=2)
    plt.title('Absolute Difference in Individual Losses')
    plt.xlabel('Epoch')
    plt.ylabel('|Loss1 - Loss2|')
    plt.yscale('log')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    return model

# Training function for a single image


def train_single_model(image_path, epochs=10, checkpoint_interval=1):
    # Load the target image
    target_image = load_image(image_path).flatten().to(device)  # Flatten to 1D

    # Create model
    model = ImageReconstructor().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # Dummy input
    dummy_input = torch.tensor([[1.0]], dtype=torch.float32).to(device)

    losses = []

    print("Starting single image training...")
    print("The network will try to reconstruct the single target image!")

    for epoch in range(epochs):
        optimizer.zero_grad()

        # Forward pass
        output = model(dummy_input).flatten()

        # Calculate loss
        loss = criterion(output, target_image)

        # Backward pass
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        # Print progress and show images at checkpoints
        if (epoch + 1) % checkpoint_interval == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")

            # Generate and display current reconstruction
            with torch.no_grad():
                reconstructed = model(dummy_input).flatten()
                reconstructed_image = tensor_to_image(reconstructed)

                # Display original and reconstructed images side by side
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

                # Original image
                original_image = tensor_to_image(target_image)
                ax1.imshow(original_image)
                ax1.set_title('Original Image')
                ax1.axis('off')

                # Reconstructed image
                ax2.imshow(reconstructed_image)
                ax2.set_title(f'Network Output (Epoch {epoch+1})')
                ax2.axis('off')

                plt.tight_layout()
                plt.show()

    # Plot loss curve
    plt.figure(figsize=(6, 4))
    plt.plot(losses, label='MSE Loss', linewidth=2)
    plt.title('Training Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    plt.show()

    return model


# --- Usage example for dual image training ---
# TODO: Replace with your actual image paths after uploading your images.
image_path_1 = '/content/image.png'  # Example path
image_path_2 = '/content/image2.png'  # Example path
# model_dual = train_dual_model(image_path_1, image_path_2, epochs=50, checkpoint_interval=5)

# Final comparison - show what the network learned
# print("\n" + "="*50)
# print("FINAL RESULT ANALYSIS (Dual Image)")
# print("="*50)

# with torch.no_grad():
#     dummy_input = torch.tensor([[1.0]], dtype=torch.float32).to(device)
#     final_output_dual = model_dual(dummy_input).flatten()
#     final_image_dual = tensor_to_image(final_output_dual)

#     # TODO: Replace with your actual image paths after uploading your images.
#     img1 = tensor_to_image(load_image('/content/image.png').flatten().to(device)) # Example path
#     img2 = tensor_to_image(load_image('/content/image2.png').flatten().to(device)) # Example path


#     # Create final comparison
#     fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 10))

#     ax1.imshow(img1)
#     ax1.set_title('Original Image 1')
#     ax1.axis('off')

#     ax2.imshow(img2)
#     ax2.set_title('Original Image 2')
#     ax2.axis('off')

#     ax3.imshow(final_image_dual)
#     ax3.set_title('Network\'s Final Output\n(Attempted Compromise)')
#     ax3.axis('off')

#     # Show pixel-wise average for comparison
#     avg_image = (img1 + img2) / 2
#     ax4.imshow(avg_image)
#     ax4.set_title('Simple Pixel Average\n(For Comparison)')
#     ax4.axis('off')

#     plt.suptitle('Dual Image Training Results', fontsize=16)
#     plt.tight_layout()
#     plt.show()

# print("\nDual image experiment complete!")
# print("The network tried to find a single output that minimizes error to BOTH images.")
# print("This is an impossible task, so it found some kind of compromise solution.")


# --- Usage example for single image training ---
# TODO: Replace with your actual image path after uploading your image.
single_image_path = 'image.jpg'  # Example path
model_single = train_single_model(
    single_image_path, epochs=50, checkpoint_interval=5)

# Final comparison for single image training
print("\n" + "="*50)
print("FINAL RESULT ANALYSIS (Single Image)")
print("="*50)

with torch.no_grad():
    dummy_input = torch.tensor([[1.0]], dtype=torch.float32).to(device)
    final_output_single = model_single(dummy_input).flatten()
    final_image_single = tensor_to_image(final_output_single)

    # TODO: Replace with your actual image path after uploading your image.
    img_original_single = tensor_to_image(load_image(
        'image.jpg').flatten().to(device))  # Example path

    # Create final comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

    ax1.imshow(img_original_single)
    ax1.set_title('Original Image')
    ax1.axis('off')

    ax2.imshow(final_image_single)
    ax2.set_title('Network\'s Final Output')
    ax2.axis('off')

    plt.suptitle('Single Image Training Results', fontsize=14)
    plt.tight_layout()
    plt.show()

print("\nSingle image experiment complete!")
print("The network tried to reconstruct the single target image.")

