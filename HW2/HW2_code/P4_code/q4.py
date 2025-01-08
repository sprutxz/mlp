import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

# Section 4.1
def load_images(image_folder):
    # Load the images
    image_files = os.listdir(image_folder)
    images = []
    for image_file in image_files:
        image = Image.open(os.path.join(image_folder, image_file))
        images.append(np.array(image))
    return images

def preprocess_images(images):
    # Preprocess the images
    images = np.array(images)
    images = images / 255.0
    images = images.reshape(images.shape[0], -1)
    return images


images = load_images('/home/sprutz/dev/mlp/HW2/HW2_data/P4_data/train') 
images = preprocess_images(images)

mean_image = np.mean(images, axis=0)

cov_matrix = np.cov(images - mean_image, rowvar=False)

evals, evecs = np.linalg.eigh(cov_matrix)

# Sort the eigenvectors by decreasing eigenvalues
idx = np.argsort(evals)[::-1]
evecs = evecs[:, idx]
evals = evals[idx]

print(f"evals.shape: {evals.shape}")
print(f"evecs.shape: {evecs.shape}")

# Section 4.2
M = [2, 10, 100, 1000, 4000]

test_img = load_images('../../HW2_data/P4_data/test')[2]
print(f"test_img.shape: {test_img.shape}")
img = preprocess_images([test_img])

print(f"img.shape: {img.shape}")
img = img.reshape(-1)
mean_image = mean_image.reshape(-1)

print(f"img.shape: {img.shape}")

for m in M:
    projection = evecs[:, :m].T @ (img - mean_image)
    img_reconstructed = mean_image + evecs[:, :m] @ projection
    
    img_reconstructed = img_reconstructed.reshape(60, 80)
    plt.imshow(img_reconstructed, cmap='gray')
    plt.title(f"Reconstructed image using {m} eigenvectors")
    plt.show()
    plt.imsave(f"../../images/reconst(M={m}).png", img_reconstructed, cmap='gray')
    
# Section 4.3

#normalize the eigenvectors
eigenfaces = evecs[:, :10]
emin = np.min(eigenfaces)
emax = np.max(eigenfaces)

eigenfaces = ((eigenfaces - emin) / (emax - emin)) * 256

fig, axs = plt.subplots(2, 5, figsize=(20, 10))
for i in range(2):
    for j in range(5):
        idx = i * 5 + j
        axs[i, j].imshow(eigenfaces[:, idx].reshape(60, 80), cmap='gray')
        axs[i, j].axis('off')
        axs[i, j].set_title(f"Eigenface {idx + 1}")
plt.show()