import matplotlib.pyplot as plt
import cv2
import numpy as np

# Load the image
image_path = 'Au_3_BSD_50kx_A.tif'
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Check if the image was loaded successfully
if img is None:
    raise FileNotFoundError(f"The image at path '{image_path}' could not be loaded. Please check the file path and ensure the file exists.")

# Display the image
plt.imshow(img, cmap='gray')
plt.title('Click the two ends of the scale bar')
pts = plt.ginput(2)  # Click two points, then close the window
plt.close()

# Calculate pixel distance
(x1, y1), (x2, y2) = pts
scale_bar_pixels = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
print(f"Scale bar length in pixels: {scale_bar_pixels:.2f}")

# Now calculate pixel size
scale_bar_nm = 200  # known from the image
pixel_size = scale_bar_nm / scale_bar_pixels
print(f"Pixel size: {pixel_size:.2f} nm/pixel")