pip install opencv-python numpy
import cv2
import numpy as np

# Function to compute Dark Channel
def dark_channel(image, window_size=15):
    # Convert the image to RGB if it's BGR
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Compute the dark channel for each pixel
    min_channels = np.min(image, axis=2)
    dark_channel = cv2.erode(min_channels, np.ones((window_size, window_size)))
    return dark_channel

# Estimate the atmospheric light
def estimate_atmospheric_light(image, dark_channel):
    # Get the top 0.1% brightest pixels in the dark channel
    num_pixels = dark_channel.size
    num_brightest_pixels = int(num_pixels * 0.001)

    # Flatten the dark channel and the image to get the brightest pixels
    flat_dark = dark_channel.ravel()
    flat_image = image.reshape(-1, 3)
    
    # Sort the dark channel pixels in descending order
    dark_channel_sorted_indices = np.argsort(flat_dark)[::-1]

    # Get the atmospheric light based on the brightest pixels in dark channel
    brightest_pixels = flat_image[dark_channel_sorted_indices[:num_brightest_pixels]]
    atmospheric_light = np.mean(brightest_pixels, axis=0)

    return atmospheric_light

# Recover the transmission map
def estimate_transmission(image, atmospheric_light, window_size=15, omega=0.95):
    # Normalize the image by atmospheric light
    norm_image = image / atmospheric_light
    dark_channel = dark_channel(norm_image, window_size)

    # Compute transmission using the formula: T = 1 - omega * DarkChannel
    transmission = 1 - omega * dark_channel
    return transmission

# Dehaze the image using the transmission and atmospheric light
def dehaze(image, atmospheric_light, transmission, t0=0.1):
    # Apply the transmission map
    transmission = np.clip(transmission, t0, 1)
    dehazed_image = (image - atmospheric_light) / transmission + atmospheric_light
    dehazed_image = np.clip(dehazed_image, 0, 255).astype(np.uint8)
    return dehazed_image

# Histogram Equalization to improve contrast
def histogram_equalization(image):
    # Convert image to YUV color space
    yuv_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    
    # Apply histogram equalization on the Y channel
    yuv_image[:, :, 0] = cv2.equalizeHist(yuv_image[:, :, 0])
    
    # Convert back to BGR color space
    equalized_image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR)
    return equalized_image

# Main function to perform dehazing and histogram equalization
def dehaze_and_equalize(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Step 1: Apply Dark Channel Prior Dehazing
    dark_channel_img = dark_channel(image)
    atmospheric_light = estimate_atmospheric_light(image, dark_channel_img)
    transmission_map = estimate_transmission(image, atmospheric_light)
    dehazed_image = dehaze(image, atmospheric_light, transmission_map)

    # Step 2: Apply Histogram Equalization for better contrast
    result_image = histogram_equalization(dehazed_image)

    return result_image

# Example usage
if __name__ == "__main__":
    # Path to the hazy image
    image_path = "hazy_image.jpg"
    
    # Dehaze and equalize the image
    result = dehaze_and_equalize(image_path)

    # Show the result
    cv2.imshow("Dehazed and Equalized Image", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save the result
    cv2.imwrite("dehazed_equalized_image.jpg", result)
