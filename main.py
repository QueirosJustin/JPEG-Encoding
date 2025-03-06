# Main script: encode_and_decode_image.py

import argparse
from PIL import Image
import numpy as np
import os
from jpeg_encoder import serialize_to_jpeg
from jpeg_decoder import JpegDecoder

'''
Example usage: python3 main.py Pictures/building.jpeg true 50
'''

# Argument parser setup
parser = argparse.ArgumentParser(description='JPEG Encoding and Decoding Script')
parser.add_argument('image_path', type=str, help='Path to the original image')
parser.add_argument('convert_to_grayscale', type=str, help='Convert to grayscale (true/false)')
parser.add_argument('quality_factor', type=int, help='Quality factor for JPEG compression (1-100)')

args = parser.parse_args()

# Validate arguments
if not os.path.isfile(args.image_path):
    raise Exception(f"Invalid image path: {args.image_path}")

if args.convert_to_grayscale.lower() not in ['true', 'false']:
    raise Exception(f"Invalid value for convert_to_grayscale: {args.convert_to_grayscale}. Must be 'true' or 'false'.")

if not (1 <= args.quality_factor <= 100):
    raise Exception(f"Invalid quality factor: {args.quality_factor}. Must be between 1 and 100.")

# Assign arguments to variables
ORIGINAL_IMAGE_PATH = args.image_path
CONVERT_TO_GRAYSCALE = args.convert_to_grayscale.lower() == 'true'
QUALITY_FACTOR = args.quality_factor


class ImageData:
    def __init__(self, data, width, height, kind):
        self.data = data
        self.width = width
        self.height = height
        self.kind = kind
        self.n = 1 if kind == 'g' else 3  # 1 channel for grayscale, 3 for RGB

def prepare_image(filepath):
    # Open the image
    img = Image.open(filepath)
    
    # Convert image based on the global grayscale parameter
    if CONVERT_TO_GRAYSCALE:
        img = img.convert('L')  # Grayscale mode
        data = np.array(img).flatten()  # Flatten 2D grayscale array to 1D
        kind = 'g'  # Grayscale identifier
    else:
        img = img.convert('RGB')  # Ensure RGB mode
        data = np.array(img).flatten()  # Flatten 3D RGB array to 1D
        kind = 'rgb'  # RGB identifier
    
    return ImageData(data.tobytes(), img.width, img.height, kind)

def decode_image(encoded_path):
    # Read the encoded JPEG data
    with open(encoded_path, "rb") as f:
        jpeg_data = f.read()
    
    # Decode the JPEG data back to image data
    decoded_jpeg = JpegDecoder(jpeg_data)
    image_data = decoded_jpeg.decompress_image()

    # Set mode based on the decoded image kind
    mode = 'L' if decoded_jpeg.kind == 'g' else 'RGB'
    decoded_image = Image.frombytes(mode, (decoded_jpeg.width, decoded_jpeg.height), bytes(image_data))
    decoded_image_path = "decoded_image.png"
    decoded_image.save(decoded_image_path)
    
    # Display the size of the decoded image
    decoded_size = os.path.getsize(decoded_image_path)
    print(f"Decoded Image Size: {decoded_size / 1024:.2f} KB")

    # Display the decoded image
    decoded_image.show()

# Prepare the image based on the grayscale parameter
image = prepare_image(ORIGINAL_IMAGE_PATH)

# Compress and encode image data directly
jpeg_data = serialize_to_jpeg(image, QUALITY_FACTOR)

# Save the encoded data as a JPEG file
encoded_image_path = "encoded_image.jpg"
with open(encoded_image_path, "wb") as f:
    f.write(jpeg_data)

# Display original and encoded image sizes
original_size = os.path.getsize(ORIGINAL_IMAGE_PATH)
encoded_size = os.path.getsize(encoded_image_path)

print(f"Original Image Size: {original_size / 1024:.2f} KB")
print(f"Encoded Image Size: {encoded_size / 1024:.2f} KB")

# Decode and display the encoded image
decode_image(encoded_image_path)
