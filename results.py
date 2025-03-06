import os
import csv
import argparse
from PIL import Image
import numpy as np
import io
import cv2
from jpeg_encoder import serialize_to_jpeg
from jpeg_decoder import JpegDecoder

'''
python3 results.py Pictures/building.jpeg true ./results ./results/image_quality_data.csv
'''

# Argument parser setup
parser = argparse.ArgumentParser(description='JPEG Encoding and Decoding Script')
parser.add_argument('image_path', type=str, help='Path to the original image')
parser.add_argument('convert_to_grayscale', type=str, help='Convert to grayscale (true/false)')
parser.add_argument('results_dir', type=str, help='Directory to save results')
parser.add_argument('csv_file_path', type=str, help='Path to save CSV file')

args = parser.parse_args()

# Validate arguments
if not os.path.isfile(args.image_path):
    raise Exception(f"Invalid image path: {args.image_path}")

if args.convert_to_grayscale.lower() not in ['true', 'false']:
    raise Exception(f"Invalid value for convert_to_grayscale: {args.convert_to_grayscale}. Must be 'true' or 'false'.")

# Assign arguments to variables
ORIGINAL_IMAGE_PATH = args.image_path
CONVERT_TO_GRAYSCALE = args.convert_to_grayscale.lower() == 'true'
RESULTS_DIR = args.results_dir
CSV_FILE_PATH = args.csv_file_path

# Ensure results directory exists
os.makedirs(RESULTS_DIR, exist_ok=True)

class ImageData:
    def __init__(self, data, width, height, kind):
        self.data = data
        self.width = width
        self.height = height
        self.kind = kind
        self.n = 1 if kind == 'g' else 3  # 1 channel for grayscale, 3 for RGB

def prepare_image(filepath):
    img = Image.open(filepath)
    original_size_kb = os.path.getsize(filepath) / 1024  # Size in KB
    if CONVERT_TO_GRAYSCALE:
        img = img.convert('L')
        data = np.array(img).flatten()
        kind = 'g'
    else:
        img = img.convert('RGB')
        data = np.array(img).flatten()
        kind = 'rgb'
    return ImageData(data.tobytes(), img.width, img.height, kind), original_size_kb

def decode_image(encoded_data):
    decoded_jpeg = JpegDecoder(encoded_data)
    image_data = decoded_jpeg.decompress_image()
    mode = 'L' if decoded_jpeg.kind == 'g' else 'RGB'
    return Image.frombytes(mode, (decoded_jpeg.width, decoded_jpeg.height), bytes(image_data))

def add_quality_factor_bar(image, quality_factor):
    img_cv2 = np.array(image)
    if img_cv2.ndim == 2:  # Grayscale image
        img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_GRAY2BGR)
    elif img_cv2.ndim == 3:
        img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_RGB2BGR)
    bar_height = 200  # Increased bar height to 200 pixels
    height, width = img_cv2.shape[:2]
    new_img = np.ones((height + bar_height, width, 3), dtype=np.uint8) * 255  # White bar
    
    # Place the original image above the bar
    new_img[:height, :width] = img_cv2
    
    # Add text with the quality factor
    text = f"Q = {quality_factor}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 4 
    thickness = 7 
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = (width - text_size[0]) // 2
    text_y = height + (bar_height + text_size[1]) // 2
    cv2.putText(new_img, text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness, lineType=cv2.LINE_AA)
    return Image.fromarray(cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB))

def get_image_size_in_kb(image):
    with io.BytesIO() as buffer:
        image.save(buffer, format='PNG')  # Save to buffer as PNG or other formats for accurate size estimation
        size_kb = len(buffer.getvalue()) / 1024  # Size in KB
    return round(size_kb, 2)

def main():
    image, original_size_kb = prepare_image(ORIGINAL_IMAGE_PATH)

    # Create CSV file and write header
    with open(CSV_FILE_PATH, mode='w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["QualityFactor", "Encoded Size (KB)", "Decoded Size (KB)", "Original Size (KB)"])

        for q in range(1, 101):
            jpeg_data = serialize_to_jpeg(image, q)
            encoded_size_kb = round(len(jpeg_data) / 1024, 2)  # Convert and round size to KB
            original_size_kb_rounded = round(original_size_kb, 2)

            # Decode the image
            decoded_image = decode_image(jpeg_data)
            result_image = add_quality_factor_bar(decoded_image, q)
            result_image.save(os.path.join(RESULTS_DIR, f"decoded_image_q{q}.jpg"))

            # Calculate decoded size
            decoded_size_kb = get_image_size_in_kb(decoded_image)

            # Write data to CSV with rounded values
            csv_writer.writerow([q, encoded_size_kb, decoded_size_kb, original_size_kb_rounded])

if __name__ == "__main__":
    main()
