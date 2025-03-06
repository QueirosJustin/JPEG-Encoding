# JPEG-Encoding

This repository contains my implementation of a **JPEG-like image compression engine** in Python for **learning demonstrations**. The project covers the full compression and decompression pipeline, including:

- **2D Discrete Cosine Transform (DCT) & Quantization** – Frequency-domain transformation to reduce redundancy.
- **Huffman Encoding** – Entropy compression for efficient data storage.
- **Image Reconstruction** – Using inverse quantization and IDCT to restore images.
- **Support for Grayscale & Color Images** – Converts RGB images to the YUV color space for better compression.
- **Configurable Quality Factor** – Adjusts compression levels to balance file size and image quality.

## Installation & Dependencies

Ensure you have Python installed and install the required dependencies using:

```bash
pip install -r requirements.txt
```

## Usage

### Compress a Single Image

To compress an image using **`main.py`**, run:

```bash
python3 main.py <image_path> <grayscale> <quality_factor>
```

- `<image_path>`: Path to the image file (e.g., `Pictures/building.jpeg`).
- `<grayscale>`: Boolean (`true` or `false`) to indicate if the image should be converted to grayscale.
- `<quality_factor>`: Integer (1-100) to control the compression level (higher means better quality, larger file size).

#### Example:

```bash
python3 main.py Pictures/building.jpeg true 50
```

This compresses `building.jpeg` as a grayscale image with a quality factor of **50**.

---

### Evaluate Compression Quality

To visualize compression results across **all quality factors (1-100)** and save data to a CSV file, use **`results.py`**:

```bash
python3 results.py <image_path> <grayscale> <output_directory> <csv_output>
```

- `<image_path>`: Path to the image file.
- `<grayscale>`: Boolean (`true` or `false`) to convert the image to grayscale.
- `<output_directory>`: Folder where the results will be stored.
- `<csv_output>`: Path to save the CSV file containing size comparisons.

#### Example:

```bash
python3 results.py Pictures/building.jpeg true ./results ./results/image_quality_data.csv
```

This will generate visualizations and save a **CSV file** containing data such as:

| Quality Factor | Encoded Size (KB) | Decoded Size (KB) | Original Size (KB) |
|---------------|------------------|------------------|------------------|
| 10           | X KB             | Y KB             | Z KB             |
| 50           | X KB             | Y KB             | Z KB             |
| 100          | X KB             | Y KB             | Z KB             |

---

## Project Structure

```
│── main.py             # Main script for JPEG compression
│── results.py          # Generates visualizations and CSV reports
│── huffman.py          # Huffman encoding and decoding logic
│── dct.py              # Implements DCT and inverse DCT
│── quantization.py     # Quantization functions
│── utils.py            # Helper functions
│── requirements.txt    # Dependencies
│── README.md           # This documentation
│── Pictures/           # Folder for input images
│── results/            # Folder for output images & CSV results
```

## Features & Future Improvements

✔️ **Current Features**:
- Full JPEG compression pipeline.
- Adjustable quality factor.
- Grayscale & color image support.
- Huffman-based lossless encoding.
- CSV-based storage of size comparisons.

🚀 **Planned Improvements**:
- None, feel free to adjust the code yourself

---

## License

This project is **open-source** and available under the [MIT License](LICENSE).
