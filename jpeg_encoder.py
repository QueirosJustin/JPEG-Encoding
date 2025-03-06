import Modules.utils as utils
from struct import pack

class EntropyEncoder(object):
    """
    Class to perform entropy encoding of image data using Huffman coding.
    """
    def __init__(self):
        # Precompute codes and sizes for faster encoding
        self.codes = [value for shift in range(15, -1, -1) for value in range(1 << shift)]
        sizes = [size for size in range(1, 16) for _ in range(1 << (size - 1))]
        self.sizes = [0] + sizes + sizes[::-1]  # Include sizes and reversed sizes for efficient lookups
        
        self.value = 0  # Current bit buffer
        self.length = 0  # Current length of bits in the buffer
        self.data = bytearray()  # Encoded output data

    def encode_block(self, previous_dc, coefficient_block, quantization_scale, dc_huffman_table, ac_huffman_table):
        """
        Encodes a single 8x8 block of coefficients using Huffman encoding.
        """
        # Step 1: Perform a forward DCT on the block of coefficients
        utils.perform_forward_dct(coefficient_block)
        
        # Step 2: Quantize the DCT coefficients
        for position in range(64):
            # The quantization formula: ((value * 2) / quantization_scale + 1) / 2
            coefficient_block[position] = (((coefficient_block[position] << 1) // quantization_scale[position]) + 1) >> 1
        
        # Step 3: Encode the DC coefficient
        dc_difference = coefficient_block[0] - previous_dc
        if dc_difference == 0:
            self.write_bits_to_buffer(*dc_huffman_table[0])  # Special case for zero DC difference
        else:
            dc_size = self.sizes[dc_difference]
            self.write_bits_to_buffer(*dc_huffman_table[dc_size])
            self.write_bits_to_buffer(self.codes[dc_difference], dc_size)
        
        # Step 4: Encode the AC coefficients
        zero_run = 0
        for zigzag_position in utils.zig_zag:
            if coefficient_block[zigzag_position] == 0:
                zero_run += 1
            else:
                # Handle long runs of zeros (length > 15)
                while zero_run > 15:
                    self.write_bits_to_buffer(*ac_huffman_table[0xf0])  # 0xf0 represents a run of 16 zeros
                    zero_run -= 16
                ac_size = self.sizes[coefficient_block[zigzag_position]]
                self.write_bits_to_buffer(*ac_huffman_table[zero_run * 16 + ac_size])  # Encode zero-run length and AC size
                self.write_bits_to_buffer(self.codes[coefficient_block[zigzag_position]], ac_size)  # Write the actual coefficient
                zero_run = 0
        
        # Step 5: If there are trailing zeros, encode EOB (End of Block)
        if zero_run > 0:
            self.write_bits_to_buffer(*ac_huffman_table[0])  # EOB symbol
        
        return coefficient_block[0]  # Return the new DC coefficient for the next block

    def write_bits_to_buffer(self, bits, length):
        """
        Writes bits to the internal buffer, handling byte alignment and stuffing.
        """
        buffer = self.data
        bits += (self.value << length)
        length += self.length
        
        while length > 7:
            length -= 8
            byte = (bits >> length) & 0xff
            if byte == 0xff:  # Handle special case where byte is 0xFF (stuffing byte)
                buffer.append(0xff)
                buffer.append(0)
            else:
                buffer.append(byte)
        
        self.value = bits & 0xff
        self.length = length

    def get_data(self):
        """
        Returns the encoded data as a byte array.
        """
        return self.data

def generate_quantization_table(base_table, quality_level):
    """
    Generates a quantization table based on the provided base table and quality level.
    """
    quality_level = max(0, min(quality_level, 100))
    scale_factor = 5000 // quality_level if quality_level < 50 else 200 - quality_level * 2  # Magic values: Derived from JPEG quality scaling formulas
    return bytes([max(1, min((value * scale_factor + 50) // 100, 255)) for value in base_table])

def create_huffman_table(code_lengths, symbols):
    """
    Constructs a Huffman table from code lengths and symbols.
    """
    max_symbol = max(symbols) + 1
    huffman_table = [None] * max_symbol
    current_code = 0
    index = 0
    bit_size = 1
    
    for length in code_lengths:
        for _ in range(length):
            huffman_table[symbols[index]] = current_code, bit_size
            current_code += 1
            index += 1
        current_code *= 2
        bit_size += 1
    
    return huffman_table

def compute_scale_factors(input_table):
    """
    Computes a scaling factor for quantization based on an input table.
    """
    scaling_factors = [value * 8 if index == 0 else 0 for index, value in enumerate(input_table)]  # Magic value 8: Used to scale DC coefficients
    zigzag_index = 1
    
    for position in utils.zig_zag:
        scaling_factors[position] = input_table[zigzag_index] * 8
        zigzag_index += 1
    
    return scaling_factors

def create_marker_segment(marker, data):
    """
    Creates a JPEG marker segment with the specified marker and data.
    """
    segment_length = len(data) + 2
    return b'\xff' + marker + pack('>H', segment_length) + data

def serialize_to_jpeg(image, quality):
    """
    Serializes an image into JPEG format using the specified quality level.
    """
    img_width, img_height, channels, img_data = image.width, image.height, image.n, image.data
    # Prepare luminance tables for encoding
    luminance_quant, luminance_dc_table, luminance_ac_table, luminance_scale = prepare_luminance_huffman_and_quant_tables(quality)
    # Prepare chrominance tables for encoding if image has color channels
    chrominance_quant, chrominance_dc_table, chrominance_ac_table, chrominance_scale = (None, None, None, None)
    if channels == 3:
        chrominance_quant, chrominance_dc_table, chrominance_ac_table, chrominance_scale = prepare_chrominance_tables(quality)
        
    entropy_encoder = EntropyEncoder()  # Create an entropy encoder instance
    # Serialize image blocks into entropy-encoded data
    serialize_blocks(img_width, img_height, channels, img_data, luminance_scale, luminance_dc_table, luminance_ac_table, 
                     chrominance_scale, chrominance_dc_table, chrominance_ac_table, entropy_encoder)
    
    # Construct the final JPEG data stream
    return construct_jpeg_data(channels, img_width, img_height, luminance_quant, chrominance_quant, entropy_encoder)

def prepare_luminance_huffman_and_quant_tables(quality):
    """
    Prepares quantization and Huffman tables for luminance (Y) channel based on quality.
    """
    luminance_quant = generate_quantization_table(utils.luminance_quant_table, quality)
    luminance_dc_table = create_huffman_table(utils.luminance_dc_code_lengths, utils.luminance_dc_values)
    luminance_ac_table = create_huffman_table(utils.luminance_ac_code_lengths, utils.luminance_ac_values)
    luminance_scale = compute_scale_factors(luminance_quant)
    return luminance_quant, luminance_dc_table, luminance_ac_table, luminance_scale

def prepare_chrominance_tables(quality):
    """
    Prepares quantization and Huffman tables for chrominance (U, V) channels based on quality.
    """
    chrominance_quant = generate_quantization_table(utils.chrominance_quant_table, quality)
    chrominance_dc_table = create_huffman_table(utils.chrominance_dc_code_lengths, utils.chrominance_dc_values)
    chrominance_ac_table = create_huffman_table(utils.chrominance_ac_code_lengths, utils.chrominance_ac_values)
    chrominance_scale = compute_scale_factors(chrominance_quant)
    return chrominance_quant, chrominance_dc_table, chrominance_ac_table, chrominance_scale

def serialize_blocks(img_width, img_height, channels, img_data, luminance_scale, luminance_dc_table, luminance_ac_table, 
                     chrominance_scale, chrominance_dc_table, chrominance_ac_table, entropy_encoder):
    """
    Serializes image data into 8x8 blocks for JPEG compression using the provided tables and encoder.
    """
    # Initialize previous DC coefficients for differential encoding
    y_prev_dc = u_prev_dc = v_prev_dc = k_prev_dc = 0
    # Initialize 8x8 matrices for storing transformed data
    y_matrix, u_matrix, v_matrix, k_matrix = [0]*64, [0]*64, [0]*64, [0]*64

    # Iterate over 8x8 blocks in the image
    for row in range(0, img_height, 8):
        for col in range(0, img_width, 8):
            # Process a single 8x8 block and perform color space transformations
            y_matrix, u_matrix, v_matrix, k_matrix = process_block(img_width, img_height, channels, img_data, row, col, 
                                                                   y_matrix, u_matrix, v_matrix, k_matrix)
            # Encode the luminance (Y) block
            y_prev_dc = entropy_encoder.encode_block(y_prev_dc, y_matrix, luminance_scale, luminance_dc_table, luminance_ac_table)
            if channels == 3:
                # Encode the chrominance (U, V) blocks for color images
                u_prev_dc = entropy_encoder.encode_block(u_prev_dc, u_matrix, chrominance_scale, chrominance_dc_table, chrominance_ac_table)
                v_prev_dc = entropy_encoder.encode_block(v_prev_dc, v_matrix, chrominance_scale, chrominance_dc_table, chrominance_ac_table)
            elif channels == 4:
                # Encode the additional channels for images with 4 channels
                u_prev_dc = entropy_encoder.encode_block(u_prev_dc, u_matrix, luminance_scale, luminance_dc_table, luminance_ac_table)
                v_prev_dc = entropy_encoder.encode_block(v_prev_dc, v_matrix, luminance_scale, luminance_dc_table, luminance_ac_table)
                k_prev_dc = entropy_encoder.encode_block(k_prev_dc, k_matrix, luminance_scale, luminance_dc_table, luminance_ac_table)

    entropy_encoder.write_bits_to_buffer(0x7f, 7)  # Padding bits (0x7f): Ensures byte alignment of the encoded stream

def process_block(img_width, img_height, channels, img_data, row, col, y_matrix, u_matrix, v_matrix, k_matrix):
    """
    Processes an 8x8 block of image data, converting RGB (or other formats) to YUV (or YUVK).
    """
    pixel_index = 0
    for sub_row in range(row, row + 8):
        for sub_col in range(col, col + 8):
            pixel_position = (min(sub_col, img_width - 1) + min(sub_row, img_height - 1) * img_width) * channels
            if channels == 1:
                # Grayscale image processing
                y_matrix[pixel_index] = img_data[pixel_position]
            elif channels == 3:
                # Convert RGB to YUV using integer approximation
                red, green, blue = img_data[pixel_position], img_data[pixel_position + 1], img_data[pixel_position + 2]
                # Magic values for YUV conversion derived from BT.601 standard
                y_matrix[pixel_index] = (19595 * red + 38470 * green + 7471 * blue + 32768) >> 16
                u_matrix[pixel_index] = (-11056 * red - 21712 * green + 32768 * blue + 8421376) >> 16
                v_matrix[pixel_index] = (32768 * red - 27440 * green - 5328 * blue + 8421376) >> 16
            else:
                NotImplementedError("channels are neither 1 nor 3!")
            pixel_index += 1
    return y_matrix, u_matrix, v_matrix, k_matrix

def construct_jpeg_data(channels, img_width, img_height, luminance_quant, chrominance_quant, entropy_encoder):
    """
    Constructs the final JPEG data stream, including markers and encoded data.
    """
    # Application-specific marker (Adobe): Tag, version, flags, and transform
    app_marker = b'Adobe\0\144\200\0\0\0\0'
    # Start of Frame marker (SOF0): Contains image dimensions and channel data
    sof_marker = b'\10' + pack('>HHB', img_height, img_width, channels) + b'\1\21\0'
    # Start of Scan marker (SOS): Contains channel-specific Huffman tables
    sos_marker = pack('B', channels) + b'\1\0'
    quant_table_marker = b'\0' + luminance_quant
    # Huffman table marker: Specifies DC and AC Huffman tables for luminance
    huffman_table_marker = b'\0' + utils.luminance_dc_code_lengths + utils.luminance_dc_values + b'\20' + utils.luminance_ac_code_lengths + utils.luminance_ac_values

    if channels == 3:
        # Add chrominance channel data for color images
        sof_marker += b'\2\21\1\3\21\1'
        sos_marker += b'\2\21\3\21'
        quant_table_marker += b'\1' + chrominance_quant
        huffman_table_marker += b'\1' + utils.chrominance_dc_code_lengths + utils.chrominance_dc_values + b'\21' + utils.chrominance_ac_code_lengths + utils.chrominance_ac_values
    elif channels == 4:
        # Add additional channel data for 4-channel images
        sof_marker += b'\2\21\0\3\21\0\4\21\0'
        sos_marker += b'\2\0\3\0\4\0'

    sos_marker += b'\0\77\0'  # Magic values: Start, end, and approximation of spectral selection

    return b''.join([
        b'\xff\xd8',  # SOI (Start of Image)
        create_marker_segment(b'\xee', app_marker) if channels == 4 else b'',  # Optional Adobe marker
        create_marker_segment(b'\xdb', quant_table_marker),  # DQT (Define Quantization Table)
        create_marker_segment(b'\xc0', sof_marker),  # SOF0 (Start of Frame, Baseline DCT)
        create_marker_segment(b'\xc4', huffman_table_marker),  # DHT (Define Huffman Table)
        create_marker_segment(b'\xda', sos_marker),  # SOS (Start of Scan)
        entropy_encoder.get_data(),  # Compressed image data
        b'\xff\xd9'  # EOI (End of Image)
    ])
