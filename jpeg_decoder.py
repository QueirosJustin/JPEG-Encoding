from Modules.misc import bounded_value
from Modules.readable import readable
import Modules.utils as utils

class EntropyDecoder(object):
    """
    This class handles entropy decoding for JPEG images,
    using Huffman tables and bitstream manipulation to extract coefficients.
    """
    
    def __init__(self, readable):
        self.readable = readable
        self.value = 0  # Holds the buffered bits for decoding
        self.length = 0  # Number of bits available in the buffer
        self.rst = 0  # Restart interval count
    
    def reset_entropy_decoder(self):
        """
        Resets the internal state of the entropy decoder, primarily used
        when a restart interval marker is encountered in the JPEG stream.
        """
        self.value, self.length = 0, 0
        self.rst = (self.rst + 1) % 8
    
    def fill_buffer(self, target_length):
        """
        Fills the internal bit buffer until it reaches or exceeds target_length.
        Handles marker segments to ensure JPEG byte alignment.
        """
        while self.length < target_length:
            new_byte = self.readable.read_uint8()
            self.value = ((self.value & 0xFFFF) << 8) | new_byte
            self.length += 8
            if new_byte == 0xFF:  # Check for marker sequence
                next_byte = self.readable.read_uint8()
                if next_byte != 0:
                    self.readable.position -= 2  # Adjust position for markers
    
    def decode_huffman(self, huffman_cache):
        """
        Decodes a symbol using Huffman tables stored in huffman_cache.
        Raises an exception for invalid sequences.
        """
        if self.length < 16:
            self.fill_buffer(16)
        huffman_key = (self.value >> (self.length - 16)) & 0xFFFF
        huffman_size = huffman_cache.sizes[huffman_key]
        if huffman_size == 255:
            raise Exception('Corrupted Huffman sequence.')
        decoded_code = (self.value >> (self.length - huffman_size)) & ((1 << huffman_size) - 1)
        self.length -= huffman_size
        return huffman_cache.values[decoded_code - huffman_cache.offsets[huffman_size]]
    
    def receive_extend(self, bit_length):
        """
        Retrieves a signed value of bit_length bits from the bitstream,
        applying sign-extension to handle negative values appropriately.
        """
        if self.length < bit_length:
            self.fill_buffer(bit_length)
        extracted_value = (self.value >> (self.length - bit_length)) & ((1 << bit_length) - 1)
        self.length -= bit_length
        if extracted_value < 1 << (bit_length - 1):
            return extracted_value - (1 << bit_length) + 1
        return extracted_value

    def decode_block(self, prev_value, coeff_block, quant_table, dc_table, ac_table):
        """
        Decodes an 8x8 block of coefficients using DC and AC Huffman tables.
        Applies quantization and inverse DCT to reconstruct pixel values.
        """
        index = 0
        while index < 64:
            coeff_block[index] = coeff_block[index + 1] = coeff_block[index + 2] = coeff_block[index + 3] = 0
            index += 4
        huffman_value = self.decode_huffman(dc_table)
        delta = 0 if huffman_value == 0 else self.receive_extend(huffman_value)
        prev_value += delta
        coeff_block[0] = prev_value  # Store DC coefficient
        index = 0
        while index < 63:
            run_size = self.decode_huffman(ac_table)
            size = run_size & 15
            run_length = run_size >> 4
            if size == 0:  # Handle zero run-lengths
                if run_length != 15:
                    break
                index += 16
            else:
                index += run_length
                coeff_block[utils.zig_zag[index]] = self.receive_extend(size)
                index += 1
        utils.perform_inverse_dct(coeff_block, quant_table)  # Perform inverse DCT
        return prev_value

class JpegDecoder(object):
    """
    This class handles JPEG file structure parsing and image decompression.
    It decodes headers, quantization and Huffman tables, and performs
    entropy decoding to reconstruct image data.
    """
    
    @staticmethod
    def is_valid_jpeg(data):
        """
        Validates the data as a JPEG image by checking for the SOI marker (0xFFD8FF).
        """
        return data.startswith(b'\xff\xd8\xff')
    
    def __init__(self, data):
        """
        Initializes the JPEG decoder, parses the file structure, and prepares
        for decompression by reading essential segments and markers.
        """
        self.readable = r = readable(data)
        self.width, self.height, self.kind, self.n = 0, 0, '', 0
        self.components = []  # Stores color components
        self.scans = {}  # Maps scan data
        self.qtables = {}  # Quantization tables
        self.htables = {}  # Huffman tables
        self.interval = 0  # Restart interval
        self.rotation = 0  # Rotation metadata
        self.progressive = False
        self.transform, app14 = False, False
        self.ecs = 0  # Start of entropy-coded segment
        r.move_forward(2)  # Skip SOI marker
        while True:
            marker = r.read_uint8()
            while marker == 0xff:
                marker = r.read_uint8()
            length = r.read_uint16() - 2
            if 0xc0 <= marker <= 0xc2:  # SOF0, SOF1, SOF2
                self.width, self.height, self.kind, self.n = utils.parse_start_of_frame(r, self.components)
                self.progressive = marker == 0xc2
                if not app14:
                    self.transform = self.kind == 'rgb'
            elif marker == 0xdb:  # Define Quantization Table
                utils.parse_quantization_table(r, length, self.qtables)
            elif marker == 0xc4:  # Define Huffman Table
                utils.parse_huffman_table(r, length, self.htables)
            elif marker == 0xda:  # Start of Scan
                utils.parse_start_of_scan(r, self.scans)
                self.ecs = r.position
                break
            elif marker == 0xdd:  # Define Restart Interval
                self.interval = utils.parse_restart_interval(r)
            elif marker == 0xe1:  # APP1 marker (e.g., EXIF metadata)
                self.rotation = utils.parse_exif_metadata(r, length, self.rotation)
            elif marker == 0xee:  # APP14 marker (Adobe transform segment)
                self.transform, app14 = utils.parse_app14_segment(r), True
            elif 0xe0 <= marker <= 0xef or marker == 0xfe:  # APP, COM segments
                r.move_forward(length)
            else:
                raise ValueError('Unsupported marker.')

    def setup_components(self, num_channels):
        """
        Sets up components for the decompression process.
        """
        y_blocks = [[0] * 64 for _ in range(4)]
        u_block, v_block, k_block = [0] * 64, [0] * 64, [0] * 64
        block_groups = [y_blocks, [u_block], [v_block], [k_block]]
        horiz_samples = [component.horizontal for component in self.components]
        vert_samples = [component.vertical for component in self.components]
        quant_tables = [self.qtables[component.destination] for component in self.components]
        dc_tables = [self.htables[self.scans[component.identifier].dc] for component in self.components]
        ac_tables = [self.htables[self.scans[component.identifier].ac] for component in self.components]
        h_sampling, v_sampling = horiz_samples[0], vert_samples[0]
        return y_blocks, block_groups, horiz_samples, vert_samples, quant_tables, dc_tables, ac_tables, h_sampling, v_sampling

    def calculate_restart_interval(self, width, height, h_sampling, v_sampling):
        """
        Calculates the restart interval if not specified.
        """
        return ((width + 8 * h_sampling - 1) // (8 * h_sampling)) * ((height + 8 * v_sampling - 1) // (8 * v_sampling))

    def handle_restart_interval(self, block_count, restart_interval, entropy_decoder, prediction_values):
        """
        Handles entropy decoder reset and prediction value re-initialization
        when the block count exceeds the restart interval.
        """
        block_count += 1
        if block_count > restart_interval:
            entropy_decoder.reset_entropy_decoder()
            prediction_values[:] = [0, 0, 0, 0]
            block_count = 1
        return block_count

    def decode_components(self, entropy_decoder, num_channels, prediction_values, block_groups, horiz_samples, vert_samples, quant_tables, dc_tables, ac_tables):
        """
        Decodes the components of the image data using entropy decoding.
        """
        for component_index in range(num_channels):
            for sample_index in range(horiz_samples[component_index] * vert_samples[component_index]):
                prediction_values[component_index] = entropy_decoder.decode_block(
                    prediction_values[component_index], block_groups[component_index][sample_index],
                    quant_tables[component_index], dc_tables[component_index], ac_tables[component_index]
                )

    def assign_pixel_values(self, num_channels, image_data, data_index, y_block, block_groups, pixel_index, block_x, block_y, color_transform):
        """
        Assigns pixel values based on the number of channels and performs optional color transformation.

        Explanation of magic values:
        - 91881: Coefficient for V (chrominance red) when converting to the red channel (derived from 1.402 * 65536 for integer math).
        - 22554: Coefficient for U (chrominance blue) when converting to the green channel (derived from -0.34414 * 65536).
        - 46802: Coefficient for V (chrominance red) when converting to the green channel (derived from -0.71414 * 65536).
        - 116130: Coefficient for U (chrominance blue) when converting to the blue channel (derived from 1.772 * 65536).
        """
        if num_channels == 1:
            image_data[data_index] = bounded_value(y_block[pixel_index] + 128)
        elif num_channels == 3:
            y_val, u_val, v_val = y_block[block_x + block_y * 8], block_groups[1][0][pixel_index], block_groups[2][0][pixel_index]
            y_val = (y_val << 16) + 8421376  # 8421376 accounts for fixed-point scaling (128 << 16)
            image_data[data_index] = bounded_value((y_val + 91881 * v_val) >> 16)  # Conversion to red channel
            image_data[data_index + 1] = bounded_value((y_val - 22554 * u_val - 46802 * v_val) >> 16)  # Conversion to green channel
            image_data[data_index + 2] = bounded_value((y_val + 116130 * u_val) >> 16)  # Conversion to blue channel
        else:  # num_channels == 4
            y_val, u_val, v_val, k_val = y_block[block_x + block_y * 8], block_groups[1][0][pixel_index], block_groups[2][0][pixel_index], block_groups[3][0][pixel_index]
            if color_transform:
                y_val = (y_val << 16) + 8421376  # 8421376 accounts for fixed-point scaling (128 << 16)
                image_data[data_index] = 255 - bounded_value((y_val + 91881 * v_val) >> 16)  # Inverted red channel with color transform
                image_data[data_index + 1] = 255 - bounded_value((y_val - 22554 * u_val - 46802 * v_val) >> 16)  # Inverted green channel
                image_data[data_index + 2] = 255 - bounded_value((y_val + 116130 * u_val) >> 16)  # Inverted blue channel
                image_data[data_index + 3] = bounded_value(k_val + 128)
            else:
                image_data[data_index] = bounded_value(y_val + 128)
                image_data[data_index + 1] = bounded_value(u_val + 128)
                image_data[data_index + 2] = bounded_value(v_val + 128)
                image_data[data_index + 3] = bounded_value(k_val + 128)

    def reconstruct_pixels(self, y_pos, x_pos, h_sampling, v_sampling, width, height, num_channels, image_data, y_blocks, block_groups, color_transform):
        """
        Reconstructs pixel data based on the decoded blocks and performs color transformation.
        """
        horiz_bits, vert_bits = h_sampling.bit_length() - 1, v_sampling.bit_length() - 1
        for v_offset in range(v_sampling):
            for h_offset in range(h_sampling):
                y_block = y_blocks[h_offset + v_offset * h_sampling]
                for block_y in range(min(8, height - y_pos - v_offset * 8)):
                    for block_x in range(min(8, width - x_pos - h_offset * 8)):
                        pixel_index = ((h_offset * 8 + block_x) >> horiz_bits) + ((v_offset * 8 + block_y) >> vert_bits) * 8
                        data_index = (x_pos + h_offset * 8 + block_x + (y_pos + v_offset * 8 + block_y) * width) * num_channels
                        self.assign_pixel_values(num_channels, image_data, data_index, y_block, block_groups, pixel_index, block_x, block_y, color_transform)

    def decompress_image(self):
        """
        Decompresses the image data using entropy decoding, inverse quantization,
        and color transformation. Outputs the reconstructed image data as bytes.
        """
        self.readable.move_to(self.ecs)  # Move to entropy-coded data start
        width, height, num_channels = self.width, self.height, self.n
        restart_interval, color_transform = self.interval, self.transform
        entropy_decoder = EntropyDecoder(self.readable)
        image_data = bytearray(width * height * num_channels)
        prediction_values = [0, 0, 0, 0]  # Prediction values for DC coefficients
        
        y_blocks, block_groups, horiz_samples, vert_samples, quant_tables, dc_tables, ac_tables, h_sampling, v_sampling = self.setup_components(num_channels)
        block_count = 0

        if restart_interval == 0:
            restart_interval = self.calculate_restart_interval(width, height, h_sampling, v_sampling)
        
        for y_pos in range(0, height, 8 * v_sampling):
            for x_pos in range(0, width, 8 * h_sampling):
                block_count = self.handle_restart_interval(block_count, restart_interval, entropy_decoder, prediction_values)
                self.decode_components(entropy_decoder, num_channels, prediction_values, block_groups, horiz_samples, vert_samples, quant_tables, dc_tables, ac_tables)
                self.reconstruct_pixels(y_pos, x_pos, h_sampling, v_sampling, width, height, num_channels, image_data, y_blocks, block_groups, color_transform)
        
        return image_data
