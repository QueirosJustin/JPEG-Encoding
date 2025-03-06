from array import array

def perform_forward_dct(data_block):
    # Ref.: Independent JPEG Group's "jfdctint.c", v8d
    # Copyright (C) 1994-1996, Thomas G. Lane
    # Modification developed 2003-2009 by Guido Vollbeding
    # Applies the forward Discrete Cosine Transform (DCT) to an 8x8 data block for image compression.
    for row in range(0, 64, 8):
        sum0 = data_block[row] + data_block[row + 7]
        sum1 = data_block[row + 1] + data_block[row + 6]
        sum2 = data_block[row + 2] + data_block[row + 5]
        sum3 = data_block[row + 3] + data_block[row + 4]
        total_sum0 = sum0 + sum3
        diff_sum0 = sum0 - sum3
        total_sum1 = sum1 + sum2
        diff_sum1 = sum1 - sum2
        sum0 = data_block[row] - data_block[row + 7]
        sum1 = data_block[row + 1] - data_block[row + 6]
        sum2 = data_block[row + 2] - data_block[row + 5]
        sum3 = data_block[row + 3] - data_block[row + 4]
        data_block[row] = (total_sum0 + total_sum1 - 8 * 128) << 2
        data_block[row + 4] = (total_sum0 - total_sum1) << 2
        scale1 = (diff_sum0 + diff_sum1) * 4433
        scale1 += 1024
        data_block[row + 2] = (scale1 + diff_sum0 * 6270) >> 11
        data_block[row + 6] = (scale1 - diff_sum1 * 15137) >> 11
        total_sum0 = sum0 + sum3
        total_sum1 = sum1 + sum2
        diff_sum0 = sum0 + sum2
        diff_sum1 = sum1 + sum3
        scale1 = (diff_sum0 + diff_sum1) * 9633
        scale1 += 1024
        sum0 = sum0 * 12299
        sum1 = sum1 * 25172
        sum2 = sum2 * 16819
        sum3 = sum3 * 2446
        total_sum0 = total_sum0 * -7373
        total_sum1 = total_sum1 * -20995
        diff_sum0 = diff_sum0 * -3196
        diff_sum1 = diff_sum1 * -16069
        diff_sum0 += scale1
        diff_sum1 += scale1
        data_block[row + 1] = (sum0 + total_sum0 + diff_sum0) >> 11
        data_block[row + 3] = (sum1 + total_sum1 + diff_sum1) >> 11
        data_block[row + 5] = (sum2 + total_sum1 + diff_sum0) >> 11
        data_block[row + 7] = (sum3 + total_sum0 + diff_sum1) >> 11
    for col in range(8):
        sum0 = data_block[col] + data_block[col + 56]
        sum1 = data_block[col + 8] + data_block[col + 48]
        sum2 = data_block[col + 16] + data_block[col + 40]
        sum3 = data_block[col + 24] + data_block[col + 32]
        total_sum0 = sum0 + sum3 + 2
        diff_sum0 = sum0 - sum3
        total_sum1 = sum1 + sum2
        diff_sum1 = sum1 - sum2
        sum0 = data_block[col] - data_block[col + 56]
        sum1 = data_block[col + 8] - data_block[col + 48]
        sum2 = data_block[col + 16] - data_block[col + 40]
        sum3 = data_block[col + 24] - data_block[col + 32]
        data_block[col] = (total_sum0 + total_sum1) >> 2
        data_block[col + 32] = (total_sum0 - total_sum1) >> 2
        scale1 = (diff_sum0 + diff_sum1) * 4433
        scale1 += 16384
        data_block[col + 16] = (scale1 + diff_sum0 * 6270) >> 15
        data_block[col + 48] = (scale1 - diff_sum1 * 15137) >> 15
        total_sum0 = sum0 + sum3
        total_sum1 = sum1 + sum2
        diff_sum0 = sum0 + sum2
        diff_sum1 = sum1 + sum3
        scale1 = (diff_sum0 + diff_sum1) * 9633
        scale1 += 16384
        sum0 = sum0 * 12299
        sum1 = sum1 * 25172
        sum2 = sum2 * 16819
        sum3 = sum3 * 2446
        total_sum0 = total_sum0 * -7373
        total_sum1 = total_sum1 * -20995
        diff_sum0 = diff_sum0 * -3196
        diff_sum1 = diff_sum1 * -16069
        diff_sum0 += scale1
        diff_sum1 += scale1
        data_block[col + 8] = (sum0 + total_sum0 + diff_sum0) >> 15
        data_block[col + 24] = (sum1 + total_sum1 + diff_sum1) >> 15
        data_block[col + 40] = (sum2 + total_sum1 + diff_sum0) >> 15
        data_block[col + 56] = (sum3 + total_sum0 + diff_sum1) >> 15

def perform_inverse_dct(data_block, quantization_table):
    # Ref.: Independent JPEG Group's "jfdctint.c", v8d
    # Copyright (C) 1994-1996, Thomas G. Lane
    # Modification developed 2003-2009 by Guido Vollbeding
    # Applies the inverse Discrete Cosine Transform (IDCT) to an 8x8 data block using the given quantization table.
    for row in range(8):
        intermediate1 = data_block[16 + row] * quantization_table[16 + row]
        intermediate2 = data_block[48 + row] * quantization_table[48 + row]
        scale1 = (intermediate1 + intermediate2) * 4433
        result1 = scale1 + intermediate1 * 6270
        result2 = scale1 - intermediate2 * 15137
        intermediate1 = data_block[row] * quantization_table[row]
        intermediate2 = data_block[32 + row] * quantization_table[32 + row]
        intermediate1 <<= 13
        intermediate2 <<= 13
        intermediate1 += 1024
        temp0 = intermediate1 + intermediate2
        temp1 = intermediate1 - intermediate2
        total_sum0 = temp0 + result1
        diff_sum0 = temp0 - result1
        total_sum1 = temp1 + result2
        diff_sum1 = temp1 - result2
        temp0 = data_block[56 + row] * quantization_table[56 + row]
        temp1 = data_block[40 + row] * quantization_table[40 + row]
        result1 = data_block[24 + row] * quantization_table[24 + row]
        result2 = data_block[8 + row] * quantization_table[8 + row]
        intermediate1 = temp0 + result1
        intermediate2 = temp1 + result2
        scale1 = (intermediate1 + intermediate2) * 9633
        intermediate1 = intermediate1 * -16069
        intermediate2 = intermediate2 * -3196
        intermediate1 += scale1
        intermediate2 += scale1
        scale1 = (temp0 + result2) * -7373
        temp0 = temp0 * 2446
        result2 = result2 * 12299
        temp0 += scale1 + intermediate1
        result2 += scale1 + intermediate2
        scale1 = (temp1 + result1) * -20995
        temp1 = temp1 * 16819
        result1 = result1 * 25172
        temp1 += scale1 + intermediate2
        result1 += scale1 + intermediate1
        data_block[row] = (total_sum0 + result2) >> 11
        data_block[56 + row] = (total_sum0 - result2) >> 11
        data_block[8 + row] = (total_sum1 + result1) >> 11
        data_block[48 + row] = (total_sum1 - result1) >> 11
        data_block[16 + row] = (diff_sum0 + temp1) >> 11
        data_block[40 + row] = (diff_sum0 - temp1) >> 11
        data_block[24 + row] = (diff_sum1 + temp0) >> 11
        data_block[32 + row] = (diff_sum1 - temp0) >> 11
    for col in range(0, 64, 8):
        intermediate1 = data_block[2 + col]
        intermediate2 = data_block[6 + col]
        scale1 = (intermediate1 + intermediate2) * 4433
        result1 = scale1 + intermediate1 * 6270
        result2 = scale1 - intermediate2 * 15137
        intermediate1 = data_block[col] + 16
        intermediate2 = data_block[4 + col]
        temp0 = (intermediate1 + intermediate2) << 13
        temp1 = (intermediate1 - intermediate2) << 13
        total_sum0 = temp0 + result1
        diff_sum0 = temp0 - result1
        total_sum1 = temp1 + result2
        diff_sum1 = temp1 - result2
        temp0 = data_block[7 + col]
        temp1 = data_block[5 + col]
        result1 = data_block[3 + col]
        result2 = data_block[1 + col]
        intermediate1 = temp0 + result1
        intermediate2 = temp1 + result2
        scale1 = (intermediate1 + intermediate2) * 9633
        intermediate1 = intermediate1 * -16069
        intermediate2 = intermediate2 * -3196
        intermediate1 += scale1
        intermediate2 += scale1
        scale1 = (temp0 + result2) * -7373
        temp0 = temp0 * 2446
        result2 = result2 * 12299
        temp0 += scale1 + intermediate1
        result2 += scale1 + intermediate2
        scale1 = (temp1 + result1) * -20995
        temp1 = temp1 * 16819
        result1 = result1 * 25172
        temp1 += scale1 + intermediate2
        result1 += scale1 + intermediate1
        data_block[col] = (total_sum0 + result2) >> 18
        data_block[7 + col] = (total_sum0 - result2) >> 18
        data_block[1 + col] = (total_sum1 + result1) >> 18
        data_block[6 + col] = (total_sum1 - result1) >> 18
        data_block[2 + col] = (diff_sum0 + temp1) >> 18
        data_block[5 + col] = (diff_sum0 - temp1) >> 18
        data_block[3 + col] = (diff_sum1 + temp0) >> 18
        data_block[4 + col] = (diff_sum1 - temp0) >> 18

"""
The zig-zag order rearranges the AC coefficients from an 8x8 block into a 1D sequence.
This order is used to group low-frequency coefficients together for better compression efficiency.
"""
zig_zag = bytes([ # Zig-zag order for AC coefficients in an 8x8 block, used in JPEG compression
         1,  8, 16,  9,  2,  3, 10, 17, 24, 32, 25, 18, 11,  4,  5,
    12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13,  6,  7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63])

"""
The luminance quantization table is used to quantize the DCT coefficients for the luminance (Y) channel.
Smaller values retain more precision, while larger values lead to higher compression.
"""
luminance_quant_table = bytes([ # Default luminance quantization table in zig-zag order
    16, 11, 12, 14, 12, 10, 16, 14, 13, 14, 18, 17, 16, 19, 24, 40,
    26, 24, 22, 22, 24, 49, 35, 37, 29, 40, 58, 51, 61, 60, 57, 51,
    56, 55, 64, 72, 92, 78, 64, 68, 87, 69, 55, 56, 80,109, 81, 87,
    95, 98,103,104,103, 62, 77,113,121,112,100,120, 92,101,103, 99])

"""
The chrominance quantization table is used to quantize the DCT coefficients for the chrominance (U, V) channels.
Higher values lead to greater compression, reflecting human perception's lower sensitivity to color detail.
"""
chrominance_quant_table = bytes([ # Default chrominance quantization table in zig-zag order
    17, 18, 18, 24, 21, 24, 47, 26, 26, 47, 99, 66, 56, 66, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99])

"""
The lengths of the codes for the DC coefficients of the luminance channel.
Each value represents the number of codes of a given length.
"""
luminance_dc_code_lengths = bytes([ # Code lengths for luminance DC Huffman table
    0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0])

"""
The actual values associated with the Huffman codes for the DC coefficients of the luminance channel.
"""
luminance_dc_values = bytes([ # Huffman values for luminance DC coefficients
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])

"""
The lengths of the codes for the AC coefficients of the luminance channel.
Each value represents the number of codes of a given length.
"""
luminance_ac_code_lengths = bytes([ # Code lengths for luminance AC Huffman table
    0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 125])

"""
The actual values associated with the Huffman codes for the AC coefficients of the luminance channel.
"""
luminance_ac_values = bytes([ # Huffman values for luminance AC coefficients
      1,  2,  3,  0,  4, 17,  5, 18, 33, 49, 65,  6, 19, 81, 97,  7, 34,113,
     20, 50,129,145,161,  8, 35, 66,177,193, 21, 82,209,240, 36, 51, 98,114,
    130,  9, 10, 22, 23, 24, 25, 26, 37, 38, 39, 40, 41, 42, 52, 53, 54, 55,
     56, 57, 58, 67, 68, 69, 70, 71, 72, 73, 74, 83, 84, 85, 86, 87, 88, 89,
     90, 99,100,101,102,103,104,105,106,115,116,117,118,119,120,121,122,131,
    132,133,134,135,136,137,138,146,147,148,149,150,151,152,153,154,162,163,
    164,165,166,167,168,169,170,178,179,180,181,182,183,184,185,186,194,195,
    196,197,198,199,200,201,202,210,211,212,213,214,215,216,217,218,225,226,
    227,228,229,230,231,232,233,234,241,242,243,244,245,246,247,248,249,250])

"""
The lengths of the codes for the DC coefficients of the chrominance channels (U, V).
Each value represents the number of codes of a given length.
"""
chrominance_dc_code_lengths = bytes([ # Code lengths for chrominance DC Huffman table
    0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0])

"""
The actual values associated with the Huffman codes for the DC coefficients of the chrominance channels (U, V).
"""
chrominance_dc_values = bytes([ # Huffman values for chrominance DC coefficients
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])

"""
The lengths of the codes for the AC coefficients of the chrominance channels (U, V).
Each value represents the number of codes of a given length.
"""
chrominance_ac_code_lengths = bytes([ # Code lengths for chrominance AC Huffman table
    0, 2, 1, 2, 4, 4, 3, 4, 7, 5, 4, 4, 0, 1, 2, 119])

"""
The actual values associated with the Huffman codes for the AC coefficients of the chrominance channels (U, V).
"""
chrominance_ac_values = bytes([ # Huffman values for chrominance AC coefficients
      0,  1,  2,  3, 17,  4,  5, 33, 49,  6, 18, 65, 81,  7, 97,113, 19, 34,
     50,129,  8, 20, 66,145,161,177,193,  9, 35, 51, 82,240, 21, 98,114,209,
     10, 22, 36, 52,225, 37,241, 23, 24, 25, 26, 38, 39, 40, 41, 42, 53, 54,
     55, 56, 57, 58, 67, 68, 69, 70, 71, 72, 73, 74, 83, 84, 85, 86, 87, 88,
     89, 90, 99,100,101,102,103,104,105,106,115,116,117,118,119,120,121,122,
    130,131,132,133,134,135,136,137,138,146,147,148,149,150,151,152,153,154,
    162,163,164,165,166,167,168,169,170,178,179,180,181,182,183,184,185,186,
    194,195,196,197,198,199,200,201,202,210,211,212,213,214,215,216,217,218,
    226,227,228,229,230,231,232,233,234,242,243,244,245,246,247,248,249,250])

def parse_exif_metadata(readable, length, rotation):
    """
    Parses EXIF metadata from the provided data buffer to extract image orientation information.
    
    Parameters:
    - readable: An instance of a readable object used for reading binary data.
    - length: Length of the EXIF metadata segment.
    - rotation: The default or initial rotation value.

    Returns:
    - The image rotation value extracted from EXIF metadata.
    """
    end_position = readable.position + length  # Calculate the end position for this segment
    if readable.peek(b'Exif\0\0'):  # Check for the EXIF header prefix
        readable.move_forward(6)  # Move past the EXIF header
        byte_order = readable.uint16()  # Read the byte order indicator (0x4D4D for big-endian, 0x4949 for little-endian)
        if byte_order == 0x4d4d:
            uint16, uint32 = readable.uint16, readable.uint32  # Set big-endian functions
        elif byte_order == 0x4949:
            uint16, uint32 = readable.uint16le, readable.uint32le  # Set little-endian functions
        else:
            raise ValueError('Invalid byte order.')  # Invalid byte order detected
        readable.move_forward(2)  # Move past a reserved field
        offset = uint32()  # Get the offset to the first IFD (Image File Directory)
        if offset < 8:
            raise ValueError('Invalid IFD0 offset.')
        readable.move_forward(offset - 8)  # Move to the start of the IFD0
        entries = uint16()  # Number of directory entries
        for _ in range(entries):
            tag = uint16()  # Read the tag
            readable.move_forward(6)  # Move past the tag type and count fields
            if tag == 0x112:  # Orientation tag
                orientation = uint16()  # Read the orientation value
                if orientation == 1:
                    rotation = 0  # Normal orientation
                elif orientation == 6:
                    rotation = 90  # Rotated 90 degrees clockwise
                elif orientation == 3:
                    rotation = 180  # Rotated 180 degrees
                elif orientation == 8:
                    rotation = 270  # Rotated 270 degrees clockwise
                else:
                    raise ValueError('Unsupported orientation value.')
                break
            readable.move_forward(4)  # Move to the next entry
    readable.jump(end_position)  # Jump to the end of the EXIF segment
    return rotation

def parse_start_of_frame(readable, components):
    """
    Parses the Start of Frame (SOF) segment to extract image dimensions and component information.
    
    Parameters:
    - readable: An instance of a readable object used for reading binary data.
    - components: A list to which parsed frame components will be appended.

    Returns:
    - width: Image width.
    - height: Image height.
    - color_type: Color format ('g', 'rgb', or 'cmyk').
    - num_components: Number of components (e.g., 1 for grayscale, 3 for RGB).
    """
    depth, height, width, num_components = readable.interpret_format('>BHHB')  # Read depth, height, width, and number of components
    if depth != 8:
        raise ValueError('Unsupported sample precision.')  # JPEG typically uses 8-bit precision
    color_type = 'g' if num_components == 1 else 'rgb' if num_components == 3 else 'cmyk'
    for _ in range(num_components):
        identifier, sampling_factors, destination = readable.interpret_format('>BBB')  # Parse component information
        horizontal, vertical = sampling_factors >> 4, sampling_factors & 15  # Extract horizontal and vertical sampling factors
        if num_components > 1 and (horizontal != 1 or vertical != 1):
            raise ValueError('Unsupported sampling factor.')  # Non-standard sampling factors are not supported
        components.append(FrameComponent(identifier, horizontal, vertical, destination))
    return width, height, color_type, num_components

def parse_quantization_table(readable, length, qtables):
    """
    Parses the quantization table segment from a JPEG file and populates the quantization tables.
    
    Parameters:
    - readable: An instance of a readable object used for reading binary data.
    - length: Length of the quantization table segment.
    - qtables: A dictionary to store parsed quantization tables.
    """
    end_position = readable.position + length  # Calculate the end position of the segment
    while readable.position < end_position:
        pqtq = readable.read_uint8()  # Read precision and table ID
        _, table_id = pqtq >> 4, pqtq & 15  # Extract precision and table ID (upper and lower 4 bits)
        elements = readable.extract_data(64)  # Read 64 elements for the quantization table
        table = bytearray(64)  # Initialize a table with 64 elements
        table[0] = elements[0]  # Assign the DC coefficient
        index = 1
        for zigzag_index in zig_zag:  # Rearrange elements according to zig-zag order
            table[zigzag_index] = elements[index]
            index += 1
        qtables[table_id] = table  # Store the parsed table
    if readable.position != end_position:
        raise ValueError('Invalid DQT length.')  # Ensure the segment length was correctly read

def parse_huffman_table(readable, length, htables):
    """
    Parses the Huffman table segment from a JPEG file and populates the Huffman tables.
    
    Parameters:
    - readable: An instance of a readable object used for reading binary data.
    - length: Length of the Huffman table segment.
    - htables: A dictionary to store parsed Huffman tables.
    """
    end_position = readable.position + length  # Calculate the end position of the segment
    while readable.position < end_position:
        tcth = readable.read_uint8()  # Read table class and table identifier
        lengths = readable.extract_data(16)  # Read code lengths (16 entries)
        values = readable.extract_data(sum(lengths))  # Read Huffman values based on code lengths
        htables[tcth] = HuffmanCache(lengths, values)  # Store the Huffman table in the cache

def parse_start_of_scan(readable, scans):
    """
    Parses the Start of Scan (SOS) segment, which defines how image components are encoded.
    
    Parameters:
    - readable: An instance of a readable object used for reading binary data.
    - scans: A dictionary to store the component scan information.
    """
    num_components = readable.read_uint8()  # Number of components in the scan
    for _ in range(num_components):
        component_selector, huffman_destinations = readable.interpret_format('>BB')  # Read component selector and Huffman table IDs
        dc_id, ac_id = huffman_destinations >> 4, huffman_destinations & 15  # Extract DC and AC table IDs
        scans[component_selector] = CodingDestination(dc_id, 16 | ac_id)  # Store the coding destination
    readable.move_forward(3)  # Move past three reserved bytes

def parse_restart_interval(readable):
    """
    Parses the restart interval marker, which defines intervals for resetting the entropy encoder in JPEG compression.
    
    Parameters:
    - readable: An instance of a readable object used for reading binary data.

    Returns:
    - The parsed restart interval.
    """
    interval = readable.uint16()  # Read a 16-bit unsigned integer
    return interval

def parse_app14_segment(readable):
    """
    Parses the APP14 segment (Adobe-specific) in JPEG files to extract transform information.
    
    Parameters:
    - readable: An instance of a readable object used for reading binary data.

    Returns:
    - The transform flag indicating the type of color transform.
    """
    readable.move_forward(11)  # Move past reserved fields and header data
    transform_flag = readable.read_uint8()  # Read the transform flag
    return transform_flag

class FrameComponent:
    """
    Represents a frame component in the JPEG structure.
    """
    __slots__ = 'identifier', 'horizontal', 'vertical', 'destination'
    
    def __init__(self, identifier, horizontal, vertical, destination):
        self.identifier = identifier  # Component identifier
        self.horizontal = horizontal  # Horizontal sampling factor
        self.vertical = vertical  # Vertical sampling factor
        self.destination = destination  # Quantization table destination

class CodingDestination:
    """
    Represents the coding destination for a component, including DC and AC Huffman table IDs.
    """
    __slots__ = 'dc', 'ac'
    
    def __init__(self, dc, ac):
        self.dc = dc  # DC Huffman table ID
        self.ac = ac  # AC Huffman table ID

class HuffmanCache:
    """
    Caches Huffman table data, including lengths, values, and lookup tables for decoding.
    """
    def __init__(self, lengths, values):
        self.values = values  # Huffman values
        self.offsets = offsets = array('H', [0])  # Offset table for faster lookup
        self.sizes = sizes = bytearray([255] * 65536)  # Size table initialized with maximum value (255)
        code = index = 0
        size = 1
        for length in lengths:  # Populate the Huffman code table
            offsets.append(code - index)
            for _ in range(length):
                hi = code << (16 - size)  # Shift code bits to the upper 16 bits
                for lo in range(1 << (16 - size)):  # Populate the size table for each code
                    sizes[hi | lo] = size
                code += 1
            code *= 2  # Double the code for the next size
            index += length
            size += 1
