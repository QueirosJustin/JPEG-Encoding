from math import copysign

def is_close(x, y, absolute=1e-10, relative=1e-10):
    """
    Checks if two floating-point numbers are close to each other within specified absolute and relative tolerances.
    
    Parameters:
    - x, y: The numbers to compare.
    - absolute (1e-10): The minimum absolute tolerance for considering two values close.
    - relative (1e-10): The minimum relative tolerance for considering two values close.
    
    Returns:
    - True if the difference between x and y is within the tolerance limits; otherwise, False.
    """
    difference = abs(x - y)
    tolerance_limit = absolute + relative * max(abs(x), abs(y))
    return difference <= tolerance_limit

def rounded_integer(x):
    """
    Rounds a floating-point number to the nearest integer using traditional rounding rules.
    
    Parameters:
    - x: The number to round.
    
    Returns:
    - The rounded integer value of x.
    """
    offset = copysign(0.5, x)  # Adjusts the rounding direction based on the sign of x
    return int(x + offset)

def bounded_value(x):
    """
    Clamps a value to the range [0, 255].
    
    Parameters:
    - x: The value to clamp.
    
    Returns:
    - 0 if x is less than 0.
    - 255 if x is greater than 255.
    - x otherwise.
    """
    if x < 0:
        return 0
    elif x > 255:
        return 255
    return x

def split_into_chunks(sequence, size):
    """
    Splits a sequence into chunks of the specified size.
    
    Parameters:
    - sequence: The input sequence to split.
    - size: The size of each chunk.
    
    Returns:
    - A generator yielding chunks of the sequence of the specified size.
    """
    return (sequence[start:start + size] for start in range(0, len(sequence), size))

def format_number(number):
    """
    Formats a floating-point number as a byte string, removing trailing zeros and the decimal point if not needed.
    
    Parameters:
    - number: The number to format.
    
    Returns:
    - A byte string representation of the formatted number.
    """
    formatted = b'%.4f' % number  # Formats the number to 4 decimal places
    return formatted.rstrip(b'0').rstrip(b'.')  # Removes trailing zeros and decimal point if not needed

def write_to_file(path, data):
    """
    Writes binary data to a file if a valid path is provided.
    
    Parameters:
    - path: The file path to write to. If None, the function does nothing.
    - data: The data to write.
    
    Returns:
    - The input data (useful for chaining operations).
    """
    if path:
        with open(path, 'wb') as file:
            file.write(data)
    return data

def convert_units(units):
    """
    Converts units to points (pt) based on predefined conversion factors.
    
    Parameters:
    - units: A string representing the unit type to convert ('pt', 'mm', 'cm', 'in').
    
    Returns:
    - The conversion factor from the given units to points.
    
    Raises:
    - ValueError if the input units are not recognized.
    
    Note:
    - Conversion factors:
      - 'pt': 1 point = 1.0 (no conversion needed).
      - 'mm': 1 millimeter = 72.0 / 25.4 points.
      - 'cm': 1 centimeter = 72.0 / 2.54 points.
      - 'in': 1 inch = 72.0 points.
    """
    unit_conversion = {
        'pt': 1.0,
        'mm': 72.0 / 25.4,  # Converts millimeters to points
        'cm': 72.0 / 2.54,  # Converts centimeters to points
        'in': 72.0  # Converts inches to points
    }
    if units not in unit_conversion:
        raise ValueError('Invalid units.')
    return unit_conversion[units]

class RangeMaximumQuery(object):
    """
    Implements a Range Maximum Query (RMQ) using a sparse table.
    Reference: Bender, M. A., Farach-Colton, M. (2000). The LCA Problem Revisited.
    """
    
    def __init__(self, sequence):
        """
        Initializes the sparse table for the input sequence.
        
        Parameters:
        - sequence: The input list for which RMQ will be performed.
        """
        self.table = [sequence]
        self._create_table(sequence)
    
    def _create_table(self, sequence):
        """
        Builds the sparse table for RMQ.
        
        Parameters:
        - sequence: The input list to build the sparse table from.
        """
        seq_length = len(sequence)
        for j in range(seq_length.bit_length() - 1):
            block_size = 1 << j  # Size of the current range (2^j)
            seq_length -= block_size
            partial_row = sequence[:seq_length]
            for idx, value in enumerate(partial_row, start=block_size):
                if value < sequence[idx]:
                    partial_row[idx - block_size] = sequence[idx]
            self.table.append(partial_row)
            sequence = partial_row

    def maximum_in_range(self, start, end):
        """
        Finds the maximum value in the specified range [start, end).
        
        Parameters:
        - start: The starting index of the range (inclusive).
        - end: The ending index of the range (exclusive).
        
        Returns:
        - The maximum value in the specified range.
        """
        level = (end - start).bit_length() - 1  # Determines the level in the sparse table to use
        row = self.table[level]
        return max(row[start], row[end - (1 << level)])  # Compares values in overlapping ranges
