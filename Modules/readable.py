from copy import copy
from struct import calcsize, unpack_from

class readable(object):
    """
    A class that provides methods for reading binary data from a byte buffer,
    with utilities for moving and managing the read position.
    """
    
    __slots__ = 'data', 'position'
    
    def __init__(self, data):
        """
        Initializes the readable object with input data and sets the position pointer to the start.
        
        Parameters:
        - data: The input binary data to read from.
        """
        self.data = data
        self.position = 0
    
    def duplicate(self):
        """
        Creates and returns a copy of the current readable instance, preserving the data and position.
        
        Returns:
        - A copy of the readable instance.
        """
        return copy(self)
    
    def move_to(self, pos):
        """
        Sets the position pointer to a specific position within the data.
        
        Parameters:
        - pos: The position to move to.
        """
        self.position = pos
    
    def move_forward(self, offset):
        """
        Moves the position pointer forward by a specified offset.
        
        Parameters:
        - offset: The number of bytes to move the position forward.
        """
        self.position += offset
    
    def check_prefix(self, prefix):
        """
        Checks if the data at the current position starts with the given prefix.
        
        Parameters:
        - prefix: The prefix to check for.
        
        Returns:
        - True if the data starts with the given prefix; otherwise, False.
        """
        return self.data.startswith(prefix, self.position)
    
    def extract_data(self, length):
        """
        Reads and returns a specified length of data starting from the current position,
        and moves the position pointer forward by the length.
        
        Parameters:
        - length: The number of bytes to read.
        
        Returns:
        - The extracted data.
        """
        start = self.position
        self.position += length
        return self.data[start:self.position]
    
    def interpret_format(self, fmt):
        """
        Unpacks data based on a format string and updates the position pointer accordingly.
        
        Parameters:
        - fmt: The format string used for unpacking (compatible with the struct module).
        
        Returns:
        - The unpacked data as a tuple.
        """
        start = self.position
        self.position += calcsize(fmt)
        return unpack_from(fmt, self.data, start)
    
    def read_uint8(self):
        """
        Reads and returns an unsigned 8-bit integer from the current position, and moves the position forward.
        
        Returns:
        - An unsigned 8-bit integer.
        """
        start = self.position
        self.position += 1
        return self.data[start]
    
    def read_uint16(self):
        """
        Reads and returns an unsigned 16-bit integer in big-endian order, and moves the position forward.
        
        Returns:
        - An unsigned 16-bit integer.
        """
        buf, start = self.data, self.position
        self.position += 2
        # Combines two bytes in big-endian order
        return buf[start] << 8 | buf[start + 1]
    
    def read_uint32(self):
        """
        Reads and returns an unsigned 32-bit integer in big-endian order, and moves the position forward.
        
        Returns:
        - An unsigned 32-bit integer.
        """
        buf, start = self.data, self.position
        self.position += 4
        # Combines four bytes in big-endian order
        return buf[start] << 24 | buf[start + 1] << 16 | buf[start + 2] << 8 | buf[start + 3]
    
    def read_int8(self):
        """
        Reads and returns a signed 8-bit integer using two's complement representation.
        
        Returns:
        - A signed 8-bit integer.
        """
        val = self.read_uint8()
        # Converts to signed by checking the sign bit (7th bit)
        return val - ((val & (1 << 7)) << 1)
    
    def read_int16(self):
        """
        Reads and returns a signed 16-bit integer in big-endian order using two's complement representation.
        
        Returns:
        - A signed 16-bit integer.
        """
        val = self.read_uint16()
        # Converts to signed by checking the sign bit (15th bit)
        return val - ((val & (1 << 15)) << 1)
    
    def read_int32(self):
        """
        Reads and returns a signed 32-bit integer in big-endian order using two's complement representation.
        
        Returns:
        - A signed 32-bit integer.
        """
        val = self.read_uint32()
        # Converts to signed by checking the sign bit (31st bit)
        return val - ((val & (1 << 31)) << 1)
    
    def read_uint16le(self):
        """
        Reads and returns an unsigned 16-bit integer in little-endian order, and moves the position forward.
        
        Returns:
        - An unsigned 16-bit integer.
        """
        buf, start = self.data, self.position
        self.position += 2
        # Combines two bytes in little-endian order
        return buf[start] | buf[start + 1] << 8
    
    def read_uint32le(self):
        """
        Reads and returns an unsigned 32-bit integer in little-endian order, and moves the position forward.
        
        Returns:
        - An unsigned 32-bit integer.
        """
        buf, start = self.data, self.position
        self.position += 4
        # Combines four bytes in little-endian order
        return buf[start] | buf[start + 1] << 8 | buf[start + 2] << 16 | buf[start + 3] << 24
    
    def read_int16le(self):
        """
        Reads and returns a signed 16-bit integer in little-endian order using two's complement representation.
        
        Returns:
        - A signed 16-bit integer.
        """
        val = self.read_uint16le()
        # Converts to signed by checking the sign bit (15th bit)
        return val - ((val & (1 << 15)) << 1)
    
    def read_int32le(self):
        """
        Reads and returns a signed 32-bit integer in little-endian order using two's complement representation.
        
        Returns:
        - A signed 32-bit integer.
        """
        val = self.read_uint32le()
        # Converts to signed by checking the sign bit (31st bit)
        return val - ((val & (1 << 31)) << 1)
