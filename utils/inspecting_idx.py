import struct
import gzip
import os

def read_idx(filename):
    # If it's compressed, open with gzip
    opener = gzip.open if filename.endswith(".gz") else open
    
    with opener(filename, "rb") as f:
        # Read magic number (first 4 bytes, big-endian integer)
        magic, = struct.unpack(">I", f.read(4))
        print(f"Magic number: {magic:#010x}")
        
        # Decode datatype and number of dimensions
        data_type = (magic >> 8) & 0xFF
        dims = magic & 0xFF
        print(f"Data type code: {data_type}, Dimensions: {dims}")
        
        # Read sizes of each dimension
        shape = []
        for _ in range(dims):
            size, = struct.unpack(">I", f.read(4))
            shape.append(size)
        print(f"Shape: {shape}")
        
        # Peek at first few data values (unsigned bytes)
        data = f.read(100)  # first 100 bytes
        print("First 20 raw values:", list(data[:20]))


read_idx("data/FashionMNIST/raw/t10k-images-idx3-ubyte")

def inspectAllFiles(folderPath):
    for root, dirs, files in os.walk(folderPath):
        for filename in files:
            print("\n\n")
            print(filename)
            read_idx(folderPath + filename)

inspectAllFiles("data/FashionMNIST/raw/")