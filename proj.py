import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import networkx as nx

def padKeyForAES(key):
    # Target length for AES-128
    target_length = 16
    
    # Convert the key to bytes if it's a string
    if isinstance(key, str):
        key_bytes = key.encode()
    else:
        key_bytes = key
    
    # Check the length of the key
    key_length = len(key_bytes)
    
    if key_length < target_length:
        # If the key is too short, pad it with zeros
        padded_key = key_bytes + b'\x00' * (target_length - key_length)
    elif key_length > target_length:
        # If the key is too long, truncate it
        padded_key = key_bytes[:target_length]
    else:
        # If the key is already the correct length, do nothing
        padded_key = key_bytes
    
    return padded_key

def text2Unicode(text):
    # Ensure text length is 16 for simplicity
    text = text.ljust(16)[:16]
    text_matrix = np.array([ord(c) for c in text], dtype=np.uint8).reshape(4, 4)
    return text_matrix

def unicode2Text(matrix):
    text = ''.join(chr(int(c)) for c in matrix.flatten())
    return text

def subBytes(A):
    # Implementing S-box substitution
    s_box = np.array([
        [0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76],
        [0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0],
        [0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15],
        [0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75],
        [0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84],
        [0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf],
        [0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8],
        [0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2],
        [0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73],
        [0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb],
        [0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79],
        [0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08],
        [0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a],
        [0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e],
        [0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf],
        [0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16]
    ], dtype=np.uint8)
    B = s_box[A // 0x10, A % 0x10]
    return B

def invSubBytes(A):
    # Implementing inverse S-box substitution
    inv_s_box = np.array([
        [0x52, 0x09, 0x6a, 0xd5, 0x30, 0x36, 0xa5, 0x38, 0xbf, 0x40, 0xa3, 0x9e, 0x81, 0xf3, 0xd7, 0xfb],
        [0x7c, 0xe3, 0x39, 0x82, 0x9b, 0x2f, 0xff, 0x87, 0x34, 0x8e, 0x43, 0x44, 0xc4, 0xde, 0xe9, 0xcb],
        [0x54, 0x7b, 0x94, 0x32, 0xa6, 0xc2, 0x23, 0x3d, 0xee, 0x4c, 0x95, 0x0b, 0x42, 0xfa, 0xc3, 0x4e],
        [0x08, 0x2e, 0xa1, 0x66, 0x28, 0xd9, 0x24, 0xb2, 0x76, 0x5b, 0xa2, 0x49, 0x6d, 0x8b, 0xd1, 0x25],
        [0x72, 0xf8, 0xf6, 0x64, 0x86, 0x68, 0x98, 0x16, 0xd4, 0xa4, 0x5c, 0xcc, 0x5d, 0x65, 0xb6, 0x92],
        [0x6c, 0x70, 0x48, 0x50, 0xfd, 0xed, 0xb9, 0xda, 0x5e, 0x15, 0x46, 0x57, 0xa7, 0x8d, 0x9d, 0x84],
        [0x90, 0xd8, 0xab, 0x00, 0x8c, 0xbc, 0xd3, 0x0a, 0xf7, 0xe4, 0x58, 0x05, 0xb8, 0xb3, 0x45, 0x06],
        [0xd0, 0x2c, 0x1e, 0x8f, 0xca, 0x3f, 0x0f, 0x02, 0xc1, 0xaf, 0xbd, 0x03, 0x01, 0x13, 0x8a, 0x6b],
        [0x3a, 0x91, 0x11, 0x41, 0x4f, 0x67, 0xdc, 0xea, 0x97, 0xf2, 0xcf, 0xce, 0xf0, 0xb4, 0xe6, 0x73],
        [0x96, 0xac, 0x74, 0x22, 0xe7, 0xad, 0x35, 0x85, 0xe2, 0xf9, 0x37, 0xe8, 0x1c, 0x75, 0xdf, 0x6e],
        [0x47, 0xf1, 0x1a, 0x71, 0x1d, 0x29, 0xc5, 0x89, 0x6f, 0xb7, 0x62, 0x0e, 0xaa, 0x18, 0xbe, 0x1b],
        [0xfc, 0x56, 0x3e, 0x4b, 0xc6, 0xd2, 0x79, 0x20, 0x9a, 0xdb, 0xc0, 0xfe, 0x78, 0xcd, 0x5a, 0xf4],
        [0x1f, 0xdd, 0xa8, 0x33, 0x88, 0x07, 0xc7, 0x31, 0xb1, 0x12, 0x10, 0x59, 0x27, 0x80, 0xec, 0x5f],
        [0x60, 0x51, 0x7f, 0xa9, 0x19, 0xb5, 0x4a, 0x0d, 0x2d, 0xe5, 0x7a, 0x9f, 0x93, 0xc9, 0x9c, 0xef],
        [0xa0, 0xe0, 0x3b, 0x4d, 0xae, 0x2a, 0xf5, 0xb0, 0xc8, 0xeb, 0xbb, 0x3c, 0x83, 0x53, 0x99, 0x61],
        [0x17, 0x2b, 0x04, 0x7e, 0xba, 0x77, 0xd6, 0x26, 0xe1, 0x69, 0x14, 0x63, 0x55, 0x21, 0x0c, 0x7d]
    ], dtype=np.uint8)
    B = inv_s_box[A // 0x10, A % 0x10]
    return B

def shiftRows(A):
    B = np.array([np.roll(row, -i) for i, row in enumerate(A)], dtype=np.uint8)
    return B

def invShiftRows(A):
    B = np.array([np.roll(row, i) for i, row in enumerate(A)], dtype=np.uint8)
    return B

def galois_multiplication(a, b):
    """Perform Galois multiplication of two bytes."""
    p = 0
    for i in range(8):
        if b & 1:
            p ^= a
        hi_bit_set = a & 0x80
        a <<= 1
        if hi_bit_set:
            a ^= 0x1b  # x^8 + x^4 + x^3 + x + 1
        b >>= 1
    return p % 256

def mixCol(A):
    """Mix columns using Galois Field multiplication."""
    B = np.zeros((4,4), dtype=int)
    for c in range(4):
        col = A[:, c]
        B[:, c] = [
            galois_multiplication(0x02, col[0]) ^ galois_multiplication(0x03, col[1]) ^ col[2] ^ col[3],
            col[0] ^ galois_multiplication(0x02, col[1]) ^ galois_multiplication(0x03, col[2]) ^ col[3],
            col[0] ^ col[1] ^ galois_multiplication(0x02, col[2]) ^ galois_multiplication(0x03, col[3]),
            galois_multiplication(0x03, col[0]) ^ col[1] ^ col[2] ^ galois_multiplication(0x02, col[3])
        ]
    return B

def invMixCol(A):
    """Inverse mix columns using Galois Field multiplication."""
    B = np.zeros((4,4), dtype=int)
    for c in range(4):
        col = A[:, c]
        B[:, c] = [
            galois_multiplication(0x0e, col[0]) ^ galois_multiplication(0x0b, col[1]) ^ galois_multiplication(0x0d, col[2]) ^ galois_multiplication(0x09, col[3]),
            galois_multiplication(0x09, col[0]) ^ galois_multiplication(0x0e, col[1]) ^ galois_multiplication(0x0b, col[2]) ^ galois_multiplication(0x0d, col[3]),
            galois_multiplication(0x0d, col[0]) ^ galois_multiplication(0x09, col[1]) ^ galois_multiplication(0x0e, col[2]) ^ galois_multiplication(0x0b, col[3]),
            galois_multiplication(0x0b, col[0]) ^ galois_multiplication(0x0d, col[1]) ^ galois_multiplication(0x09, col[2]) ^ galois_multiplication(0x0e, col[3])
        ]
    return B

def addRoundKey(A, key):
    return np.bitwise_xor(A, key)

def keyExpansion(key):
    # Placeholder for the key expansion implementation
    # This should expand the initial key to an array of 176 bytes (11 keys of 16 bytes each for AES-128)
    expanded_key = np.zeros((176), dtype=np.uint8)
    # Implement the actual key expansion logic here
    return expanded_key.reshape((11, 4, 4))  # Reshape for easier use in AES rounds

def aesEncrypt(plain_text, key):
    key_matrix = text2Unicode(key)
    expanded_keys = keyExpansion(key_matrix.flatten())  # Flatten for simplicity in key expansion logic
    
    text_matrix = text2Unicode(plain_text)
    text_matrix = addRoundKey(text_matrix, expanded_keys[0])
    
    for round in range(1, 10):  # 9 rounds with all steps
        text_matrix = subBytes(text_matrix)
        text_matrix = shiftRows(text_matrix)
        text_matrix = mixCol(text_matrix)
        text_matrix = addRoundKey(text_matrix, expanded_keys[round])
    
    # Final round without mixColumns
    text_matrix = subBytes(text_matrix)
    text_matrix = shiftRows(text_matrix)
    text_matrix = addRoundKey(text_matrix, expanded_keys[10])
    
    cipher_text = unicode2Text(text_matrix)
    return cipher_text

def aesDecrypt(cipher_text, key):
    key_matrix = text2Unicode(key)
    expanded_keys = keyExpansion(key_matrix.flatten())
    
    cipher_matrix = text2Unicode(cipher_text)
    cipher_matrix = addRoundKey(cipher_matrix, expanded_keys[10])
    
    for round in range(9, 0, -1):  # Reverse order for decryption
        cipher_matrix = invShiftRows(cipher_matrix)
        cipher_matrix = invSubBytes(cipher_matrix)
        cipher_matrix = addRoundKey(cipher_matrix, expanded_keys[round])
        cipher_matrix = invMixCol(cipher_matrix)
    
    # Final round without invMixColumns
    cipher_matrix = invShiftRows(cipher_matrix)
    cipher_matrix = invSubBytes(cipher_matrix)
    cipher_matrix = addRoundKey(cipher_matrix, expanded_keys[0])
    
    decrypted_text = unicode2Text(cipher_matrix)
    return decrypted_text

def gf_add(a, b):
    return a ^ b 

def visualize_matrix(matrix, title):
    plt.figure(figsize=(6, 4))
    plt.imshow(matrix, cmap='viridis', interpolation='nearest')
    plt.title(title)
    plt.colorbar()
    plt.show()

def visualize_sbox_substitution(input_matrix, output_matrix):
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))

    # Input matrix visualization
    axs[0].imshow(input_matrix, cmap='viridis', interpolation='nearest')
    axs[0].set_title('Input Matrix (Before SubBytes)')
    axs[0].set_xticks(np.arange(0, 4, 1))
    axs[0].set_yticks(np.arange(0, 4, 1))

    # Output matrix visualization
    axs[1].imshow(output_matrix, cmap='viridis', interpolation='nearest')
    axs[1].set_title('Output Matrix (After SubBytes)')
    axs[1].set_xticks(np.arange(0, 4, 1))
    axs[1].set_yticks(np.arange(0, 4, 1))

    # Highlight differences between input and output
    for i in range(4):
        for j in range(4):
            if input_matrix[i, j] != output_matrix[i, j]:
                axs[0].text(j, i, f'{input_matrix[i, j]:02X}', ha='center', va='center', color='red', fontsize=10)
                axs[1].text(j, i, f'{output_matrix[i, j]:02X}', ha='center', va='center', color='red', fontsize=10)

    plt.tight_layout()
    plt.show()

def visualize_shiftrows(input_matrix, output_matrix):
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))

    # Input matrix visualization
    axs[0].imshow(input_matrix, cmap='viridis', interpolation='nearest')
    axs[0].set_title('Input Matrix (Before ShiftRows)')
    axs[0].set_xticks(np.arange(0, 4, 1))
    axs[0].set_yticks(np.arange(0, 4, 1))

    # Output matrix visualization
    axs[1].imshow(output_matrix, cmap='viridis', interpolation='nearest')
    axs[1].set_title('Output Matrix (After ShiftRows)')
    axs[1].set_xticks(np.arange(0, 4, 1))
    axs[1].set_yticks(np.arange(0, 4, 1))

    # Highlight differences between input and output
    for i in range(4):
        for j in range(4):
            if input_matrix[i, j] != output_matrix[i, j]:
                axs[0].text(j, i, f'{input_matrix[i, j]:02X}', ha='center', va='center', color='red', fontsize=10)
                axs[1].text(j, i, f'{output_matrix[i, j]:02X}', ha='center', va='center', color='red', fontsize=10)

    plt.tight_layout()
    plt.show()

def visualize_mixcolumns(input_matrix, output_matrix):
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))

    # Input matrix visualization
    axs[0].imshow(input_matrix, cmap='viridis', interpolation='nearest')
    axs[0].set_title('Input Matrix (Before MixColumns)')
    axs[0].set_xticks(np.arange(0, 4, 1))
    axs[0].set_yticks(np.arange(0, 4, 1))

    # Output matrix visualization
    axs[1].imshow(output_matrix, cmap='viridis', interpolation='nearest')
    axs[1].set_title('Output Matrix (After MixColumns)')
    axs[1].set_xticks(np.arange(0, 4, 1))
    axs[1].set_yticks(np.arange(0, 4, 1))

    # Highlight differences between input and output
    for i in range(4):
        for j in range(4):
            if input_matrix[i, j] != output_matrix[i, j]:
                axs[0].text(j, i, f'{input_matrix[i, j]:02X}', ha='center', va='center', color='red', fontsize=10)
                axs[1].text(j, i, f'{output_matrix[i, j]:02X}', ha='center', va='center', color='red', fontsize=10)

    plt.tight_layout()
    plt.show()

def visualize_add_round_key(state_matrix, round_key_matrix):
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    # State matrix visualization
    axs[0].imshow(state_matrix, cmap='viridis', interpolation='nearest')
    axs[0].set_title('State Matrix (Before AddRoundKey)')
    axs[0].set_xticks(np.arange(0, 4, 1))
    axs[0].set_yticks(np.arange(0, 4, 1))

    # Round key matrix visualization
    axs[1].imshow(round_key_matrix, cmap='viridis', interpolation='nearest')
    axs[1].set_title('Round Key Matrix')
    axs[1].set_xticks(np.arange(0, 4, 1))
    axs[1].set_yticks(np.arange(0, 4, 1))

    # Resultant matrix visualization after adding round key
    result_matrix = np.bitwise_xor(state_matrix, round_key_matrix)
    axs[2].imshow(result_matrix, cmap='viridis', interpolation='nearest')
    axs[2].set_title('Result Matrix (After AddRoundKey)')
    axs[2].set_xticks(np.arange(0, 4, 1))
    axs[2].set_yticks(np.arange(0, 4, 1))

    # Highlight differences between state matrix and round key matrix
    for i in range(4):
        for j in range(4):
            if state_matrix[i, j] != result_matrix[i, j]:
                axs[0].text(j, i, f'{state_matrix[i, j]:02X}', ha='center', va='center', color='red', fontsize=10)
                axs[1].text(j, i, f'{round_key_matrix[i, j]:02X}', ha='center', va='center', color='red', fontsize=10)
                axs[2].text(j, i, f'{result_matrix[i, j]:02X}', ha='center', va='center', color='red', fontsize=10)

    plt.tight_layout()
    plt.show()

def visualize_gf_operations():
    # Visualize addition in GF(2^8)
    gf_addition = np.zeros((256, 256), dtype=np.uint8)
    for i in range(256):
        gf_addition[:, i] = np.bitwise_xor(np.arange(256, dtype=np.uint8), i)

    plt.figure(figsize=(8, 6))
    plt.imshow(gf_addition, cmap='viridis', interpolation='nearest')
    plt.title('Addition in GF(2^8)')
    plt.xlabel('Input Byte')
    plt.ylabel('Addition Constant')
    plt.colorbar(label='Result Byte')
    plt.show()

    # Visualize multiplication in GF(2^8)
    gf_multiplication = np.zeros((256, 256), dtype=np.uint8)
    for i in range(256):
        gf_multiplication[:, i] = np.array([galois_multiplication(i, j) for j in range(256)], dtype=np.uint8)



    plt.figure(figsize=(8, 6))
    plt.imshow(gf_multiplication, cmap='viridis', interpolation='nearest')
    plt.title('Multiplication in GF(2^8)')
    plt.xlabel('Input Byte')
    plt.ylabel('Multiplication Constant')
    plt.colorbar(label='Result Byte')
    plt.show()

def galois_op():
    # Define the elements of GF(2^8)
    elements = [i for i in range(256)]  # 0 to 255

    addition_graph = nx.DiGraph()
    for a in elements:
        for b in elements:
            addition_graph.add_edge(a, gf_add(a, b))

    # Create directed graph for multiplication table
    multiplication_graph = nx.DiGraph()
    for a in elements:
        for b in elements:
            multiplication_graph.add_edge(a, galois_multiplication(a, b))

    # Plot the graphs
    plt.figure(figsize=(12, 6))

    plt.subplot(121)
    plt.title('Addition Table (GF(2^8))')
    nx.draw(addition_graph, with_labels=True, node_size=300, font_size=8)

    plt.subplot(122)
    plt.title('Multiplication Table (GF(2^8))')
    nx.draw(multiplication_graph, with_labels=True, node_size=300, font_size=8)

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    plain_text = input("Enter a 16 character string to be encoded : ")
    cipher_key = input("Enter a 16 character long key for encryption : ")    
    print("Encrypting : ")    
    print("Original:", plain_text)
    
    encrypted = aesEncrypt(plain_text, cipher_key)
    print("Encrypted:", encrypted)
    
    decrypted = aesDecrypt(encrypted, cipher_key)
    print("Decrypted:", decrypted)
    K = text2Unicode(cipher_key)
    P = text2Unicode(plain_text)
    S = subBytes(P)
    T = shiftRows(S)
    R = mixCol(T)
    Q = addRoundKey(T,K)
    visualize_matrix(P, 'Plaintext Matrix')
    visualize_sbox_substitution(P,S)
    visualize_shiftrows(S, T)
    visualize_mixcolumns(T, R)
    visualize_add_round_key(R, K)
    visualize_gf_operations()
    galois_op()
    eq_system = sp.Matrix([P,S,T,R,Q])
    sp.pprint(eq_system)
