import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import networkx as nx


def preparetext(text):
    text = text.ljust(16)[:16]
    textm = np.array([ord(c) for c in text], dtype=np.uint8).reshape(4, 4)
    return textm

def matrixtotext(matrix):
    text = ''.join(chr(int(c)) for c in matrix.flatten())
    return text

def subBytes(A):
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
# Take one byte input and get a byte from the S-box.
# It is assumed that the input 'A' is an integer (byte value).
# Using matrix indexing, we locate the appropriate element where:
#: The leftmost four bits of 'A' decide the row; the rightmost four bits of 'A' determine the column.
# This is accomplished by taking 'A' modulo 16 for the column (mask off the top bits) and dividing it by 16 for the row (shift right 4 bits).

    B = s_box[A // 0x10, A % 0x10]

    return B

def invSubBytes(A):
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
# Getting a byte from the inverse S-box # The input byte 'A' is what we need to determine its inverse transformation for.
# To find the row, we first divide 'A' by 16 (or shift 4 bits to the right).
# Next, we calculate the column by dividing 'A' by 16 and finding the remainder.
# The top and lower 4 bits of the byte are extracted as the foundation for this indexing.
    B = inv_s_box[A // 0x10, A % 0x10]

    return B

def shiftRows(A):
    shiftedrows = []
    for j, row in enumerate(A):
        newrow = np.roll(row, -j)
        shiftedrows.append(newrow)
    B = np.array(shiftedrows, dtype=np.uint8)
    return B

def invShiftRows(A):
    shiftedbackrows = []
    for j, row in enumerate(A):
       
        newrow = np.roll(row, j)
        shiftedbackrows.append(newrow)
    B = np.array(shiftedbackrows, dtype=np.uint8)
    return B

def multgal(a, b):
    p = 0
    for i in range(8):
        if b & 1:
            p ^= a
        hi_bit_set = a & 0x80
        a <<= 1
        if hi_bit_set:
            a ^= 0x1b  
        b >>= 1
    return p % 256

def mixCol(A):
# galois field mulyt
    B = np.zeros((4,4), dtype=int)
    for c in range(4):
        col = A[:, c]
        B[:, c] = [
            multgal(0x02, col[0]) ^ multgal(0x03, col[1]) ^ col[2] ^ col[3],
            col[0] ^ multgal(0x02, col[1]) ^ multgal(0x03, col[2]) ^ col[3],
            col[0] ^ col[1] ^ multgal(0x02, col[2]) ^ multgal(0x03, col[3]),
            multgal(0x03, col[0]) ^ col[1] ^ col[2] ^ multgal(0x02, col[3])
        ]
    return B

def invMixCol(A):
    B = np.zeros((4,4), dtype=int)
    for c in range(4):
        col = A[:, c]
        B[:, c] = [
            multgal(0x0e, col[0]) ^ multgal(0x0b, col[1]) ^ multgal(0x0d, col[2]) ^ multgal(0x09, col[3]),
            multgal(0x09, col[0]) ^ multgal(0x0e, col[1]) ^ multgal(0x0b, col[2]) ^ multgal(0x0d, col[3]),
            multgal(0x0d, col[0]) ^ multgal(0x09, col[1]) ^ multgal(0x0e, col[2]) ^ multgal(0x0b, col[3]),
            multgal(0x0b, col[0]) ^ multgal(0x0d, col[1]) ^ multgal(0x09, col[2]) ^ multgal(0x0e, col[3])
        ]
    return B

def addRoundKey(A, key):
    return np.bitwise_xor(A, key)

def keyExpansion(key):
    expanded_key = np.zeros((176), dtype=np.uint8)

    return expanded_key.reshape((11, 4, 4)) 

def encrypt(plain_text, key):
    keym = preparetext(key)
    expandedkey = keyExpansion(keym.flatten()) 
    
    textm = preparetext(plain_text)
    textm = addRoundKey(textm, expandedkey[0])
    
    for round in range(1, 10): 
        textm = subBytes(textm)
        textm = shiftRows(textm)
        textm = mixCol(textm)
        textm = addRoundKey(textm, expandedkey[round])
    
    textm = subBytes(textm)
    textm = shiftRows(textm)
    textm = addRoundKey(textm, expandedkey[10])
    
    ctext = matrixtotext(textm)
    return ctext

def decrypt(ctext, key):
    keym = preparetext(key)
    expandedkey = keyExpansion(keym.flatten())
    
    cmatrix = preparetext(ctext)
    cmatrix = addRoundKey(cmatrix, expandedkey[10])
    
    for round in range(9, 0, -1):  
        cmatrix = invShiftRows(cmatrix)
        cmatrix = invSubBytes(cmatrix)
        cmatrix = addRoundKey(cmatrix, expandedkey[round])
        cmatrix = invMixCol(cmatrix)
    
    
    cmatrix = invShiftRows(cmatrix)
    cmatrix = invSubBytes(cmatrix)
    cmatrix = addRoundKey(cmatrix, expandedkey[0])
    
    dectext = matrixtotext(cmatrix)
    return dectext

def gf_add(a, b):
    return a ^ b 

def plaintextmat(matrix, title):
    plt.figure(figsize=(6, 4))
    plt.imshow(matrix, cmap='viridis', interpolation='nearest')
    plt.title(title)
    plt.colorbar()
    plt.show()

def sboxvisual(input_matrix, output_matrix):
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))

    axs[0].imshow(input_matrix, cmap='viridis', interpolation='nearest')
    axs[0].set_title('Input Matrix (Before SubBytes)')
    axs[0].set_xticks(np.arange(0, 4, 1))
    axs[0].set_yticks(np.arange(0, 4, 1))


    axs[1].imshow(output_matrix, cmap='viridis', interpolation='nearest')
    axs[1].set_title('Output Matrix (After SubBytes)')
    axs[1].set_xticks(np.arange(0, 4, 1))
    axs[1].set_yticks(np.arange(0, 4, 1))


    for i in range(4):
        for j in range(4):
            if input_matrix[i, j] != output_matrix[i, j]:
                axs[0].text(j, i, f'{input_matrix[i, j]:02X}', ha='center', va='center', color='red', fontsize=10)
                axs[1].text(j, i, f'{output_matrix[i, j]:02X}', ha='center', va='center', color='red', fontsize=10)

    plt.tight_layout()
    plt.show()

def shiftrowvis(input_matrix, output_matrix):
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))


    axs[0].imshow(input_matrix, cmap='viridis', interpolation='nearest')
    axs[0].set_title('Input Matrix (Before ShiftRows)')
    axs[0].set_xticks(np.arange(0, 4, 1))
    axs[0].set_yticks(np.arange(0, 4, 1))

  
    axs[1].imshow(output_matrix, cmap='viridis', interpolation='nearest')
    axs[1].set_title('Output Matrix (After ShiftRows)')
    axs[1].set_xticks(np.arange(0, 4, 1))
    axs[1].set_yticks(np.arange(0, 4, 1))

  
    for i in range(4):
        for j in range(4):
            if input_matrix[i, j] != output_matrix[i, j]:
                axs[0].text(j, i, f'{input_matrix[i, j]:02X}', ha='center', va='center', color='red', fontsize=10)
                axs[1].text(j, i, f'{output_matrix[i, j]:02X}', ha='center', va='center', color='red', fontsize=10)

    plt.tight_layout()
    plt.show()

def mixcolvis(input_matrix, output_matrix):
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))

    
    axs[0].imshow(input_matrix, cmap='viridis', interpolation='nearest')
    axs[0].set_title('Input Matrix (Before MixColumns)')
    axs[0].set_xticks(np.arange(0, 4, 1))
    axs[0].set_yticks(np.arange(0, 4, 1))

   
    axs[1].imshow(output_matrix, cmap='viridis', interpolation='nearest')
    axs[1].set_title('Output Matrix (After MixColumns)')
    axs[1].set_xticks(np.arange(0, 4, 1))
    axs[1].set_yticks(np.arange(0, 4, 1))

    
    for i in range(4):
        for j in range(4):
            if input_matrix[i, j] != output_matrix[i, j]:
                axs[0].text(j, i, f'{input_matrix[i, j]:02X}', ha='center', va='center', color='red', fontsize=10)
                axs[1].text(j, i, f'{output_matrix[i, j]:02X}', ha='center', va='center', color='red', fontsize=10)

    plt.tight_layout()
    plt.show()

def addroundvis(state_matrix, round_keym):
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

   
    axs[0].imshow(state_matrix, cmap='viridis', interpolation='nearest')
    axs[0].set_title('State Matrix (Before AddRoundKey)')
    axs[0].set_xticks(np.arange(0, 4, 1))
    axs[0].set_yticks(np.arange(0, 4, 1))

   
    axs[1].imshow(round_keym, cmap='viridis', interpolation='nearest')
    axs[1].set_title('Round Key Matrix')
    axs[1].set_xticks(np.arange(0, 4, 1))
    axs[1].set_yticks(np.arange(0, 4, 1))

   
    result_matrix = np.bitwise_xor(state_matrix, round_keym)
    axs[2].imshow(result_matrix, cmap='viridis', interpolation='nearest')
    axs[2].set_title('Result Matrix (After AddRoundKey)')
    axs[2].set_xticks(np.arange(0, 4, 1))
    axs[2].set_yticks(np.arange(0, 4, 1))

   
    for i in range(4):
        for j in range(4):
            if state_matrix[i, j] != result_matrix[i, j]:
                axs[0].text(j, i, f'{state_matrix[i, j]:02X}', ha='center', va='center', color='red', fontsize=10)
                axs[1].text(j, i, f'{round_keym[i, j]:02X}', ha='center', va='center', color='red', fontsize=10)
                axs[2].text(j, i, f'{result_matrix[i, j]:02X}', ha='center', va='center', color='red', fontsize=10)

    plt.tight_layout()
    plt.show()

def gfaddmultvis():
    
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

    
    gf_multiplication = np.zeros((256, 256), dtype=np.uint8)
    for i in range(256):
        gf_multiplication[:, i] = np.array([multgal(i, j) for j in range(256)], dtype=np.uint8)



    plt.figure(figsize=(8, 6))
    plt.imshow(gf_multiplication, cmap='viridis', interpolation='nearest')
    plt.title('Multiplication in GF(2^8)')
    plt.xlabel('Input Byte')
    plt.ylabel('Multiplication Constant')
    plt.colorbar(label='Result Byte')
    plt.show()

def operationgal():
    
    elements = [i for i in range(256)] 

    addition_graph = nx.DiGraph()
    for a in elements:
        for b in elements:
            addition_graph.add_edge(a, gf_add(a, b))

    
    multiplication_graph = nx.DiGraph()
    for a in elements:
        for b in elements:
            multiplication_graph.add_edge(a, multgal(a, b))

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
    
    encrypted = encrypt(plain_text, cipher_key)
    print("Encrypted:", encrypted)
    
    decrypted = decrypt(encrypted, cipher_key)
    print("Decrypted:", decrypted)
    K = preparetext(cipher_key)
    P = preparetext(plain_text)
    S = subBytes(P)
    T = shiftRows(S)
    R = mixCol(T)
    Q = addRoundKey(T,K)
    plaintextmat(P, 'Plaintext Matrix')
    sboxvisual(P,S)
    shiftrowvis(S, T)
    mixcolvis(T, R)
    addroundvis(R, K)
    gfaddmultvis()
    operationgal()
    eq_system = sp.Matrix([P,S,T,R,Q])
    sp.pprint(eq_system)
