# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 20:09:00 2026

@author: pruthvi
"""

import numpy as np
import matplotlib.pyplot as plt


NUM_BITS = 100
SNR_dB = 10
USE_CODING = True

def add_hamming_parity(data_bits):
    d1, d2, d3, d4 = data_bits
    
    p1 = (d1 + d2 + d3) % 2
    p2 = (d1 + d2 + d4) % 2
    p3 = (d1 + d3 + d4) % 2
    
    return [d1, d2, d3, d4, p1, p2, p3]

def encode_with_hamming(bits):
    if len(bits) % 4 != 0:
        bits = np.append(bits, [0] * (4 - len(bits) % 4))
    
    encoded_bits = []
    for i in range(0, len(bits), 4):
        block = bits[i:i+4]
        encoded_block = add_hamming_parity(block)
        encoded_bits.extend(encoded_block)
    
    return np.array(encoded_bits)

def fix_errors_and_decode(received_bits):
    if len(received_bits) % 7 != 0:
        received_bits = received_bits[:len(received_bits) - len(received_bits) % 7]
    
    decoded_bits = []
    
    for i in range(0, len(received_bits), 7):
        block = received_bits[i:i+7]
        
        d1, d2, d3, d4, p1, p2, p3 = block
        
        s1 = (d1 + d2 + d3 + p1) % 2
        s2 = (d1 + d2 + d4 + p2) % 2
        s3 = (d1 + d3 + d4 + p3) % 2
        
        syndrome = [s1, s2, s3]
        
        if syndrome == [0,0,0]:
            decoded_bits.extend([d1, d2, d3, d4])
        else:
            if syndrome == [1,1,1]:
                d1 = 1 - d1
            elif syndrome == [1,1,0]:
                d2 = 1 - d2
            elif syndrome == [1,0,1]:
                d3 = 1 - d3
            elif syndrome == [0,1,1]:
                d4 = 1 - d4
            
            decoded_bits.extend([d1, d2, d3, d4])
    
    return np.array(decoded_bits)

def modulate_qpsk(bits):
    symbols = []
    
    for i in range(0, len(bits), 2):
        if i+1 >= len(bits):
            break
        
        b1, b2 = bits[i], bits[i+1]
        
        if b1 == 0 and b2 == 0:
            symbol = 1 + 1j
        elif b1 == 0 and b2 == 1:
            symbol = -1 + 1j
        elif b1 == 1 and b2 == 1:
            symbol = -1 - 1j
        else:
            symbol = 1 - 1j
        
        symbols.append(symbol)
    
    symbols = np.array(symbols) / np.sqrt(2)
    
    return symbols

def demodulate_qpsk(symbols):
    bits = []
    
    for sym in symbols:
        sym = sym * np.sqrt(2)
        
        if sym.real > 0 and sym.imag > 0:
            bits.extend([0, 0])
        elif sym.real < 0 and sym.imag > 0:
            bits.extend([0, 1])
        elif sym.real < 0 and sym.imag < 0:
            bits.extend([1, 1])
        else:
            bits.extend([1, 0])
    
    return np.array(bits)

def simulate_channel(symbols, snr_db):
    snr_linear = 10 ** (snr_db / 10)
    noise_power = 1.0 / snr_linear
    
    noise_real = np.sqrt(noise_power/2) * np.random.randn(len(symbols))
    noise_imag = np.sqrt(noise_power/2) * np.random.randn(len(symbols))
    noise = noise_real + 1j * noise_imag
    
    received = symbols + noise
    
    return received

print("=" * 50)
print("COMMUNICATION SYSTEM SIMULATION")
print("=" * 50)

original_bits = np.random.randint(0, 2, NUM_BITS)
print(f"\n1. Generated {len(original_bits)} random bits")

if USE_CODING:
    encoded_bits = encode_with_hamming(original_bits)
    print(f"2. Added error correction: {len(original_bits)} bits -> {len(encoded_bits)} bits")
else:
    encoded_bits = original_bits
    print(f"2. No error correction (direct transmission)")

tx_symbols = modulate_qpsk(encoded_bits)
print(f"3. Modulated {len(encoded_bits)} bits into {len(tx_symbols)} symbols")

rx_symbols = simulate_channel(tx_symbols, SNR_dB)
print(f"4. Sent through channel with SNR = {SNR_dB} dB")

demodulated_bits = demodulate_qpsk(rx_symbols)
print(f"5. Demodulated {len(rx_symbols)} symbols into {len(demodulated_bits)} bits")

if USE_CODING:
    decoded_bits = fix_errors_and_decode(demodulated_bits)
    print(f"6. Removed error correction: {len(demodulated_bits)} bits -> {len(decoded_bits)} bits")
    decoded_bits = decoded_bits[:len(original_bits)]
else:
    decoded_bits = demodulated_bits[:len(original_bits)]
    print(f"6. No error correction to remove")

num_errors = np.sum(original_bits != decoded_bits)
ber = num_errors / len(original_bits)
print(f"\n" + "=" * 50)
print(f"RESULTS:")
print(f"  Original bits: {len(original_bits)}")
print(f"  Received bits: {len(decoded_bits)}")
print(f"  Errors: {num_errors}")
print(f"  Bit Error Rate (BER): {ber:.2e}")
print("=" * 50)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(tx_symbols.real, tx_symbols.imag, c='blue', alpha=0.6)
plt.title('Transmitted Symbols (QPSK)')
plt.xlabel('Real Part')
plt.ylabel('Imaginary Part')
plt.grid(True)
plt.axis('equal')
plt.xlim(-2, 2)
plt.ylim(-2, 2)

plt.subplot(1, 2, 2)
plt.scatter(rx_symbols.real, rx_symbols.imag, c='red', alpha=0.6)
plt.title(f'Received Symbols (SNR = {SNR_dB} dB)')
plt.xlabel('Real Part')
plt.ylabel('Imaginary Part')
plt.grid(True)
plt.axis('equal')
plt.xlim(-2, 2)
plt.ylim(-2, 2)

plt.tight_layout()
plt.show()

def plot_ber_curve():
    print("\n" + "=" * 50)
    print("GENERATING BER CURVE...")
    print("=" * 50)
    
    snr_values = range(-5, 31, 5)
    ber_coded = []
    ber_uncoded = []
    
    for snr in snr_values:
        print(f"Testing SNR = {snr} dB...", end=" ")
        
        bits = np.random.randint(0, 2, 1000)
        encoded = encode_with_hamming(bits)
        tx = modulate_qpsk(encoded)
        rx = simulate_channel(tx, snr)
        demod = demodulate_qpsk(rx)
        decoded = fix_errors_and_decode(demod)[:len(bits)]
        errors = np.sum(bits != decoded)
        ber_coded.append(errors / len(bits))
        
        tx2 = modulate_qpsk(bits)
        rx2 = simulate_channel(tx2, snr)
        demod2 = demodulate_qpsk(rx2)[:len(bits)]
        errors2 = np.sum(bits != demod2)
        ber_uncoded.append(errors2 / len(bits))
        
        print(f"Coded BER = {ber_coded[-1]:.2e}, Uncoded BER = {ber_uncoded[-1]:.2e}")
    
    plt.figure(figsize=(10, 6))
    plt.semilogy(snr_values, ber_coded, 'b-o', label='With Error Correction', linewidth=2)
    plt.semilogy(snr_values, ber_uncoded, 'r-s', label='Without Error Correction', linewidth=2)
    plt.xlabel('SNR (dB)', fontsize=12)
    plt.ylabel('Bit Error Rate (BER)', fontsize=12)
    plt.title('BER vs SNR: With and Without Error Correction', fontsize=14)
    plt.grid(True, which='both', alpha=0.3)
    plt.legend(fontsize=12)
    plt.savefig('BER_SNR.png', dpi=150)
    plt.show()
    
    print("\nBER curve saved as 'BER_SNR.png'")


