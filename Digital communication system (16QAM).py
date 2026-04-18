# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 20:13:58 2026

@author: pruthvi
"""

import numpy as np
import matplotlib.pyplot as plt


usf_filter = 8

H = np.array([[1,1,1,0,1,0,0],
              [1,1,0,1,0,1,0],
              [1,0,1,1,0,0,1]])


def encode(bits):
    while len(bits) % 4 != 0:
        bits = np.append(bits, 0)
    
    output = []
    for i in range(0, len(bits), 4):
        d = bits[i:i+4]
        
        p1 = (d[0] + d[1] + d[2]) % 2
        p2 = (d[0] + d[1] + d[3]) % 2
        p3 = (d[0] + d[2] + d[3]) % 2
        
        output.extend([d[0], d[1], d[2], d[3], p1, p2, p3])
    
    return np.array(output)


def decode(bits):
    while len(bits) % 7 != 0:
        bits = bits[:-1]
    
    output = []
    for i in range(0, len(bits), 7):
        c = bits[i:i+7].copy()
        
        s1 = (c[0] + c[1] + c[2] + c[4]) % 2
        s2 = (c[0] + c[1] + c[3] + c[5]) % 2
        s3 = (c[0] + c[2] + c[3] + c[6]) % 2
        
        if s1 == 1 and s2 == 1 and s3 == 1:
            c[0] = 1 - c[0]
        elif s1 == 1 and s2 == 1 and s3 == 0:
            c[1] = 1 - c[1]
        elif s1 == 1 and s2 == 0 and s3 == 1:
            c[2] = 1 - c[2]
        elif s1 == 0 and s2 == 1 and s3 == 1:
            c[3] = 1 - c[3]
        
        output.extend(c[0:4])
    
    return np.array(output)


def modulate(bits):
    while len(bits) % 4 != 0:
        bits = np.append(bits, 0)
    
    symbols = []
    for i in range(0, len(bits), 4):
        b = bits[i:i+4]
        
        if b[0] == 0 and b[1] == 0:
            R = -3
        elif b[0] == 0 and b[1] == 1:
            R = -1
        elif b[0] == 1 and b[1] == 1:
            R = 1
        else:
            R = 3
        
        if b[2] == 0 and b[3] == 0:
            I = -3
        elif b[2] == 0 and b[3] == 1:
            I = -1
        elif b[2] == 1 and b[3] == 1:
            I = 1
        else:
            I = 3
        
        symbols.append(R + 1j*I)
    
    symbols = np.array(symbols) / np.sqrt(10)
    
    return symbols


def demodulate(symbols):
    symbols = symbols * np.sqrt(10)
    
    bits = []
    for sym in symbols:
        R = sym.real
        I = sym.imag
        
        if R < -2:
            bits.extend([0, 0])
        elif R < 0:
            bits.extend([0, 1])
        elif R < 2:
            bits.extend([1, 1])
        else:
            bits.extend([1, 0])
        
        if I < -2:
            bits.extend([0, 0])
        elif I < 0:
            bits.extend([0, 1])
        elif I < 2:
            bits.extend([1, 1])
        else:
            bits.extend([1, 0])
    
    return np.array(bits)


def tx_filter(symbols):
    L = usf_filter
    
    upsampled = np.zeros(len(symbols) * L, dtype=complex)
    upsampled[::L] = symbols
    
    filter_size = 16
    smooth = np.ones(filter_size) / filter_size
    
    output = np.convolve(upsampled, smooth, mode='same')
    
    output = output / np.sqrt(np.mean(np.abs(output)**2))
    
    return output


def rx_filter(signal):
    L = usf_filter
    
    filter_size = 16
    smooth = np.ones(filter_size) / filter_size
    
    filtered = np.convolve(signal, smooth, mode='same')
    
    delay = filter_size // 2
    
    symbols = filtered[delay::L]
    
    symbols = symbols * L
    
    expected_len = len(signal) // L
    symbols = symbols[:expected_len]
    
    return symbols


def channel(signal, snr_db):
    distance = 36000e3
    frequency = 20e9
    speed_light = 3e8
    antenna_gain = 40
    
    path_loss_db = 20 * np.log10(4 * np.pi * distance * frequency / speed_light) - 80
    path_loss = 10 ** (-path_loss_db / 20)
    
    signal = signal * path_loss
    
    signal_power = path_loss**2
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    
    noise = np.sqrt(noise_power/2) * (np.random.randn(len(signal)) + 1j*np.random.randn(len(signal)))
    
    return signal + noise


def calculate_ber(original, received):
    min_len = min(len(original), len(received))
    errors = np.sum(original[:min_len] != received[:min_len])
    return errors / min_len


def test_one_snr():
    snr = 15
    num_bits = 1000
    
    print(f"\n--- TESTING SNR = {snr} dB ---")
    
    original = np.random.randint(0, 2, num_bits)
    
    encoded = encode(original)
    tx_symbols = modulate(encoded)
    tx_signal = tx_filter(tx_symbols)
    rx_signal = channel(tx_signal, snr)
    rx_symbols = rx_filter(rx_signal)
    rx_bits_coded = demodulate(rx_symbols)
    decoded = decode(rx_bits_coded)
    
    tx_uncoded = modulate(original[:len(original)-len(original)%4])
    tx_signal_unc = tx_filter(tx_uncoded)
    rx_signal_unc = channel(tx_signal_unc, snr)
    rx_symbols_unc = rx_filter(rx_signal_unc)
    rx_bits_uncoded = demodulate(rx_symbols_unc)
    
    ber_coded = calculate_ber(original, decoded)
    ber_uncoded = calculate_ber(original[:len(rx_bits_uncoded)], rx_bits_uncoded)
    
    print(f"Coded BER:   {ber_coded:.2e}")
    print(f"Uncoded BER: {ber_uncoded:.2e}")
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(tx_symbols.real, tx_symbols.imag, 'bo', alpha=0.5)
    plt.title('Transmitted Symbols (16-QAM)')
    plt.axis('equal')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(rx_symbols.real, rx_symbols.imag, 'ro', alpha=0.5)
    plt.title(f'Received Symbols (SNR = {snr} dB)')
    plt.axis('equal')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()


def run_ber_sweep():
    snr_values = range(-5, 35, 5)
    num_bits = 10000
    
    ber_coded = []
    ber_uncoded = []
    
    print("\n" + "="*50)
    print("RUNNING BER SWEEP")
    print("="*50)
    
    for snr in snr_values:
        print(f"SNR = {snr:3d} dB...", end=" ")
        
        bits = np.random.randint(0, 2, num_bits)
        
        enc = encode(bits)
        tx_c = modulate(enc)
        tx_sig_c = tx_filter(tx_c)
        rx_sig_c = channel(tx_sig_c, snr)
        rx_sym_c = rx_filter(rx_sig_c)
        rx_bit_c = demodulate(rx_sym_c)
        dec = decode(rx_bit_c)
        err_c = calculate_ber(bits, dec)
        ber_coded.append(err_c)
        
        bits_unc = bits[:len(bits)-len(bits)%4]
        tx_u = modulate(bits_unc)
        tx_sig_u = tx_filter(tx_u)
        rx_sig_u = channel(tx_sig_u, snr)
        rx_sym_u = rx_filter(rx_sig_u)
        rx_bit_u = demodulate(rx_sym_u)
        err_u = calculate_ber(bits_unc, rx_bit_u)
        ber_uncoded.append(err_u)
        
        print(f"Coded: {err_c:.2e}, Uncoded: {err_u:.2e}")
    
    plt.figure(figsize=(10, 6))
    plt.semilogy(snr_values, ber_coded, 'b-o', label='With Hamming Code', linewidth=2)
    plt.semilogy(snr_values, ber_uncoded, 'r-s', label='Without Coding', linewidth=2)
    plt.xlabel('SNR (dB)', fontsize=12)
    plt.ylabel('Bit Error Rate (BER)', fontsize=12)
    plt.title('BER vs SNR: 16-QAM with and without Hamming Code', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.savefig('BER_SNR.png', dpi=150)
    plt.show()
    
    print("\n" + "="*50)
    print("BER curve saved as 'BER_SNR.png'")
    print("="*50)


if __name__ == "__main__":
    
    test_one_snr()
    # run_ber_sweep()
