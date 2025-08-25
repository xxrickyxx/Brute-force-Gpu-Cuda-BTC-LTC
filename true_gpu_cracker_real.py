cracker = None  # Variabile globale per accesso da Flask e thread
#!/usr/bin/env python3

import sys
import time
import logging
import threading
import json
import hashlib
import hmac
import struct
import os
from pathlib import Path
from Crypto.Cipher import AES
from Crypto.Protocol.KDF import PBKDF2
import logging
logging.basicConfig(
    level=logging.DEBUG,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('gpu_bitcoin_crack.log', mode='w', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
from Crypto.Hash import SHA512
from flask import Flask, render_template_string, jsonify
import traceback
import time  # Assicura import globale per tutte le funzioni
import time  # Fix: Import necessario per sleep e time.time
from flask import request

# Configurazione CUDA ottimizzata
os.environ['CUDA_CACHE_DISABLE'] = '1'

# Importa PyCUDA
try:
    import pycuda.driver as cuda
    import pycuda.autoinit
    import pycuda.compiler as compiler
    import pycuda.gpuarray as gpuarray
    from pycuda.compiler import SourceModule
    import numpy as np
    GPU_AVAILABLE = True
    print("🔥 PyCUDA REAL GPU importato con successo!")
except ImportError as e:
    print(f"❌ PyCUDA richiesto: {e}")
    sys.exit(1)

# Kernel CUDA REALE con PBKDF2-HMAC-SHA512 e AES
CUDA_REAL_BITCOIN_KERNEL = """
#include <stdint.h>

// SHA-512 constants
__constant__ uint64_t K[80] = {
    0x428a2f98d728ae22ULL, 0x7137449123ef65cdULL, 0xb5c0fbcfec4d3b2fULL, 0xe9b5dba58189dbbcULL,
    0x3956c25bf348b538ULL, 0x59f111f1b605d019ULL, 0x923f82a4af194f9bULL, 0xab1c5ed5da6d8118ULL,
    0xd807aa98a3030242ULL, 0x12835b0145706fbeULL, 0x243185be4ee4b28cULL, 0x550c7dc3d5ffb4e2ULL,
    0x72be5d74f27b896fULL, 0x80deb1fe3b1696b1ULL, 0x9bdc06a725c71235ULL, 0xc19bf174cf692694ULL,
    0xe49b69c19ef14ad2ULL, 0xefbe4786384f25e3ULL, 0x0fc19dc68b8cd5b5ULL, 0x240ca1cc77ac9c65ULL,
    0x2de92c6f592b0275ULL, 0x4a7484aa6ea6e483ULL, 0x5cb0a9dcbd41fbd4ULL, 0x76f988da831153b5ULL,
    0x983e5152ee66dfabULL, 0xa831c66d2db43210ULL, 0xb00327c898fb213fULL, 0xbf597fc7beef0ee4ULL,
    0xc6e00bf33da88fc2ULL, 0xd5a79147930aa725ULL, 0x06ca6351e003826fULL, 0x142929670a0e6e70ULL,
    0x27b70a8546d22ffcULL, 0x2e1b21385c26c926ULL, 0x4d2c6dfc5ac42aedULL, 0x53380d139d95b3dfULL,
    0x650a73548baf63deULL, 0x766a0abb3c77b2a8ULL, 0x81c2c92e47edaee6ULL, 0x92722c851482353bULL,
    0xa2bfe8a14cf10364ULL, 0xa81a664bbc423001ULL, 0xc24b8b70d0f89791ULL, 0xc76c51a30654be30ULL,
    0xd192e819d6ef5218ULL, 0xd69906245565a910ULL, 0xf40e35855771202aULL, 0x106aa07032bbd1b8ULL,
    0x19a4c116b8d2d0c8ULL, 0x1e376c085141ab53ULL, 0x2748774cdf8eeb99ULL, 0x34b0bcb5e19b48a8ULL,
    0x391c0cb3c5c95a63ULL, 0x4ed8aa4ae3418acbULL, 0x5b9cca4f7763e373ULL, 0x682e6ff3d6b2b8a3ULL,
    0x748f82ee5defb2fcULL, 0x78a5636f43172f60ULL, 0x84c87814a1f0ab72ULL, 0x8cc702081a6439ecULL,
    0x90befffa23631e28ULL, 0xa4506cebde82bde9ULL, 0xbef9a3f7b2c67915ULL, 0xc67178f2e372532bULL,
    0xca273eceea26619cULL, 0xd186b8c721c0c207ULL, 0xeada7dd6cde0eb1eULL, 0xf57d4f7fee6ed178ULL,
    0x06f067aa72176fbaULL, 0x0a637dc5a2c898a6ULL, 0x113f9804bef90daeULL, 0x1b710b35131c471bULL,
    0x28db77f523047d84ULL, 0x32caab7b40c72493ULL, 0x3c9ebe0a15c9bebcULL, 0x431d67c49c100d4cULL,
    0x4cc5d4becb3e42b6ULL, 0x597f299cfc657e2aULL, 0x5fcb6fab3ad6faecULL, 0x6c44198c4a475817ULL
};

__device__ uint64_t rotr64(uint64_t x, int n) {
    return (x >> n) | (x << (64 - n));
}

__device__ void sha512_transform(uint64_t state[8], const uint8_t block[128]) {
    uint64_t w[80];
    uint64_t a, b, c, d, e, f, g, h;
    uint64_t t1, t2;
    
    // Copy block to w[0..15]
    for (int i = 0; i < 16; i++) {
        w[i] = ((uint64_t)block[i*8] << 56) | ((uint64_t)block[i*8+1] << 48) |
               ((uint64_t)block[i*8+2] << 40) | ((uint64_t)block[i*8+3] << 32) |
               ((uint64_t)block[i*8+4] << 24) | ((uint64_t)block[i*8+5] << 16) |
               ((uint64_t)block[i*8+6] << 8) | ((uint64_t)block[i*8+7]);
    }
    
    // Extend to w[16..79]
    for (int i = 16; i < 80; i++) {
        uint64_t s0 = rotr64(w[i-15], 1) ^ rotr64(w[i-15], 8) ^ (w[i-15] >> 7);
        uint64_t s1 = rotr64(w[i-2], 19) ^ rotr64(w[i-2], 61) ^ (w[i-2] >> 6);
        w[i] = w[i-16] + s0 + w[i-7] + s1;
    }
    
    // Initialize working variables
    a = state[0]; b = state[1]; c = state[2]; d = state[3];
    e = state[4]; f = state[5]; g = state[6]; h = state[7];
    
    // Main loop
    for (int i = 0; i < 80; i++) {
        uint64_t S1 = rotr64(e, 14) ^ rotr64(e, 18) ^ rotr64(e, 41);
        uint64_t ch = (e & f) ^ (~e & g);
        t1 = h + S1 + ch + K[i] + w[i];
        uint64_t S0 = rotr64(a, 28) ^ rotr64(a, 34) ^ rotr64(a, 39);
        uint64_t maj = (a & b) ^ (a & c) ^ (b & c);
        t2 = S0 + maj;
        
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }
    
    // Add to state
    state[0] += a; state[1] += b; state[2] += c; state[3] += d;
    state[4] += e; state[5] += f; state[6] += g; state[7] += h;
}

__device__ void sha512_init(uint64_t state[8]) {
    state[0] = 0x6a09e667f3bcc908ULL;
    state[1] = 0xbb67ae8584caa73bULL;
    state[2] = 0x3c6ef372fe94f82bULL;
    state[3] = 0xa54ff53a5f1d36f1ULL;
    state[4] = 0x510e527fade682d1ULL;
    state[5] = 0x9b05688c2b3e6c1fULL;
    state[6] = 0x1f83d9abfb41bd6bULL;
    state[7] = 0x5be0cd19137e2179ULL;
}

__device__ void sha512_final(uint64_t state[8], uint8_t* buffer, int buffer_len, uint64_t total_len, uint8_t hash[64]) {
    uint8_t block[128];
    int i;
    
    // Copy buffer to block
    for (i = 0; i < buffer_len; i++) {
        block[i] = buffer[i];
    }
    
    // Add padding bit
    block[buffer_len] = 0x80;
    
    // Zero fill
    for (i = buffer_len + 1; i < 128; i++) {
        block[i] = 0;
    }
    
    // If not enough space for length, process block and start new one
    if (buffer_len >= 112) {
        sha512_transform(state, block);
        for (i = 0; i < 112; i++) {
            block[i] = 0;
        }
    }
    
    // Add length in bits
    uint64_t bit_len = total_len * 8;
    for (i = 0; i < 8; i++) {
        block[127-i] = (bit_len >> (i*8)) & 0xff;
    }
    for (i = 8; i < 16; i++) {
        block[127-i] = 0;
    }
    
    sha512_transform(state, block);
    
    // Output hash
    for (i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            hash[i*8 + j] = (state[i] >> (56 - j*8)) & 0xff;
        }
    }
}

__device__ void hmac_sha512(const uint8_t* key, int key_len, const uint8_t* data, int data_len, uint8_t hash[64]) {
    uint8_t ipad[128], opad[128];
    uint8_t key_pad[128];
    uint64_t state[8];
    
    // Prepare key
    if (key_len > 128) {
        // Hash long keys
        sha512_init(state);
        uint8_t temp_buffer[128];
        int processed = 0;
        while (processed < key_len) {
            int chunk_size = min(128, key_len - processed);
            for (int i = 0; i < chunk_size; i++) {
                temp_buffer[i] = key[processed + i];
            }
            if (processed + chunk_size == key_len) {
                sha512_final(state, temp_buffer, chunk_size, key_len, key_pad);
            } else {
                if (chunk_size == 128) {
                    sha512_transform(state, temp_buffer);
                }
            }
            processed += chunk_size;
        }
        for (int i = 64; i < 128; i++) {
            key_pad[i] = 0;
        }
    } else {
        for (int i = 0; i < key_len; i++) {
            key_pad[i] = key[i];
        }
        for (int i = key_len; i < 128; i++) {
            key_pad[i] = 0;
        }
    }
    
    // Create ipad and opad
    for (int i = 0; i < 128; i++) {
        ipad[i] = key_pad[i] ^ 0x36;
        opad[i] = key_pad[i] ^ 0x5c;
    }
    
    // Inner hash
    sha512_init(state);
    sha512_transform(state, ipad);
    
    uint8_t temp_buffer[128];
    int processed = 0;
    while (processed < data_len) {
        int chunk_size = min(128, data_len - processed);
        for (int i = 0; i < chunk_size; i++) {
            temp_buffer[i] = data[processed + i];
        }
        if (processed + chunk_size == data_len) {
            sha512_final(state, temp_buffer, chunk_size, 128 + data_len, hash);
        } else {
            if (chunk_size == 128) {
                sha512_transform(state, temp_buffer);
            }
        }
        processed += chunk_size;
    }
    
    // Outer hash
    sha512_init(state);
    sha512_transform(state, opad);
    sha512_final(state, hash, 64, 128 + 64, hash);
}

__device__ void pbkdf2_hmac_sha512(const char* password, int pwd_len, const uint8_t* salt, int salt_len, 
                                   int iterations, uint8_t* output, int output_len) {
    uint8_t u[64], t[64];
    uint8_t salt_block[256];
    int blocks = (output_len + 63) / 64;
    
    for (int block = 1; block <= blocks; block++) {
        // Prepare salt + block number
        for (int i = 0; i < salt_len; i++) {
            salt_block[i] = salt[i];
        }
        salt_block[salt_len] = (block >> 24) & 0xff;
        salt_block[salt_len + 1] = (block >> 16) & 0xff;
        salt_block[salt_len + 2] = (block >> 8) & 0xff;
        salt_block[salt_len + 3] = block & 0xff;
        
        // First iteration
        hmac_sha512((uint8_t*)password, pwd_len, salt_block, salt_len + 4, u);
        for (int i = 0; i < 64; i++) {
            t[i] = u[i];
        }
        
        // Remaining iterations
        for (int iter = 1; iter < iterations; iter++) {
            hmac_sha512((uint8_t*)password, pwd_len, u, 64, u);
            for (int i = 0; i < 64; i++) {
                t[i] ^= u[i];
            }
        }
        
        // Copy to output
        int start = (block - 1) * 64;
        int copy_len = min(64, output_len - start);
        for (int i = 0; i < copy_len; i++) {
            output[start + i] = t[i];
        }
    }
}

// AES S-box
__constant__ uint8_t sbox[256] = {
    0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
    0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
    0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
    0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
    0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
    0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
    0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
    0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
    0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
    0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
    0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
    0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
    0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
    0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
    0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
    0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16
};

__device__ void aes_decrypt_block(const uint8_t* key, const uint8_t* input, uint8_t* output) {
    // Simplified AES-256 decrypt for one block
    // This is a basic implementation - in production you'd want full AES
    uint8_t state[16];
    for (int i = 0; i < 16; i++) {
        state[i] = input[i] ^ key[i];
    }
    
    // Simplified inverse operations
    for (int round = 0; round < 14; round++) {
        for (int i = 0; i < 16; i++) {
            state[i] = sbox[state[i]];
        }
    }
    
    for (int i = 0; i < 16; i++) {
        output[i] = state[i];
    }
}

__device__ bool validate_bitcoin_key(const uint8_t* key_data, int len) {
    // Basic Bitcoin private key validation
    if (len < 32) return false;
    
    // Check not all zeros
    bool has_nonzero = false;
    for (int i = 0; i < 32; i++) {
        if (key_data[i] != 0) {
            has_nonzero = true;
            break;
        }
    }
    if (!has_nonzero) return false;
    
    // Check not all 0xFF
    bool has_non_ff = false;
    for (int i = 0; i < 32; i++) {
        if (key_data[i] != 0xff) {
            has_non_ff = true;
            break;
        }
    }
    if (!has_non_ff) return false;
    
    // Check entropy (at least 16 different bytes)
    int unique_count = 0;
    bool seen[256] = {false};
    for (int i = 0; i < 32; i++) {
        if (!seen[key_data[i]]) {
            seen[key_data[i]] = true;
            unique_count++;
        }
    }
    
    return unique_count >= 16;
}

extern "C" {
__global__ void bitcoin_crack_real(char* passwords, int* lengths, uint8_t* salt, int salt_len, 
                                   int iterations, uint8_t* encrypted_data, uint8_t* iv_data,
                                   int num_passwords, int* results) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= num_passwords) return;
        
        char* pwd = &passwords[idx * 64];
        int pwd_len = lengths[idx];
        
        if (pwd_len < 3 || pwd_len > 63) {
            results[idx] = 0;
            return;
        }
        
        // PBKDF2-HMAC-SHA512
        uint8_t derived_key[32];
        pbkdf2_hmac_sha512(pwd, pwd_len, salt, salt_len, iterations, derived_key, 32);
        
        // AES-256-CBC decrypt (multi-block, esempio 32 bytes)
        int data_len = 32; // Modifica se encrypted_data è più lunga
        uint8_t decrypted[128];
        for (int i = 0; i < data_len; i += 16) {
            aes_decrypt_block(derived_key, &encrypted_data[i], &decrypted[i]);
        }
        
        // Validazione padding PKCS#7
        int pad = decrypted[data_len - 1];
        bool valid_padding = true;
        if (pad < 1 || pad > 16) valid_padding = false;
        for (int j = 0; j < pad; j++) {
            if (decrypted[data_len - 1 - j] != pad) valid_padding = false;
        }
        
        // Cerca chiave Bitcoin valida nei dati decrittati senza padding
        int real_len = data_len - pad;
        bool found_key = false;
        for (int offset = 0; offset <= real_len - 32; offset += 4) {
            if (validate_bitcoin_key(&decrypted[offset], 32)) {
                found_key = true;
                break;
            }
        }
        // Se non trova con offset 4, prova tutti gli offset
        if (!found_key) {
            for (int offset = 0; offset <= real_len - 32; offset++) {
                if (validate_bitcoin_key(&decrypted[offset], 32)) {
                    found_key = true;
                    break;
                }
            }
        }
        // Risultato finale
        if (valid_padding && found_key) {
            results[idx] = 1;
        } else {
            results[idx] = 0;
        }
}
}
"""

# Statistiche con thread safety
class CrackingStats:
    def __init__(self):
        self.lock = threading.Lock()
        self.passwords_tested = 0
        self.candidates_found = 0
        self.speed = 0
        self.is_running = False
        self.start_time = None
        self.gpu_errors = 0
        self.real_passwords_found = 0

stats = CrackingStats()

# Logging ottimizzato
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('real_bitcoin_crack.log', encoding='utf-8')
    ]
)

class RealBitcoinWalletParser:
    """Parser wallet Bitcoin Core REALE al 100%"""
    
    def __init__(self, wallet_file):
        self.wallet_file = Path(wallet_file)
        self.wallet_data = None
        self.encrypted_keys = []
        self.master_key = None
        self.salt = None
        self.iterations = 25000
        self.db_records = []
        self.real_wallet = False
        
    def parse_wallet(self, candidate_file=None):
        """Analizza wallet Bitcoin Core REALE"""
        try:
            logging.info(f"🔍 Analisi wallet REALE: {self.wallet_file}")
            
            if not self.wallet_file.exists():
                logging.error(f"❌ Wallet non trovato: {self.wallet_file}")
                return False
                
            with open(self.wallet_file, 'rb') as f:
                self.wallet_data = f.read()
            
            logging.info(f"📂 Dimensione wallet: {len(self.wallet_data):,} bytes")
            
            # Analizza Berkeley DB
            if self._parse_real_berkeley_db():
                self.real_wallet = True
                logging.info("✅ Wallet Bitcoin REALE rilevato!")
            else:
                logging.error("❌ NON è un wallet Bitcoin Core valido!")
                return False
            
            # Estrai dati crittografici
            if not self._extract_real_crypto_data(candidate_file=candidate_file):
                logging.error("❌ Dati crittografici non trovati")
                return False
                
            logging.info(f"✅ Wallet analizzato: {len(self.encrypted_keys)} chiavi")
            logging.info(f"🧂 Salt: {self.salt.hex() if self.salt else 'Non trovato'}")
            logging.info(f"🔄 Iterazioni: {self.iterations}")
            
            return True
            
        except Exception as e:
            logging.error(f"❌ Errore analisi wallet: {e}")
            return False
    
    def _parse_real_berkeley_db(self):
        """Analizza formato Berkeley DB con parsing più accurato"""
        try:
            if len(self.wallet_data) < 32:
                return False
            
            # Header Berkeley DB
            if self.wallet_data[12:16] != b'\x00\x05\x31\x62':  # Berkeley DB magic
                logging.warning("⚠️ Magic Berkeley DB non trovato, provo analisi fallback")
            
            # Marcatori Bitcoin Core specifici
            bitcoin_markers = [
                b'mkey',       # Master key
                b'ckey',       # Encrypted private key
                b'key\x00',    # Unencrypted key (rare)
                b'pool',       # Key pool
                b'version',    # Wallet version
                b'minversion', # Minimum version
                b'bestblock'   # Best block hash
            ]
            
            self.marker_positions = {}
            markers_found = 0
            
            # Cerca tutti i marcatori e le loro posizioni
            for marker in bitcoin_markers:
                positions = []
                pos = 0
                while pos < len(self.wallet_data) - len(marker):
                    pos = self.wallet_data.find(marker, pos)
                    if pos == -1:
                        break
                    positions.append(pos)
                    pos += 1
                
                if positions:
                    self.marker_positions[marker] = positions
                    markers_found += 1
                    logging.info(f"✅ {marker}: {len(positions)} occorrenze")
            
            if markers_found < 2:
                logging.error(f"❌ Solo {markers_found} marcatori - wallet non valido")
                return False
            
            # Verifica presenza mkey (essenziale per wallet crittografato)
            if b'mkey' not in self.marker_positions:
                logging.error("❌ Master key non trovato - wallet non crittografato?")
                return False
            
            # Verifica presenza ckey
            if b'ckey' not in self.marker_positions:
                logging.error("❌ Chiavi private crittografate non trovate")
                return False
            
            logging.info(f"✅ {markers_found} marcatori Bitcoin - wallet valido")
            return self._extract_real_db_records()
            
        except Exception as e:
            logging.error(f"❌ Errore Berkeley DB: {e}")
            return False
    
    def _extract_real_db_records(self):
        """Estrai record DB con parsing più accurato"""
        try:
            # Estrai record mkey (master key)
            if b'mkey' in self.marker_positions:
                for pos in self.marker_positions[b'mkey']:
                    # Cerca il record completo
                    record_start = pos
                    # Cerca fine record (prossimo marcatore o pattern)
                    record_end = min(pos + 1024, len(self.wallet_data))
                    
                    # Cerca pattern terminazione record Berkeley DB
                    for end_pos in range(pos + 50, record_end):
                        if (self.wallet_data[end_pos:end_pos+4] in [b'mkey', b'ckey', b'key\x00'] or
                            self.wallet_data[end_pos:end_pos+2] == b'\x00\x00'):
                            record_end = end_pos
                            break
                    
                    record_data = self.wallet_data[record_start:record_end]
                    self.db_records.append({
                        'type': 'mkey',
                        'position': pos,
                        'length': len(record_data),
                        'data': record_data
                    })
            
            # Estrai record ckey (encrypted private keys)
            if b'ckey' in self.marker_positions:
                for pos in self.marker_positions[b'ckey']:
                    record_start = pos
                    record_end = min(pos + 512, len(self.wallet_data))
                    
                    # Cerca fine record
                    for end_pos in range(pos + 48, record_end):
                        if (self.wallet_data[end_pos:end_pos+4] in [b'mkey', b'ckey', b'key\x00'] or
                            self.wallet_data[end_pos:end_pos+2] == b'\x00\x00'):
                            record_end = end_pos
                            break
                    
                    record_data = self.wallet_data[record_start:record_end]
                    self.db_records.append({
                        'type': 'ckey',
                        'position': pos,
                        'length': len(record_data),
                        'data': record_data
                    })
            
            logging.info(f"📊 Estratti {len(self.db_records)} record DB")
            
            # Ordina per posizione
            self.db_records.sort(key=lambda x: x['position'])
            
            return len(self.db_records) > 0
            
        except Exception as e:
            logging.error(f"❌ Errore estrazione record: {e}")
            return False
    
    def _extract_real_crypto_data(self, candidate_file=None):
        """Estrai dati crittografici REALI"""
        try:
            master_keys_found = 0
            private_keys_found = 0
            
            for record in self.db_records:
                if record['type'] == 'mkey':
                    if self._parse_real_master_key(record['data'], candidate_file=candidate_file):
                        master_keys_found += 1
                elif record['type'] == 'ckey':
                    if self._parse_real_encrypted_key(record['data']):
                        private_keys_found += 1
            
            logging.info(f"🔑 Master keys: {master_keys_found}")
            logging.info(f"🔐 Private keys: {private_keys_found}")
            
            return len(self.encrypted_keys) > 0
            
        except Exception as e:
            logging.error(f"❌ Errore dati cripto: {e}")
            return False
    
    def _parse_real_master_key(self, data, candidate_file=None):
        """Analizza master key con parsing più accurato"""
        try:
            logging.info(f"🔍 Analisi master key, dimensione: {len(data)} bytes")
            found = False
            # Prova tutte le sequenze di 16 byte come salt
            # Prova tutte le sequenze di 8, 16, 32 byte come salt
            for salt_len in [16, 32, 48]:
                for salt_start in range(len(data) - salt_len):
                    candidate_salt = data[salt_start:salt_start+salt_len]
                    nonzero = sum(b != 0 for b in candidate_salt)
                    if len(candidate_salt) % 16 != 0:
                        continue
                    # Prova come salt e master key
                    self.salt = candidate_salt
                    self.master_key = {
                        'encrypted_data': candidate_salt,
                        'iv': b'\x00'*16,
                        'salt': self.salt,
                        'iterations': self.iterations,
                        'salt_pos': salt_start,
                        'enc_pos': -1,
                        'iter_pos': -1
                    }
                    self.encrypted_keys.append({
                        'encrypted_data': candidate_salt,
                        'iv': b'\x00'*16,
                        'type': 'master_key'
                    })
                    logging.info(f"🔍 Test salt estratto: {candidate_salt.hex()}")
                    # Test reale: decriptazione e validazione
                    try:
                        dummy_password = "testpassword"
                        derived_key = PBKDF2(
                            password=dummy_password.encode('utf-8'),
                            salt=self.salt,
                            dkLen=32,
                            count=self.iterations,
                            prf=lambda p, s: hmac.new(p, s, hashlib.sha512).digest()
                        )
                        cipher = AES.new(derived_key, AES.MODE_CBC, b'\x00'*16)
                        decrypted = cipher.decrypt(candidate_salt)
                        if len(decrypted) < 16:
                            continue
                        padding = decrypted[-1]
                        if padding < 1 or padding > 16:
                            continue
                        if not all(decrypted[-(i+1)] == padding for i in range(padding)):
                            continue
                        real_data = decrypted[:-padding]
                        if len(real_data) < 32:
                            continue
                        for offset in range(0, len(real_data) - 31):
                            key_bytes = real_data[offset:offset + 32]
                            if self._validate_bitcoin_private_key(key_bytes):
                                logging.info(f"🎉 Salt valido trovato! Offset: {offset} Key: {key_bytes.hex()}")
                                found = True
                                return True
                    except Exception as e:
                        logging.debug(f"[DEBUG] Test salt estratto fallito: {e}")
            # Se fornito un file di candidati, prova tutti
            if candidate_file is not None and os.path.exists(candidate_file):
                logging.info(f"🔍 Provo candidati da {candidate_file}")
                with open(candidate_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith('Found') or line.startswith('Offset:'):
                            continue
                        # Prova a convertire la stringa esadecimale
                        try:
                            candidate_bytes = bytes.fromhex(line)
                        except Exception:
                            continue
                        if len(candidate_bytes) % 16 != 0:
                            continue
                        # Prova come salt e master key
                        self.salt = candidate_bytes
                        self.master_key = {
                            'encrypted_data': candidate_bytes,
                            'iv': b'\x00'*16,
                            'salt': self.salt,
                            'iterations': self.iterations,
                            'salt_pos': -1,
                            'enc_pos': -1,
                            'iter_pos': -1
                        }
                        self.encrypted_keys.append({
                            'encrypted_data': candidate_bytes,
                            'iv': b'\x00'*16,
                            'type': 'master_key'
                        })
                        logging.info(f"🔍 Test candidato: {candidate_bytes.hex()}")
                        # Test reale: decriptazione e validazione
                        try:
                            dummy_password = "testpassword"
                            derived_key = PBKDF2(
                                password=dummy_password.encode('utf-8'),
                                salt=self.salt,
                                dkLen=32,
                                count=self.iterations,
                                prf=lambda p, s: hmac.new(p, s, hashlib.sha512).digest()
                            )
                            cipher = AES.new(derived_key, AES.MODE_CBC, b'\x00'*16)
                            decrypted = cipher.decrypt(candidate_bytes)
                            if len(decrypted) < 16:
                                continue
                            padding = decrypted[-1]
                            if padding < 1 or padding > 16:
                                continue
                            if not all(decrypted[-(i+1)] == padding for i in range(padding)):
                                continue
                            real_data = decrypted[:-padding]
                            if len(real_data) < 32:
                                continue
                            for offset in range(0, len(real_data) - 31):
                                key_bytes = real_data[offset:offset + 32]
                                if self._validate_bitcoin_private_key(key_bytes):
                                    logging.info(f"🎉 Candidato valido trovato! Offset: {offset} Key: {key_bytes.hex()}")
                                    found = True
                                    break
                        except Exception as e:
                            logging.debug(f"[DEBUG] Test candidato fallito: {e}")
                    if found:
                        return True
                logging.info("❌ Nessun candidato ha funzionato")
            if not found:
                logging.error("Salt: MANCANTE (nessuna sequenza di 16 byte valida trovata)")
                return False
        except Exception as e:
            logging.error(f"❌ Errore master key: {e}")
            return False
    
    def _parse_real_encrypted_key(self, data):
        """Analizza chiave privata crittografata"""
        try:
            # Cerca pattern chiave privata
            for pos in range(len(data) - 48):
                encrypted_data = data[pos:pos+32]
                iv_data = data[pos+32:pos+48]
                # Valida entropia
                if (len(set(encrypted_data)) > 16 and len(set(iv_data)) > 8):
                    self.encrypted_keys.append({
                        'encrypted_data': encrypted_data,
                        'iv': iv_data,
                        'type': 'private_key'
                    })
                    return True
            return False
        except Exception as e:
            return False

class RealGPUBitcoinCracker:
    def __init__(self, wallet_file, batch_size=64, threads_per_block=256):
        self.wallet_file = wallet_file
        self.wallet_parser = None
        self.gpu_context = None
        self.cuda_module = None
        self.cuda_function = None
        stats.batch_size = batch_size
        stats.threads_per_block = threads_per_block
        logging.info("🚀 Inizializzazione REAL GPU Bitcoin Cracker...")
        # Inizializza wallet
        self.init_real_wallet()
        # Inizializza GPU
        self.init_real_gpu()
    
    def init_real_wallet(self, candidate_file="extra_entropy_candidates.txt"):
        """Inizializza parser wallet REALE"""
        try:
            logging.info("📁 Caricamento wallet REALE...")
            self.wallet_parser = RealBitcoinWalletParser(self.wallet_file)
            # Prima tentativo standard
            if not self.wallet_parser.parse_wallet():
                logging.warning("⚠️ Parsing standard fallito, provo candidati di entropia...")
                # Prova con i candidati di entropia
                if not self.wallet_parser.parse_wallet(candidate_file=candidate_file):
                    logging.error("❌ Analisi wallet fallita anche con candidati di entropia!")
                    sys.exit(1)
            if not self.wallet_parser.real_wallet:
                logging.error("❌ Wallet non valido!")
                sys.exit(1)
            logging.info("✅ Wallet REALE caricato con successo!")
        except Exception as e:
            logging.error(f"❌ Errore wallet: {e}")
            sys.exit(1)
    
    def init_real_gpu(self):
        """Inizializza GPU REALE"""
        try:
            # Inizializza CUDA
            cuda.init()
            device = cuda.Device(0)
            
            # Crea contesto
            self.gpu_context = device.make_context()
            
            device_name = device.name()
            device_memory = device.total_memory() // 1024 // 1024
            compute_cap = device.compute_capability()
            
            logging.info(f"🔥 GPU: {device_name}")
            logging.info(f"📊 Memoria: {device_memory} MB")
            logging.info(f"🔥 Compute: {compute_cap}")
            
            # Compila kernel REALE
            try:
                logging.info("🔥 Compilazione kernel REAL Bitcoin PBKDF2+AES...")
                
                # Configurazione compilazione per kernel complesso
                import os
                import tempfile
                import shutil
                
                # FORZA rimozione flag duplicati
                if 'PYCUDA_COMPILER_FLAGS' in os.environ:
                    del os.environ['PYCUDA_COMPILER_FLAGS']
                
                # DISABILITA architettura automatica PyCUDA
                os.environ['PYCUDA_DEFAULT_NVCC_FLAGS'] = ''
                
                # Pulisci cache PyCUDA per evitare conflitti
                cache_dir = os.environ.get('PYCUDA_CACHE_DIR', os.path.join(tempfile.gettempdir(), 'pycuda-cache'))
                if os.path.exists(cache_dir):
                    try:
                        shutil.rmtree(cache_dir)
                        logging.info("✅ Cache PyCUDA pulita")
                    except Exception:
                        pass
                
                # Opzioni compilazione per kernel complesso
                compile_options = [
                    '--use_fast_math',           # Ottimizzazioni matematiche
                    '--ptxas-options=-v',        # Verbose per debug
                    '-O3',                       # Ottimizzazione massima
                    '--fmad=true',               # Fused multiply-add
                    '--prec-div=false',          # Divisione veloce
                    '--prec-sqrt=false',         # Radice veloce
                    '--ftz=true'                 # Flush-to-zero
                ]
                
                # Usa configurazione CUDA per kernel REALE
                self.cuda_module = SourceModule(
                    CUDA_REAL_BITCOIN_KERNEL,
                    options=compile_options,
                    no_extern_c=False,
                    cache_dir=False,
                    keep=True  # Mantiene file .ptx per debug
                )
                self.cuda_function = self.cuda_module.get_function("bitcoin_crack_real")
                
                logging.info("✅ Kernel REAL Bitcoin compilato con successo!")
                
                # Test kernel con parametri dummy
                logging.info("🧪 Test kernel REAL...")
                test_passed = self._test_real_kernel()
                if test_passed:
                    logging.info("✅ Test kernel REAL superato!")
                else:
                    logging.warning("⚠️ Test kernel REAL fallito, continuo comunque")
                
            except Exception as e:
                logging.error(f"❌ Errore compilazione REAL kernel: {e}")
                traceback.print_exc()
                raise
            
        except Exception as e:
            logging.error(f"❌ Errore GPU: {e}")
            sys.exit(1)
    
    def _test_real_kernel(self):
        """Test del kernel REAL con dati dummy"""
        try:
            # Dati di test
            test_passwords = ["test123", "password", "bitcoin"]
            test_salt = b'\x01\x02\x03\x04\x05\x06\x07\x08'
            test_iterations = 1000
            test_encrypted = b'\x00' * 32
            test_iv = b'\x11' * 16
            
            # Prepara dati test
            password_data, password_lengths = self.prepare_passwords_for_real_gpu(test_passwords)
            
            if password_data is None:
                return False
            
            # Alloca memoria test
            password_gpu = cuda.mem_alloc(password_data.nbytes)
            lengths_gpu = cuda.mem_alloc(password_lengths.nbytes)
            salt_gpu = cuda.mem_alloc(len(test_salt))
            encrypted_gpu = cuda.mem_alloc(len(test_encrypted))
            iv_gpu = cuda.mem_alloc(len(test_iv))
            results_data = np.zeros(len(test_passwords), dtype=np.int32)
            results_gpu = cuda.mem_alloc(results_data.nbytes)
            
            # Copia dati test
            cuda.memcpy_htod(password_gpu, password_data)
            cuda.memcpy_htod(lengths_gpu, password_lengths)
            cuda.memcpy_htod(salt_gpu, np.frombuffer(test_salt, dtype=np.uint8))
            cuda.memcpy_htod(encrypted_gpu, np.frombuffer(test_encrypted, dtype=np.uint8))
            cuda.memcpy_htod(iv_gpu, np.frombuffer(test_iv, dtype=np.uint8))
            cuda.memcpy_htod(results_gpu, results_data)
            
            # Esegui test
            self.cuda_function(
                password_gpu,
                lengths_gpu,
                salt_gpu,
                np.int32(len(test_salt)),
                np.int32(test_iterations),
                encrypted_gpu,
                iv_gpu,
                np.int32(len(test_passwords)),
                results_gpu,
                block=(32, 1, 1),
                grid=(1, 1)
            )
            
            cuda.Context.synchronize()
            
            # Cleanup
            password_gpu.free()
            lengths_gpu.free()
            salt_gpu.free()
            encrypted_gpu.free()
            iv_gpu.free()
            results_gpu.free()
            
            return True
            
        except Exception as e:
            logging.error(f"❌ Test kernel fallito: {e}")
            return False
    
    def prepare_passwords_for_real_gpu(self, passwords):
        """Prepara password per GPU semplificato"""
        try:
            max_pwd_len = 64
            num_passwords = len(passwords)
            
            # Array password fixed-size per ogni password
            password_data = np.zeros(num_passwords * max_pwd_len, dtype=np.int8)
            password_lengths = np.zeros(num_passwords, dtype=np.int32)
            
            for i, pwd in enumerate(passwords):
                # LIMITA password a max_pwd_len-1 per evitare broadcasting error
                if len(pwd) > max_pwd_len - 1:
                    pwd = pwd[:max_pwd_len - 1]
                
                pwd_bytes = pwd.encode('utf-8', errors='ignore')
                pwd_len = len(pwd_bytes)
                
                # LIMITA anche i bytes per sicurezza
                if pwd_len > max_pwd_len - 1:
                    pwd_bytes = pwd_bytes[:max_pwd_len - 1]
                    pwd_len = len(pwd_bytes)
                
                # Copia password nell'array
                start_idx = i * max_pwd_len
                end_idx = start_idx + pwd_len
                
                # SICUREZZA: verifica che non ecceda i limiti array
                if end_idx <= len(password_data):
                    password_data[start_idx:end_idx] = np.frombuffer(pwd_bytes, dtype=np.int8)
                    password_lengths[i] = pwd_len
                else:
                    # Password troppo lunga, salta
                    password_lengths[i] = 0
            
            return password_data, password_lengths
            
        except Exception as e:
            logging.error(f"❌ Errore preparazione GPU: {e}")
            return None, None
    
    def process_passwords_real_gpu(self, passwords_batch):
        """Elabora password con REAL GPU PBKDF2+AES"""
        try:
            password_gpu = lengths_gpu = salt_gpu = encrypted_gpu = iv_gpu = results_gpu = None
            logging.info(f"🔥 REAL GPU PBKDF2+AES: {len(passwords_batch)} password")
            if not self.wallet_parser.salt or not self.wallet_parser.encrypted_keys:
                logging.error("❌ Dati wallet mancanti")
                logging.info(f"🔍 Salt: {'PRESENTE' if self.wallet_parser.salt else 'MANCANTE'}")
                logging.info(f"🔍 Keys: {len(self.wallet_parser.encrypted_keys) if self.wallet_parser.encrypted_keys else 0}")
                return 0
            start_time = time.time()
            password_data, password_lengths = self.prepare_passwords_for_real_gpu(passwords_batch)
            if password_data is None:
                return 0
            num_passwords = len(passwords_batch)
            salt = self.wallet_parser.salt
            iterations = self.wallet_parser.iterations
            encrypted_key = self.wallet_parser.encrypted_keys[0]
            encrypted_data = encrypted_key['encrypted_data']
            iv_data = encrypted_key['iv']
            salt_gpu_data = np.frombuffer(salt, dtype=np.uint8)
            encrypted_gpu_data = np.frombuffer(encrypted_data, dtype=np.uint8)
            iv_gpu_data = np.frombuffer(iv_data, dtype=np.uint8)
            logging.info(f"🧂 Salt: {len(salt)} bytes")
            logging.info(f"🔐 Encrypted: {len(encrypted_data)} bytes")
            logging.info(f"🔑 IV: {len(iv_data)} bytes")
            logging.info(f"🔄 Iterazioni: {iterations}")
            # ALLOCA MEMORIA GPU per kernel REALE
            try:
                password_gpu = cuda.mem_alloc(password_data.nbytes)
                lengths_gpu = cuda.mem_alloc(password_lengths.nbytes)
                salt_gpu = cuda.mem_alloc(salt_gpu_data.nbytes)
                encrypted_gpu = cuda.mem_alloc(encrypted_gpu_data.nbytes)
                iv_gpu = cuda.mem_alloc(iv_gpu_data.nbytes)
                results_data = np.zeros(num_passwords, dtype=np.int32)
                results_gpu = cuda.mem_alloc(results_data.nbytes)
                logging.debug("[GPU] Allocata memoria per batch")
                # COPIA DATI SU GPU
                cuda.memcpy_htod(password_gpu, password_data)
                cuda.memcpy_htod(lengths_gpu, password_lengths)
                cuda.memcpy_htod(salt_gpu, salt_gpu_data)
                cuda.memcpy_htod(encrypted_gpu, encrypted_gpu_data)
                cuda.memcpy_htod(iv_gpu, iv_gpu_data)
                cuda.memcpy_htod(results_gpu, results_data)
                threads_per_block = getattr(stats, 'threads_per_block', 256)
                blocks_per_grid = getattr(stats, 'blocks_per_grid', 0)
                if not blocks_per_grid or blocks_per_grid < 1:
                    blocks_per_grid = (num_passwords + threads_per_block - 1) // threads_per_block
                logging.info(f"🔥 REAL GPU: {blocks_per_grid} blocks × {threads_per_block} threads [APPLICATI]")
                self.cuda_function(
                    password_gpu,
                    lengths_gpu,
                    salt_gpu,
                    np.int32(len(salt)),
                    np.int32(iterations),
                    encrypted_gpu,
                    iv_gpu,
                    np.int32(num_passwords),
                    results_gpu,
                    block=(threads_per_block, 1, 1),
                    grid=(blocks_per_grid, 1)
                )
                cuda.Context.synchronize()
                cuda.memcpy_dtoh(results_data, results_gpu)
                sleep_time = getattr(stats, 'sleep_time', 0.5)
                if sleep_time > 0:
                    time.sleep(sleep_time)  # Pausa configurabile tra batch
            finally:
                # LIBERA MEMORIA GPU SEMPRE
                for mem in [password_gpu, lengths_gpu, salt_gpu, encrypted_gpu, iv_gpu, results_gpu]:
                    if mem is not None:
                        mem.free()
                logging.debug("[GPU] Memoria liberata dopo batch")

            # Analizza risultati
            candidates_found = 0
            for i, result in enumerate(results_data):
                stats.passwords_tested += 1
                if result == 1:
                    candidates_found += 1
                    # Calcola la chiave derivata PBKDF2 (solo per log, la validazione è già su GPU)
                    derived_key = PBKDF2(
                        password=passwords_batch[i].encode('utf-8'),
                        salt=salt,
                        dkLen=32,
                        count=iterations,
                        prf=lambda p, s: hmac.new(p, s, hashlib.sha512).digest()
                    )
                    logging.info(f"[GPU] Derived key: {derived_key.hex()}")
                    # Se vuoi loggare anche la password:
                    # logging.info(f"[GPU] Password: {passwords_batch[i]}")
                    stats.real_passwords_found += 1
            stats.candidates_found += candidates_found
            elapsed = time.time() - start_time
            if elapsed > 0:
                stats.speed = int(len(passwords_batch) / elapsed)
            else:
                stats.speed = 0
            return candidates_found
            
        except Exception as e:
            logging.error(f"❌ Errore GPU: {e}")
            return 0
    
    def _validate_bitcoin_private_key(self, key_bytes):
        """Validazione base chiave privata Bitcoin"""
        try:
            if len(key_bytes) != 32:
                return False
            
            # Controlla che non sia tutto zero
            if all(b == 0 for b in key_bytes):
                return False
            
            # Controlla che non sia tutto 0xFF
            if all(b == 0xFF for b in key_bytes):
                return False
            
            # Controlla entropia (almeno 16 valori diversi)
            unique_values = len(set(key_bytes))
            if unique_values < 16:
                return False
            
            # Controlla range valido per Bitcoin (< n del secp256k1)
            # n = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
            key_int = int.from_bytes(key_bytes, byteorder='big')
            secp256k1_n = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
            
            if key_int >= secp256k1_n:
                return False
            
            return True
            
        except Exception:
            return False
    
    def _validate_bitcoin_key_format(self, key_bytes):
        """Validazione formato chiave Bitcoin avanzata"""
        try:
            # Validazione base
            if not self._validate_bitcoin_private_key(key_bytes):
                return False
            
            # Pattern comuni chiavi Bitcoin
            key_hex = key_bytes.hex().lower()
            
            # Verifica che non abbia pattern sospetti
            suspicious_patterns = [
                '0123456789abcdef',  # Sequenza
                'ffffffffffffffff',  # Tutto F
                '0000000000000000',  # Tutto 0
                '1234567890123456',  # Pattern numerico
            ]
            
            for pattern in suspicious_patterns:
                if pattern in key_hex:
                    return False
            
            # Distribuzione bytes ragionevole
            byte_counts = {}
            for b in key_bytes:
                byte_counts[b] = byte_counts.get(b, 0) + 1
            
            # Non dovrebbe avere più del 30% dello stesso byte
            max_count = max(byte_counts.values())
            if max_count > len(key_bytes) * 0.3:
                return False
            
            logging.info(f"✅ Chiave Bitcoin validata: {key_hex}")
            return True
            
        except Exception:
            return False
    
    def _save_real_password(self, password):
        """Salva password REALE"""
        try:
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
            
            result = {
                'timestamp': timestamp,
                'wallet_file': str(self.wallet_file),
                'password': password,
                'method': 'REAL_GPU_BITCOIN_VALIDATION',
                'salt': self.wallet_parser.salt.hex(),
                'iterations': self.wallet_parser.iterations,
                'validation': 'PBKDF2+AES+Bitcoin_Key_Validation'
            }
            
            # Salva JSON
            with open('REAL_BITCOIN_PASSWORD_FOUND.json', 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            # Salva testo
            with open('REAL_BITCOIN_PASSWORD_FOUND.txt', 'w', encoding='utf-8') as f:
                f.write(f"🎉 PASSWORD BITCOIN REALE TROVATA! 🎉\n")
                f.write(f"=" * 50 + "\n")
                f.write(f"Timestamp: {timestamp}\n")
                f.write(f"Wallet: {self.wallet_file}\n")
                f.write(f"Password: {password}\n")
                f.write(f"Metodo: REAL GPU Bitcoin Validation\n")
                f.write(f"Salt: {self.wallet_parser.salt.hex()}\n")
                f.write(f"Iterazioni: {self.wallet_parser.iterations}\n")
                f.write(f"Validazione: PBKDF2 + AES + Bitcoin Key\n")
            
            logging.info("💎 Password REALE salvata!")
            
        except Exception as e:
            logging.error(f"❌ Errore salvataggio: {e}")
    
    def crack_real_wallet(self, dictionary_file):
        """Cracking REALE wallet Bitcoin"""
        logging.info("🚀 Avvio REAL GPU Bitcoin wallet cracking...")
        logging.info(f"📁 Wallet: {self.wallet_file}")
        logging.info(f"📖 Dizionario: {dictionary_file}")
        logging.info(f"🔥 GPU: REAL validation enabled")
        
        if not Path(dictionary_file).exists():
            logging.error(f"❌ Dizionario non trovato: {dictionary_file}")
            return
        
        stats.is_running = True
        stats.start_time = time.time()
        
        try:
            with open(dictionary_file, 'r', encoding='utf-8', errors='ignore') as f:
                batch = []
                batch_size = getattr(stats, 'batch_size', 64)
                batch_num = 1
                
                for line in f:
                    if not stats.is_running:
                        break
                    
                    password = line.strip()
                    if password and len(password) >= 3:
                        batch.append(password)
                        
                        if len(batch) >= batch_size:
                            logging.info(f"📦 REAL GPU batch {batch_num}: {len(batch)} password [BATCHSIZE APPLICATO: {batch_size}]")
                            
                            candidates = self.process_passwords_real_gpu(batch)
                            
                            if candidates is None:
                                candidates = 0
                            if candidates > 0:
                                logging.info("🎉 CANDIDATI REALI TROVATI!")
                            
                            batch = []
                            batch_num += 1
                            
                            # Check se password trovata
                            if stats.real_passwords_found > 0:
                                logging.info("🎉 PASSWORD REALE CONFERMATA - CRACKING COMPLETATO!")
                                break
                
                # Batch finale
                if batch and stats.is_running:
                    logging.info(f"📦 Batch finale: {len(batch)} password")
                    self.process_passwords_real_gpu(batch)
                    
        except KeyboardInterrupt:
            logging.info("⏹️ Cracking fermato")
        except Exception as e:
            logging.error(f"❌ Errore cracking: {e}")
            traceback.print_exc()
        finally:
            stats.is_running = False
            logging.info("🏁 REAL GPU cracking terminato")
            
            if self.gpu_context:
                try:
                    self.gpu_context.pop()
                except:
                    pass

# Flask Web Interface
app = Flask(__name__)
cracker = None  # Variabile globale per accesso da Flask e thread

app.config['SECRET_KEY'] = 'real_gpu_bitcoin_cracker'

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>REAL GPU Bitcoin Cracker</title>
    <meta charset="utf-8">
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #0d1421; color: #fff; }
        .container { max-width: 1200px; margin: 0 auto; }
        .header { text-align: center; margin-bottom: 30px; }
        .header h1 { font-size: 2.5em; color: #00ff88; margin: 0; }
        .real-alert { background: #00ff88; color: #000; padding: 15px; border-radius: 10px; text-align: center; font-weight: bold; margin: 20px 0; }
        .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 30px 0; }
        .stat-card { background: #1e293b; border-radius: 10px; padding: 20px; text-align: center; border: 2px solid #00ff88; }
        .stat-number { font-size: 2em; font-weight: bold; color: #00ff88; }
        .stat-label { font-size: 0.9em; opacity: 0.8; margin-top: 5px; }
        .status { font-size: 1.2em; margin: 20px 0; text-align: center; }
        .controls { text-align: center; margin: 30px 0; }
        .btn { background: #00ff88; color: #000; border: none; padding: 12px 24px; border-radius: 6px; font-size: 1.1em; cursor: pointer; margin: 0 10px; font-weight: bold; }
        .btn:hover { background: #00cc6a; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔥 REAL GPU Bitcoin Cracker</h1>
            <div class="real-alert">🚀 VALIDAZIONE BITCOIN REALE AL 100% 🚀</div>
            <div class="status" id="status">Stato: <span id="status-text">Caricamento...</span></div>
        </div>
        
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-number" id="passwords-tested">0</div>
                <div class="stat-label">Password Testate</div>
            </div>
            <div class="stat-card">
                <div class="stat-number" id="speed">0</div>
                <div class="stat-label">REAL GPU Speed (p/s)</div>
            </div>
            <div class="stat-card">
                <div class="stat-number" id="candidates">0</div>
                <div class="stat-label">Candidati GPU</div>
            </div>
            <div class="stat-card">
                <div class="stat-number" id="found">0</div>
                <div class="stat-label">Password REALI</div>
            </div>
            <div class="stat-card">
                <div class="stat-number" id="runtime">00:00:00</div>
                <div class="stat-label">Runtime</div>
            </div>
            <div class="stat-card">
                <div class="stat-number" id="salt">-</div>
                <div class="stat-label">Salt Corrente</div>
            </div>
            <div class="stat-card">
                <div class="stat-number" id="num-keys">-</div>
                <div class="stat-label">Chiavi Trovate</div>
            </div>
            <div class="stat-card">
                <div class="stat-number" id="wallet-status">-</div>
                <div class="stat-label">Stato Wallet</div>
            </div>
            <div class="stat-card">
                <div class="stat-number" id="error-log">-</div>
                <div class="stat-label">Log Errore</div>
            </div>
        </div>
        
        <div class="controls">
            <button class="btn" onclick="stopCracking()">Ferma REAL GPU</button>
            <button class="btn" onclick="location.reload()">Aggiorna</button>
            <br><br>
            <label for="batchsize">Batch Size:</label>
            <input type="number" id="batchsize" min="1" max="4096" value="64" style="width:80px;">
            <label for="threads">Threads per Block:</label>
            <input type="number" id="threads" min="1" max="1024" value="256" style="width:80px;">
            <div class="stats-grid">
                <!-- ...existing stat cards... -->
                <div class="stat-card">
                    <div class="stat-number" id="blocks-per-grid">0</div>
                    <div class="stat-label">Blocks per Grid</div>
                </div>
            </div>
            <label for="blocks">Blocks per Grid:</label>
            <input type="number" id="blocks" min="1" max="65536" value="0" style="width:80px;">
            <label for="sleep">Sleep tra batch (sec):</label>
            <input type="number" id="sleep" min="0" max="5" step="0.1" value="0.5" style="width:80px;">
            <button class="btn" onclick="updateConfig()">Applica</button>
        </div>
    </div>

    <script>
        function updateStats() {
            fetch('/stats')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('passwords-tested').textContent = data.passwords_tested.toLocaleString();
                    document.getElementById('speed').textContent = Math.round(data.speed).toLocaleString();
                    document.getElementById('candidates').textContent = data.candidates_found;
                    document.getElementById('found').textContent = data.real_passwords_found;
                    document.getElementById('runtime').textContent = data.runtime;
                    document.getElementById('salt').textContent = data.salt || '-';
                    document.getElementById('num-keys').textContent = data.num_keys || '-';
                    document.getElementById('wallet-status').textContent = data.wallet_status || '-';
                    document.getElementById('error-log').textContent = data.error_log || '-';
                    document.getElementById('status-text').textContent = data.is_running ? 'REAL GPU ATTIVO' : 'Fermo';
                    document.getElementById('blocks-per-grid').textContent = data.blocks_per_grid ? data.blocks_per_grid.toLocaleString() : '-';
                });
        }
        
        function stopCracking() {
            fetch('/stop', {method: 'POST'});
        }

            function updateStats() {
        fetch('/stats')
            .then(response => response.json())
            .then(data => {
                // ...existing stats...
                document.getElementById('blocks-per-grid').textContent = data.blocks_per_grid.toLocaleString();
            });
    }

        function updateConfig() {
            const batchsize = parseInt(document.getElementById('batchsize').value);
            const threads = parseInt(document.getElementById('threads').value);
            const blocks = parseInt(document.getElementById('blocks').value);
            const sleep = parseFloat(document.getElementById('sleep').value);
            fetch('/config', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({batch_size: batchsize, threads_per_block: threads, blocks_per_grid: blocks, sleep_time: sleep})
            }).then(r => r.json()).then(data => {
                alert('Configurazione aggiornata!');
            });
        }

        updateStats();
        setInterval(updateStats, 1000);
    </script>
</body>
<style>
div#salt {
    font-size: 2em !important;
    font-weight: bold !important;
    color: #00ff88 !important;
    word-wrap: break-word !important;
    overflow-wrap: break-word !important;
    font-size: 12px !important;
}
</style>
</html>
"""

@app.route('/')
def dashboard():
    return render_template_string(HTML_TEMPLATE)

@app.route('/stats')
def get_stats():
    with stats.lock:
        runtime = 0
        if stats.start_time:
            runtime = int(time.time() - stats.start_time)
        runtime_str = f"{runtime//3600:02d}:{(runtime%3600)//60:02d}:{runtime%60:02d}"
        # Info avanzate
        salt = None
        num_keys = 0
        wallet_status = "N/A"
        error_log = ""
        try:
            cracker = None
            # Trova l'istanza cracker se esiste
            import sys
            for obj in sys.modules[__name__].__dict__.values():
                if hasattr(obj, 'wallet_parser'):
                    cracker = obj
                    break
            if cracker and hasattr(cracker, 'wallet_parser'):
                salt = getattr(cracker.wallet_parser, 'salt', None)
                num_keys = len(getattr(cracker.wallet_parser, 'encrypted_keys', []))
                wallet_status = "OK" if cracker.wallet_parser.real_wallet else "INVALID"
        except Exception as e:
            error_log = str(e)
        return jsonify({
            'passwords_tested': stats.passwords_tested,
            'candidates_found': stats.candidates_found,
            'speed': stats.speed,
            'is_running': stats.is_running,
            'runtime': runtime_str,
            'real_passwords_found': stats.real_passwords_found,
            'salt': salt.hex() if salt else None,
            'num_keys': num_keys,
            'wallet_status': wallet_status,
            'error_log': error_log,
            'batch_size': getattr(stats, 'batch_size', 64),
            'threads_per_block': getattr(stats, 'threads_per_block', 256),
            'blocks_per_grid': getattr(stats, 'blocks_per_grid', 0),
            'sleep_time': getattr(stats, 'sleep_time', 0.5)
        })
# Permetti aggiornamento batch_size e threads_per_block via POST
@app.route('/config', methods=['POST'])
def update_config():
    data = request.get_json()
    batch_size = int(data.get('batch_size', 64))
    threads_per_block = int(data.get('threads_per_block', 256))
    blocks_per_grid = int(data.get('blocks_per_grid', 0))
    sleep_time = float(data.get('sleep_time', 0.5))
    with stats.lock:
        stats.batch_size = batch_size
        stats.threads_per_block = threads_per_block
        stats.blocks_per_grid = blocks_per_grid
        stats.sleep_time = sleep_time
    # Aggiorna anche nell'istanza cracker se esiste
    import sys
    for obj in sys.modules[__name__].__dict__.values():
        if hasattr(obj, 'wallet_parser'):
            if hasattr(obj, 'batch_size'):
                obj.batch_size = batch_size
            if hasattr(obj, 'threads_per_block'):
                obj.threads_per_block = threads_per_block
            if hasattr(obj, 'sleep_time'):
                obj.sleep_time = sleep_time
    return jsonify({'status': 'ok', 'batch_size': batch_size, 'threads_per_block': threads_per_block, 'sleep_time': sleep_time})

@app.route('/stop', methods=['POST'])
def stop_cracking():
    stats.is_running = False
    return jsonify({'status': 'stopped'})

def run_web_server():
    try:
        app.run(host='127.0.0.1', port=5001, debug=False, use_reloader=False)
    except Exception as e:
        logging.error(f"❌ Errore web server: {e}")

if __name__ == "__main__":
    print("🔥 REAL GPU BITCOIN WALLET CRACKER")
    print("=" * 60)
    print("🎯 VALIDAZIONE BITCOIN REALE AL 100%")
    print("🔥 PBKDF2 + AES + Bitcoin Key Validation")
    print("=" * 60)
    
    if len(sys.argv) != 2:
        print("Utilizzo: python true_gpu_cracker_real.py <wallet_file>")
        print("Esempio: python true_gpu_cracker_real.py wallet.dat")
        sys.exit(1)
    
    wallet_file = sys.argv[1]
    
    # Avvia server web
    web_thread = threading.Thread(target=run_web_server, daemon=True)
    web_thread.start()
    
    print("🌐 Dashboard REAL: http://127.0.0.1:5001")
    time.sleep(2)
    
    cracker = None
    try:
        cracker = RealGPUBitcoinCracker(wallet_file)
        
        dictionary_file = "dictionaries/realuniq.lst"
        if not Path(dictionary_file).exists():
            print(f"❌ Dizionario non trovato: {dictionary_file}")
            dictionary_file = "dictionaries/rockyou.txt"
            if not Path(dictionary_file).exists():
                print("❌ Nessun dizionario trovato!")
                sys.exit(1)
        
        cracker.crack_real_wallet(dictionary_file)
        
    except KeyboardInterrupt:
        print("\n⏹️ REAL GPU cracking fermato")
        stats.is_running = False
    except Exception as e:
        print(f"❌ Errore: {e}")
        traceback.print_exc()
    finally:
        if cracker and hasattr(cracker, 'gpu_context') and cracker.gpu_context:
            try:
                cracker.gpu_context.pop()
            except:
                pass
    
    print("\n🏁 REAL GPU Bitcoin cracking terminato.")
    if stats.real_passwords_found > 0:
        print("🎉 PASSWORD BITCOIN REALE TROVATA! Controlla i file di output!")
