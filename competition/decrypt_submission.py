import sys
from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_OAEP, AES

# usage:
# python competition/decrypt_submission.py submissions/test.enc encryption/private_key.pem decrypted.csv

enc_path = sys.argv[1]
private_key_path = sys.argv[2]
out_csv = sys.argv[3]

# Load private key
with open(private_key_path, "rb") as f:
    private_key = RSA.import_key(f.read())

rsa_len = private_key.size_in_bytes()  # 256 bytes for 2048-bit RSA

# Read encrypted file parts
with open(enc_path, "rb") as f:
    enc_session_key = f.read(rsa_len)
    nonce = f.read(16)
    tag = f.read(16)
    ciphertext = f.read()

# RSA decrypt AES key
cipher_rsa = PKCS1_OAEP.new(private_key)
session_key = cipher_rsa.decrypt(enc_session_key)

# AES decrypt
cipher_aes = AES.new(session_key, AES.MODE_EAX, nonce=nonce)
data = cipher_aes.decrypt_and_verify(ciphertext, tag)

# Write decrypted CSV
with open(out_csv, "wb") as f:
    f.write(data)

print("Decryption successful ->", out_csv)

