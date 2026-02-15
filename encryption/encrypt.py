import sys
from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_OAEP
from Crypto.Random import get_random_bytes
from Crypto.Cipher import AES

# usage:
# python encrypt.py input.csv public_key.pem output.enc

input_file = sys.argv[1]
public_key_file = sys.argv[2]
output_file = sys.argv[3]

# read public key
with open(public_key_file, "rb") as f:
    public_key = RSA.import_key(f.read())

# generate random AES key
session_key = get_random_bytes(16)

# encrypt AES key with RSA
cipher_rsa = PKCS1_OAEP.new(public_key)
enc_session_key = cipher_rsa.encrypt(session_key)

# encrypt data with AES
cipher_aes = AES.new(session_key, AES.MODE_EAX)
with open(input_file, "rb") as f:
    data = f.read()

ciphertext, tag = cipher_aes.encrypt_and_digest(data)

# save encrypted file
with open(output_file, "wb") as f:
    f.write(enc_session_key)
    f.write(cipher_aes.nonce)
    f.write(tag)
    f.write(ciphertext)

print("Encryption successful ->", output_file)
