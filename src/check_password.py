"""
Utility script to verify bcrypt password hashes.
This can help diagnose issues with password verification.
"""
import os
import sys
from passlib.context import CryptContext
import bcrypt
import getpass

# Setup password context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def verify_password(plain_password, hashed_password):
    """Verify a password against a hash"""
    return pwd_context.verify(plain_password, hashed_password)

def hash_password(password):
    """Hash a password for storage"""
    return pwd_context.hash(password)

def main():
    print("=== Bcrypt Password Verification Tool ===")
    print(f"Bcrypt version: {bcrypt.__version__}")
    
    # Get password hash
    password_hash = input("Enter the bcrypt hash from database: ")
    
    # Get password to verify
    password = getpass.getpass("Enter password to verify: ")
    
    # Verify the password
    try:
        result = verify_password(password, password_hash)
        if result:
            print("\n✅ Password verification SUCCESSFUL")
        else:
            print("\n❌ Password verification FAILED")
        
        print(f"\nPassword: {password}")
        print(f"Hash: {password_hash}")
        
        # Generate a new hash for the same password to compare
        new_hash = hash_password(password)
        print(f"\nNew hash for same password: {new_hash}")
    except Exception as e:
        print(f"\n❌ Error during verification: {str(e)}")

if __name__ == "__main__":
    main() 