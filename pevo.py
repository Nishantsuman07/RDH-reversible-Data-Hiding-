import cv2
import numpy as np
import math

def rhombus_predictor(img):
    """Paper-compliant rhombus predictor (Eq.1)"""
    h, w = img.shape
    pred = np.zeros_like(img, dtype=np.float32)
    for i in range(1, h-1):
        for j in range(1, w-1):
            pred[i,j] = (img[i-1,j] + img[i+1,j] + img[i,j-1] + img[i,j+1]) / 4.0
    return pred

def calculate_local_complexity(block):
    """Calculate noise level (Eq.16)"""
    d1 = np.abs(block[1,0] - block[0,1])
    d2 = np.abs(block[0,1] - block[1,2])
    d3 = np.abs(block[1,2] - block[2,1])
    d4 = np.abs(block[2,1] - block[1,0])
    avg_d = (d1 + d2 + d3 + d4) / 4.0
    return np.mean([(avg_d-d1)**2, (avg_d-d2)**2, (avg_d-d3)**2, (avg_d-d4)**2])

def pevo_embed(cover_path, secret_data, thresholds=[15,30]):
    # 1. Preprocess image
    img = cv2.imread(cover_path, cv2.IMREAD_GRAYSCALE)
    if img is None: raise FileNotFoundError("Image not found")
    
    # Resize and prevent overflow
    img = cv2.resize(img, (512,512))
    img = np.where(img == 255, 254, img)
    img = np.where(img == 0, 1, img)
    original = img.copy()
    
    # 2. Calculate prediction errors
    pred = rhombus_predictor(img)
    errors = img.astype(np.float32) - pred
    
    # 3. Double-layer embedding
    marked = img.copy()
    data_idx = 0
    block_size = 3
    
    for layer in [0, 1]:  # Blank and shadow layers
        for i in range(layer, 512, block_size):
            for j in range(layer, 512, block_size):
                if i+block_size > 512 or j+block_size > 512: continue
                
                # Extract block and sort errors
                block = img[i:i+block_size, j:j+block_size]
                block_errors = errors[i:i+block_size, j:j+block_size].flatten()
                sorted_idx = np.argsort(block_errors)
                sorted_errors = np.sort(block_errors)
                
                # Calculate noise level and determine k
                nl = calculate_local_complexity(block)
                k = 1
                for t in reversed(thresholds):
                    if nl <= t: k += 1
                k = min(k, len(sorted_errors)//2)
                
                # Multiple embedding for max values
                for m in range(k):
                    if data_idx >= len(secret_data): break
                    
                    idx = -1 - m
                    diff = sorted_errors[idx] - sorted_errors[idx-1]
                    new_diff = diff * 2 + int(secret_data[data_idx])
                    sorted_errors[idx] = sorted_errors[idx-1] + new_diff
                    data_idx += 1
                
                # Reconstruct block
                reconstructed = np.zeros_like(sorted_errors)
                for idx, val in zip(sorted_idx, sorted_errors):
                    reconstructed[idx] = val
                marked_block = (pred[i:i+block_size, j:j+block_size] + 
                               reconstructed.reshape(block.shape))
                
                # Prevent overflow
                marked_block = np.clip(marked_block, 1, 254).astype(np.uint8)
                marked[i:i+block_size, j:j+block_size] = marked_block

    # Calculate PSNR
    mse = np.mean((original.astype(float) - marked.astype(float)) ** 2)
    psnr = 10 * np.log10(255**2/mse) if mse !=0 else float('inf')
    
    return marked, psnr

def pevo_extract(marked_path, data_length, thresholds=[15,30]):
    # 1. Load marked image
    marked = cv2.imread(marked_path, cv2.IMREAD_GRAYSCALE)
    if marked is None: raise FileNotFoundError("Marked image not found")
    
    # 2. Calculate prediction errors
    pred = rhombus_predictor(marked)
    errors = marked.astype(np.float32) - pred
    
    # 3. Extract data
    extracted_bits = []
    block_size = 3
    
    for layer in reversed([0,1]):
        for i in reversed(range(layer, 512, block_size)):
            for j in reversed(range(layer, 512, block_size)):
                if i+block_size > 512 or j+block_size > 512: continue
                
                # Process block
                block = marked[i:i+block_size, j:j+block_size]
                block_errors = errors[i:i+block_size, j:j+block_size].flatten()
                sorted_errors = np.sort(block_errors)
                
                # Calculate noise level
                nl = calculate_local_complexity(block)
                k = 1
                for t in reversed(thresholds):
                    if nl <= t: k += 1
                k = min(k, len(sorted_errors)//2)
                
                # Extract bits
                for m in reversed(range(k)):
                    if len(extracted_bits) >= data_length: break
                    idx = -1 - m
                    diff = sorted_errors[idx] - sorted_errors[idx-1]
                    extracted_bits.append(str(int(diff % 2)))
    
    return ''.join(reversed(extracted_bits))[:data_length]

# Example usage
if __name__ == "__main__":
    # Embed
    secret_bits = '1'*10000  # 10,000 bits as in paper
    marked_img, psnr = pevo_embed("Images/Pepper.bmp", secret_bits)
    cv2.imwrite("Marked_Pepper.bmp", marked_img)
    
    # Extract
    extracted = pevo_extract("Marked_Pepper.bmp", len(secret_bits))
    accuracy = sum(1 for a,b in zip(secret_bits, extracted) if a==b)/len(secret_bits)
    
    print(f"PSNR: {psnr:.2f} dB")
    print(f"Extraction Accuracy: {accuracy:.2%}")
