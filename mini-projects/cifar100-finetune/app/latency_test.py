import requests
import time
import numpy as np
import os

SERVER_URL = "http://127.0.0.1:8000"  
SINGLE_ENDPOINT = f"{SERVER_URL}/predict/single"
BATCH_ENDPOINT = f"{SERVER_URL}/predict/batch"

test_images_dir= os.path.join(os.getcwd(), "../data/test/") 
TEST_IMAGES = [
    test_images_dir + "test6.jpeg",  # Image 1 (single and batch)
    test_images_dir + "test5.jpg"  # Image 2 (for batch)
]  

NUM_REQUESTS = 50  # Number of requests per test
BATCH_SIZES = [1, 2, 4]  # Test these
TIMEOUT = 60  # Seconds per request

def test_latency(batch_size: int):
    """
    Test latency for a given batch size.
    Files are opened before post and closed after to avoid "read of closed file".
    """
    print(f"\n=== Testing Batch Size {batch_size} ({NUM_REQUESTS} requests) ===")
    times = []

    for i in range(NUM_REQUESTS):
        open_files = []  # List to track and clean up files
        start = time.perf_counter()

        try:
            if batch_size == 1:
                # Single endpoint
                image_path = TEST_IMAGES[0]
                if not os.path.exists(image_path):
                    raise FileNotFoundError(f"Missing image: {image_path}")
                filename = os.path.basename(image_path)
                f = open(image_path, 'rb')  # Open without with (to pass to requests)
                open_files.append(f)
                
                # Explicit content-type
                files = {'image': (filename, f, 'image/jpeg')}
                response = requests.post(SINGLE_ENDPOINT, files=files, timeout=TIMEOUT)
            else:
                # Batch endpoint – Open all B files first (cycle if < batch_size)
                files_list = []
                for j in range(batch_size):
                    img_path = TEST_IMAGES[j % len(TEST_IMAGES)]  # Cycle images
                    if not os.path.exists(img_path):
                        raise FileNotFoundError(f"Missing image: {img_path}")
                    filename = os.path.basename(img_path)
                    f = open(img_path, 'rb')  # Open without with
                    open_files.append(f)
                    files_list.append(('images', (filename, f, 'image/jpeg')))
                
                response = requests.post(BATCH_ENDPOINT, files=files_list, timeout=TIMEOUT)

            status_code = response.status_code
            if status_code != 200:
                error_msg = response.text[:200]
                print(f"Request {i+1}: Error {status_code} - {error_msg}")
                if 'content-type' in error_msg.lower():
                    print("Tip: Ensure images are JPEG/PNG. Try renaming to .jpg.")
                # Optional: Print full response for debug
                # print("Full error:", response.json() if response.headers.get('content-type') == 'application/json' else response.text)
                continue  # Skip timing

            end = time.perf_counter()
            latency_ms = (end - start) * 1000  # End-to-end ms
            times.append(latency_ms)
            
            # Print partial response for success
            try:
                resp_data = response.json()
                pred_class = resp_data.get('predicted_class', 'N/A') if batch_size == 1 else f"[{len(resp_data.get('predictions', []))} classes]"
                print(f"Request {i+1}: OK ({pred_class}), Latency {latency_ms:.2f} ms")
            except:
                print(f"Request {i+1}: OK, Latency {latency_ms:.2f} ms")

        except FileNotFoundError as e:
            print(f"Request {i+1}: File error - {str(e)}")
            print("Tip: Check TEST_IMAGES absolute paths exist.")
        except Exception as e:
            print(f"Request {i+1}: Exception - {str(e)}")
        finally:
            # Always close open files
            for f in open_files:
                try:
                    f.close()
                except:
                    pass

    if times:
        p50 = np.percentile(times, 50)
        mean = np.mean(times)
        min_lat = min(times)
        max_lat = max(times)
        per_image = p50 / batch_size if batch_size > 1 else p50

        print(f"\nBatch Size {batch_size} Results:")
        print(f"  p50 (Median) Latency: {p50:.2f} ms")
        print(f"  Per-Image p50: {per_image:.2f} ms")
        print(f"  Mean Latency: {mean:.2f} ms")
        print(f"  Min Latency: {min_lat:.2f} ms")
        print(f"  Max Latency: {max_lat:.2f} ms")
        print(f"  Success Rate: {len(times)/NUM_REQUESTS * 100:.1f}% ({len(times)}/{NUM_REQUESTS})")
        return p50
    else:
        print(f"Batch Size {batch_size}: No successful requests! Check images/API/server.")
        return None

if __name__ == "__main__":
    print("Verifying test images...")
    for path in TEST_IMAGES:
        if os.path.exists(path):
            size = os.path.getsize(path)
            print(f"OK: {path} (size: {size / 1024:.1f} KB)")
        else:
            print(f"ERROR: Missing {path} Fix with absolute paths!")
            exit(1)  # Stop if any missing
    
    print("Starting Latency Test – Ensure server is running at", SERVER_URL)
    print(f"Using {NUM_REQUESTS} requests per batch size.")
    print(f"Test images: {TEST_IMAGES}\n")
    
    all_p50 = {}
    for bs in BATCH_SIZES:
        p50 = test_latency(bs)
        all_p50[bs] = p50
    

    print("\n=== Summary ===")
    pass_count = 0
    total = len(BATCH_SIZES)
    for bs, p50 in all_p50.items():
        if p50 is not None and p50 <= 100:
            status = "PASS (≤100ms)"
            pass_count += 1
        elif p50 is not None:
            status = "FAIL (>100ms)"
        else:
            status = "ERROR (No successes)"
            p50_str = "N/A"
        p50_str = f"{p50:.2f}" if p50 is not None else "N/A"
        print(f"Batch {bs}: p50 = {p50_str} ms – {status}")
        
    goal_status = 'PASS' if pass_count == total else f'FAIL ({pass_count}/{total}); Check logs/images'
    print(f"Sprint Goal: All p50 ≤100ms = {goal_status}.")