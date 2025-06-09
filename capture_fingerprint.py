import MFS100

def capture_fingerprint():
    """
    Captures a fingerprint image from the connected sensor and saves it as an image file.
    """
    try:
        # Initialize the fingerprint device
        mfs100 = MFS100.MFS100()

        if mfs100.Init():
            print("Fingerprint device initialized successfully.")
        else:
            print("Failed to initialize fingerprint device.")
            return None

        # Capture the fingerprint
        print("Place your finger on the sensor...")
        if mfs100.CaptureFinger(timeout=10000):  # Timeout in milliseconds
            fingerprint_image = mfs100.GetLastCapturedImage()
            with open("fingerprint.bmp", "wb") as f:
                f.write(fingerprint_image)
            print("Fingerprint captured successfully!")
            return "fingerprint.bmp"
        else:
            print("Failed to capture fingerprint.")
            return None

    except Exception as e:
        print(f"Error capturing fingerprint: {e}")
        return None

if __name__ == "__main__":
    capture_fingerprint()
