from google.cloud import storage

BUCKET_NAME = "mlops-try"
TEST_FILE_NAME = "gcs_write_test.txt"

try:
    print("Attempting to connect to GCS...")
    client = storage.Client()
    
    print(f"Connecting to bucket: {BUCKET_NAME}")
    bucket = client.get_bucket(BUCKET_NAME)
    
    print(f"Creating test file: {TEST_FILE_NAME}")
    blob = bucket.blob(TEST_FILE_NAME)
    
    blob.upload_from_string("This is a successful write test.")
    
    print("\n--- SUCCESS! ---")
    print(f"File '{TEST_FILE_NAME}' was successfully written to your bucket.")
    print("Your authentication and permissions are working correctly.")

except Exception as e:
    print("\n--- FAILED! ---")
    print("Could not write to GCS. See error below:")
    print(e)
    print("\nThis is an authentication or permissions problem.")
    print("Make sure you ran 'gcloud auth application-default login' and that your user has 'Storage Object Admin' permissions on this bucket.")