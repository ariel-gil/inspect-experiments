from datasets import load_dataset

def test_math_dataset_loading():
    try:
        # Load the MATH dataset
        ds = load_dataset("nlile/hendrycks-MATH-benchmark")
        
        # Basic checks to verify dataset is loaded correctly
        print("Dataset loaded successfully!")
        
        # Print some basic information about the dataset
        print("Dataset info:")
        print(f"Number of splits: {len(ds)}")
        
        # Try to access the first few samples to ensure data is accessible
        for split_name, split in ds.items():
            print(f"\nSplit: {split_name}")
            print(f"Number of samples: {len(split)}")
            
            # Print first few samples (if available)
            if len(split) > 0:
                print("First sample:")
                print(split[0])
            
    except Exception as e:
        print(f"Error loading dataset: {e}")

# Run the test
if __name__ == "__main__":
    test_math_dataset_loading()