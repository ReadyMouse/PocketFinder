import fiftyone as fo
import fiftyone.zoo as foz
import os
from tqdm import tqdm
import json
import shutil
from PIL import Image
import numpy as np

class OpenImagesPoolDownloader:
    def __init__(self, output_dir="pool_table_dataset"):
        self.output_dir = output_dir
        self.images_dir = os.path.join(output_dir, "images")
        self.class_label = "Billiard table"  # Primary class label
        
        # Create all necessary directories
        for directory in [self.images_dir]:
            os.makedirs(directory, exist_ok=True)

    def download_dataset(self):
        """Download complete pool table images"""
        print("\nStarting download of pool/billiards table dataset...")
        
        # Load dataset
        dataset = foz.load_zoo_dataset(
            "open-images-v7",
            split="test",
            classes=[self.class_label],
            # dataset_dir="/path/to/your/desired/location", # output
            max_samples=1000
        )

        print(f"\nDataset downloaded. Found {len(dataset)} samples")

        # Process dataset
        print("\nProcessing dataset...")
        successful_images = 0
        
        for sample in tqdm(dataset, desc="Processing samples"):
            try:
                # Process image
                img_filename = os.path.basename(sample.filepath)
                img_path = os.path.join(self.images_dir, img_filename)
                
                # Copy image if it doesn't exist
                if not os.path.exists(img_path):
                    try:
                        shutil.copy2(sample.filepath, img_path)
                        successful_images += 1
                    except Exception as e:
                        print(f"\nError copying image {img_filename}: {str(e)}")
                        continue
                else:
                    successful_images += 1

            except Exception as e:
                print(f"\nError processing sample {sample.id}: {str(e)}")
                continue

        # Save summary
        print("\nDataset download completed!")
        print(f"Total samples found: {len(dataset)}")
        print(f"Successfully exported images: {successful_images}")
        print(f"\nDataset saved in: {self.output_dir}")

def main():
    try:
        downloader = OpenImagesPoolDownloader()
        downloader.download_dataset()
    except Exception as e:
        print(f"\nError during execution: {str(e)}")
        import traceback
        print("\nFull traceback:")
        print(traceback.format_exc())

if __name__ == "__main__":
    main()
