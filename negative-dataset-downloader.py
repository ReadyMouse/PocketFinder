import fiftyone as fo
import fiftyone.zoo as foz
import fiftyone.utils.openimages as fouo
import os
from tqdm import tqdm
import json
import shutil
from PIL import Image
import numpy as np

class OpenImagesNegativeDownloader:
    def __init__(self, output_dir="negative_dataset"):
        self.output_dir = output_dir
        self.images_dir = os.path.join(output_dir, "images")
        self.annotations_dir = os.path.join(output_dir, "annotations")
        
        # First, let's check available classes
        print("Checking available OpenImages classes...")
        self.available_classes = fouo.get_classes()
        
        # Print relevant classes
        print("\nAvailable relevant classes:")
        relevant_terms = ["bar", "counter", "dining table", "bar stool", "chair", "table"]
        for class_name in self.available_classes:
            if any(term.lower() in class_name.lower() for term in relevant_terms):
                print(f"- {class_name}")
        
        # Target classes we want - we'll update these based on what's available
        self.target_classes = [
            "Coffee table",
            "Countertop",
            "Table",
            "Kitchen & dining room table"
        ]
        
        # Class we want to exclude
        self.exclude_class = "Billiard table"
        
        # Create necessary directories
        for directory in [self.images_dir, self.annotations_dir]:
            os.makedirs(directory, exist_ok=True)

    def get_image_metadata(self, image_path):
        """Get image dimensions safely"""
        try:
            with Image.open(image_path) as img:
                return {"width": img.width, "height": img.height}
        except Exception as e:
            print(f"Warning: Could not get metadata for {image_path}: {str(e)}")
            return {"width": None, "height": None}

    def download_dataset(self):
        """Download dataset of bars, restaurants, and hotels excluding pool tables"""
        print("\nStarting download of negative dataset...")
        
        # First get the pool table images to exclude
        print("\nGetting pool table image IDs to exclude...")
        pool_table_dataset = foz.load_zoo_dataset(
            "open-images-v7",
            split="train",
            label_types=["detections"],
            classes=[self.exclude_class],
            dataset_name="openimages-pool-tables"
        )
        pool_table_ids = set(sample.id for sample in pool_table_dataset)
        
        # Download and process each target class
        combined_dataset = []
        
        for target_class in self.target_classes:
            print(f"\nDownloading {target_class} images...")
            
            try:
                # Create unique dataset name for each class
                dataset_name = f"openimages-{target_class.lower().replace(' ', '-')}"
                
                dataset = foz.load_zoo_dataset(
                    "open-images-v7",
                    split="train",
                    label_types=["detections"],
                    classes=[target_class],
                    max_samples=350,
                    dataset_name=dataset_name
                )
                
                # Manually filter out pool table images
                for sample in dataset:
                    if sample.id not in pool_table_ids:
                        combined_dataset.append(sample)
                        
            except Exception as e:
                print(f"\nError downloading {target_class} images: {str(e)}")
                continue
        
        print(f"\nTotal filtered samples: {len(combined_dataset)}")
        
        # Process filtered dataset
        print("\nProcessing filtered dataset...")
        successful_images = 0
        dataset_info = []
        
        for sample in tqdm(combined_dataset, desc="Processing samples"):
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

                # Get image metadata
                img_metadata = self.get_image_metadata(img_path)
                
                # Get detected classes for verification
                detected_classes = []
                if hasattr(sample, "ground_truth") and hasattr(sample.ground_truth, "detections"):
                    detected_classes = [det.label for det in sample.ground_truth.detections]
                
                # Store sample information
                sample_info = {
                    "image_id": sample.id,
                    "filename": img_filename,
                    "width": img_metadata["width"],
                    "height": img_metadata["height"],
                    "detected_classes": detected_classes
                }
                dataset_info.append(sample_info)

            except Exception as e:
                print(f"\nError processing sample {sample.id}: {str(e)}")
                continue

        # Create dataset summary
        summary = {
            "dataset_info": {
                "name": "OpenImages V7 Negative Dataset",
                "target_classes": self.target_classes,
                "excluded_class": self.exclude_class,
                "total_samples": len(combined_dataset),
                "successful_images": successful_images,
                "split": "train"
            },
            "samples": dataset_info
        }

        # Save summary
        summary_path = os.path.join(self.annotations_dir, "dataset_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        print("\nNegative dataset download completed!")
        print(f"Total samples found: {len(combined_dataset)}")
        print(f"Successfully exported images: {successful_images}")
        print(f"\nDataset saved in: {self.output_dir}")
        print(f"Summary saved to: {summary_path}")

        # Cleanup datasets
        fo.delete_dataset("openimages-pool-tables")
        for target_class in self.target_classes:
            dataset_name = f"openimages-{target_class.lower().replace(' ', '-')}"
            try:
                fo.delete_dataset(dataset_name)
            except:
                pass

def main():
    try:
        downloader = OpenImagesNegativeDownloader()
        downloader.download_dataset()
    except Exception as e:
        print(f"\nError during execution: {str(e)}")
        import traceback
        print("\nFull traceback:")
        print(traceback.format_exc())

if __name__ == "__main__":
    main()
