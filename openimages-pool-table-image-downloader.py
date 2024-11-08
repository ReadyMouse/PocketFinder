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
        self.masks_dir = os.path.join(output_dir, "masks")
        self.annotations_dir = os.path.join(output_dir, "annotations")
        self.class_label = "Billiard table"  # Primary class label
        
        # Create all necessary directories
        for directory in [self.images_dir, self.masks_dir, self.annotations_dir]:
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
        """Download complete pool table dataset including images and masks"""
        print("\nStarting download of pool/billiards table dataset...")
        
        # Load dataset
        dataset = foz.load_zoo_dataset(
            "open-images-v7",
            split="train",
            label_types=["segmentations"],
            classes=[self.class_label],
            max_samples=1000
        )
        
        print(f"\nDataset downloaded. Found {len(dataset)} samples")
        
        # Print dataset information
        print("\nDataset structure:")
        if len(dataset) > 0:
            sample = dataset.first()
            print("\nSample fields:")
            for field in sample.field_names:
                value = getattr(sample, field)
                print(f"  {field}: {type(value)}")
                if hasattr(value, "detections"):
                    print("    Detections:")
                    for det in value.detections:
                        print(f"      - Label: {det.label}")

        # Process dataset
        print("\nProcessing dataset...")
        successful_images = 0
        successful_masks = 0
        dataset_info = []
        
        for sample in tqdm(dataset, desc="Processing samples"):
            try:
                # Process image
                img_filename = os.path.basename(sample.filepath)
                img_path = os.path.join(self.images_dir, img_filename)
                base_name = os.path.splitext(img_filename)[0]
                
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
                
                # Process masks
                mask_info = []
                segmentations = sample.ground_truth if hasattr(sample, "ground_truth") else None
                
                if segmentations and hasattr(segmentations, "detections"):
                    for idx, det in enumerate(segmentations.detections):
                        if det.label == self.class_label:
                            try:
                                # Get mask data
                                mask = None
                                if hasattr(det, "mask"):
                                    mask = det.mask
                                elif hasattr(det, "segmentation"):
                                    mask = det.segmentation
                                
                                if mask is not None:
                                    mask_filename = f"{base_name}_mask_{idx}.png"
                                    mask_path = os.path.join(self.masks_dir, mask_filename)
                                    
                                    # Save mask
                                    if not os.path.exists(mask_path):
                                        if hasattr(mask, "save"):
                                            mask.save(mask_path)
                                        elif isinstance(mask, np.ndarray):
                                            Image.fromarray(mask).save(mask_path)
                                        successful_masks += 1
                                    else:
                                        successful_masks += 1

                                    mask_info.append({
                                        "mask_file": mask_filename,
                                        "detection_index": idx
                                    })
                            except Exception as e:
                                print(f"\nError processing mask {idx} for {img_filename}: {str(e)}")

                # Store sample information
                sample_info = {
                    "image_id": sample.id,
                    "filename": img_filename,
                    "width": img_metadata["width"],
                    "height": img_metadata["height"],
                    "masks": mask_info
                }
                dataset_info.append(sample_info)

            except Exception as e:
                print(f"\nError processing sample {sample.id}: {str(e)}")
                continue

        # Create dataset summary
        summary = {
            "dataset_info": {
                "name": "OpenImages V7 Pool Tables Dataset",
                "class_label": self.class_label,
                "total_samples": len(dataset),
                "successful_images": successful_images,
                "successful_masks": successful_masks,
                "split": "train"
            },
            "samples": dataset_info
        }

        # Save summary
        summary_path = os.path.join(self.annotations_dir, "dataset_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        print("\nDataset download completed!")
        print(f"Total samples found: {len(dataset)}")
        print(f"Successfully exported images: {successful_images}")
        print(f"Successfully exported masks: {successful_masks}")
        print(f"\nDataset saved in: {self.output_dir}")
        print(f"Summary saved to: {summary_path}")

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
