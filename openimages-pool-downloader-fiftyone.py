import fiftyone as fo
import fiftyone.zoo as foz
from fiftyone import ViewField as F
import os
import json
from tqdm import tqdm
import shutil
from PIL import Image 

class OpenImagesPoolDownloader:
    def __init__(self, output_dir="pool_table_dataset"):
        self.output_dir = output_dir
        self.images_dir = os.path.join(output_dir, "images")
        self.masks_dir = os.path.join(output_dir, "masks")
        self.annotations_dir = os.path.join(output_dir, "annotations")
        
        # Pool/billiards related labels
        self.pool_labels = [
            "Billiard table",
            "Pool table",
            "Billiards",
            "Bar billiards",
            "English billiards",
            "Pocket billiards"
        ]
        
        for directory in [self.images_dir, self.masks_dir, self.annotations_dir]:
            os.makedirs(directory, exist_ok=True)

    def download_dataset(self):
        """Download pool table images and segmentations from OpenImages"""
        print("Starting download of pool/billiards table dataset...")
        
        # Download OpenImages dataset with our specific labels
        dataset = foz.load_zoo_dataset(
            "open-images-v7",
            split="train",
            label_types=["segmentations"],
            classes=self.pool_labels,
            max_samples=1000  # Adjust this number as needed
        )
        
        print(f"\nDataset downloaded. Found {len(dataset)} samples")
        
        # Print available fields
        print("\nAvailable dataset fields:")
        for field in dataset.get_field_schema():
            print(f"  - {field}")

        # Get sample information
        if len(dataset) > 0:
            sample = dataset.first()
            print("\nSample metadata:")
            for field in sample.field_names:
                value = sample[field]
                print(f"  {field}: {type(value)}")
                if hasattr(value, 'detections'):
                    print("    Detections:")
                    for det in value.detections:
                        print(f"      - Label: {det.label}, Confidence: {det.confidence}")
        

      # Export images and annotations
        print("\nExporting dataset...")
        for sample in tqdm(dataset):
            # Export image
            img_filename = os.path.basename(sample.filepath)
            img_path = os.path.join(self.images_dir, img_filename)
            if not os.path.exists(img_path):
                # sample.copy_image(img_path)
                shutil.copy2(sample.filepath, img_path)
            
            # Metadata
            with Image.open(img_path) as img:
                return {"width": img.width, "height": img.height}

            # Export mask if available
            try:
                if hasattr(sample, 'segmentations'):
                    for idx, det in enumerate(sample.segmentations.detections):
                        if det.label == self.class_label:
                            mask_filename = f"{os.path.splitext(img_filename)[0]}_mask_{idx}.png"
                            mask_path = os.path.join(self.masks_dir, mask_filename)
                            if not os.path.exists(mask_path):
                                det.mask.save(mask_path)
            except Exception as e:
                print(f"Warning: Could not export mask for {img_filename}: {str(e)}")
        
        # Create dataset summary
        print("\nCreating dataset summary...")
        summary = {
            "num_images": len(dataset),
            "class_label": self.pool_labels[1],
            "images": [],
            "dataset_info": {
                "name": dataset.name,
                "num_samples": len(dataset),
                "split": dataset.info.get("split", "unknown"),
                "version": dataset.info.get("version", "unknown")
            }
        }
        
        # Add information about each sample
        for sample in dataset:
            sample_info = {
                "image_id": sample.id,
                "filename": os.path.basename(sample.filepath),
                "width": sample.metadata.width if hasattr(sample, 'metadata') else None,
                "height": sample.metadata.height if hasattr(sample, 'metadata') else None,
                "num_detections": len(sample.segmentations.detections) if hasattr(sample, 'segmentations') else 0,
            }
            summary["images"].append(sample_info)
        
        # Save summary
        summary_path = os.path.join(self.annotations_dir, "dataset_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        
        print("\nDataset download completed!")
        print(f"Total images: {summary['num_images']}")
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
