from ultralytics import YOLO
import os
import yaml
import torch
from pathlib import Path
import shutil
import argparse

class PoolTableTrainer:
    def __init__(self, data_yaml="yolo_pool_dataset/dataset.yaml",
                 epochs=50,# was 100
                 batch_size=16,
                 imgsz=640,
                 device=None, 
                 runs_dir='' ):
        """
        Initialize the pool table trainer
        """
        self.data_yaml = data_yaml
        self.epochs = epochs
        self.batch_size = batch_size
        self.imgsz = imgsz
        self.runs_dir = runs_dir
        
        # Device selection with MPS support
        if device is None:
            if torch.backends.mps.is_available():
                self.device = 'mps'
            elif torch.cuda.is_available():
                self.device = 'cuda'
            else:
                self.device = 'cpu'
        else:
            self.device = device
        
        # Create runs directory for outputs
        os.makedirs(self.runs_dir, exist_ok=True)
        
        # Load dataset config
        with open(data_yaml, 'r') as f:
            self.dataset_config = yaml.safe_load(f)

    def setup_model(self, pretrained_weights="yolov8s-seg.pt"):
        """
        Setup YOLOv8 model with pretrained weights
        """
        print(f"\nSetting up YOLOv8 model with {pretrained_weights}")
        print(f"Using device: {self.device}")

        # Download pretrained weights if not exists
        if not os.path.exists(pretrained_weights):
            print(f"Downloading {pretrained_weights}...")
            model = YOLO(pretrained_weights)
        else:
            model = YOLO(pretrained_weights)
        
        print(f"Model loaded successfully on {self.device}")
        return model


    def train(self, model):
        """
        Train the model with optimized settings
        """
        print("\nStarting training...")
        print(f"Training configuration:")
        print(f"- Epochs: {self.epochs}")
        print(f"- Batch size: {self.batch_size}")
        print(f"- Image size: {self.imgsz}")
        print(f"- Device: {self.device}")
    
        # Start training with optimized parameters
        results = model.train(
            project=os.path.dirname(self.runs_dir),  # Parent directory
            name=os.path.basename(self.runs_dir),    # Run name
            data=self.data_yaml,    # path to yaml file containing config
            epochs=self.epochs,     
            batch=self.batch_size,
            imgsz=self.imgsz,
            device=self.device,
            nms=True,               # non-maximum suppression for filtering overlapping detections
            iou=0.65,               # Intersection over Union threshold for NMS
            max_det=100,            # Maximum number of detections per image
            cache='disk',           # More stable than RAM cache
            workers=2,              # Number of works for data loading
            patience=5,             # Number of epochs to wait before early stopping if no improvement
            save=True,              # Save checkpoints
            save_period=10,         # Save every 10 epochs
            pretrained=True,
            optimizer='Adam',
            lr0=0.001,              # Learning rate
            lrf=0.01,               # Final learning rate factor
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=3.0,
            warmup_momentum=0.8,
            warmup_bias_lr=0.1,
            box=7.5,                # Box loss weight
            cls=0.5,                # Classification loss weight (up if struggling)
            dfl=1.5,                # Distribution focal loss weight
            plots=True,
            exist_ok=True,
            overlap_mask=True,
            mask_ratio=4,
            single_cls=True,        # Set True for single-class detection
            rect=True,              # Rectangular training for efficiency
            amp=True,               # Automatic mixed precision training
            close_mosaic=10         # Disables mosaic augmentation in last 10 epochs for stability
        ) 
        return results

    def validate(self, model):
        """
        Validate the model
        """
        print("\nRunning validation...")
        try:
            results = model.val(
                data=self.data_yaml,
                imgsz=self.imgsz,
                batch=self.batch_size,
                device=self.device,
                split='val',
                plots=True
            )
            return results
        
        except Exception as e:
            print(f"Error during validation: {str(e)}")
            raise

    def export_model(self, model, format='onnx'):
        """
        Export the model to specified format
        """
        print(f"\nExporting model to {format}...")
        try:
            model.export(format=format)
        except Exception as e:
            print(f"Error exporting model: {str(e)}")
            raise

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train YOLOv8 for pool table segmentation')
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size')
    parser.add_argument('--img-size', type=int, default=640, help='image size')
    parser.add_argument('--weights', type=str, default='yolov8s-seg.pt', help='initial weights path')
    parser.add_argument('--device', type=str, default=None, help='device (mps, cuda, or cpu)')
    args = parser.parse_args()

    print("PyTorch device availability:")
    print(f"MPS (Apple Silicon): {torch.backends.mps.is_available()}")
    print(f"CUDA (NVIDIA): {torch.cuda.is_available()}")

    # Update runs_dir path construction
    project_dir = '/Users/mouse/src/PocketFinder/runs'
    name = 'segment'
    runs_dir = os.path.join(project_dir, name)
    
    # Initialize trainer
    trainer = PoolTableTrainer(
        epochs=args.epochs,
        batch_size=args.batch_size,
        imgsz=args.img_size,
        device=args.device,
        runs_dir=runs_dir 
    )
    
    # Setup model
    model = trainer.setup_model(args.weights)
    
    # Train model
    results = trainer.train(model)
    
    # Validate model
    val_results = trainer.validate(model)
    
    # Export model
    trainer.export_model(model)
    
    print("\nTraining completed successfully!")
    print(f"Results saved in: {trainer.runs_dir}")        

if __name__ == "__main__":
    main()
