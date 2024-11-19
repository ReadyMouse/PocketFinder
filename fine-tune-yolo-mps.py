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
                 runs_dir='',
                 pretrained_weights="yolov8x-seg.pt" ):
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

        # Download pretrained weights if not exists
        print(f"\nSetting up YOLOv8 model with {pretrained_weights}")
        print(f"Using device: {self.device}")

        if not os.path.exists(pretrained_weights):
            print(f"Downloading {pretrained_weights}...")
            self.model = YOLO(pretrained_weights)
        else:
            self.model = YOLO(pretrained_weights)
        print(f"Model loaded successfully on {self.device}")

        return 

    def print_model_weights(self, model, layer_names=None, first_n=5):
        """
        Print weights from specified layers or first few weights of all layers.
        """
        state_dict = model.model.state_dict()  # Note the double model - YOLOv8 specific
        
        if layer_names:
            weights_to_check = {k: v for k, v in state_dict.items() if any(name in k for name in layer_names)}
        else:
            weights_to_check = state_dict
            
        for name, param in weights_to_check.items():
            if param.dim() > 0:  # Skip scalar parameters
                print(f"\nLayer: {name}")
                print(f"Shape: {param.shape}")
                print(f"First {first_n} weights: {param.flatten()[:first_n].tolist()}")
                print(f"Mean: {param.mean().item():.6f}")
                print(f"Std: {param.std().item():.6f}")

    def inspect_training_weights(self, epoch):
        """
        Inspect model weights at specific epoch
        """
        print(f"\n=== Weights at epoch {epoch} ===")
        
        # YOLOv8 specific layers
        important_layers = [
            '0',  # First conv layer
            '24',  # Detection layers
            '23'  # Pre-detection layers
        ]
        
        self.print_model_weights(self.model, important_layers)

    def train(self):
        """
        Train the model with optimized settings
        """
        print("\nStarting training...")
        print(f"Training configuration:")
        print(f"- Epochs: {self.epochs}")
        print(f"- Batch size: {self.batch_size}")
        print(f"- Image size: {self.imgsz}")
        print(f"- Device: {self.device}")
        model = self.model
    
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
            iou=0.5,               # Intersection over Union threshold for NMS
            max_det=10,             # Maximum number of detections per image
            cache='disk',           # More stable than RAM cache
            workers=2,              # Number of works for data loading
            patience=5,             # Number of epochs to wait before early stopping if no improvement
            save=True,              # Save checkpoints
            save_period=10,         # Save every 10 epochs
            pretrained=True,
            optimizer='Adam',
            lr0=0.01,               # Learning rate
            lrf=0.01,               # Final learning rate factor
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=5.0,
            warmup_momentum=0.8,
            warmup_bias_lr=0.1,
            box=6.5,                # Box loss weight
            cls=0.8,                # Classification loss weight (up if struggling)
            dfl=1.5,                # Distribution focal loss weight
            plots=True,
            exist_ok=True,
            overlap_mask=True,
            mask_ratio=4,
            # single_cls=True,        # Set True for single-class detection
            rect=True,              # Rectangular training for efficiency
            amp=True,               # Automatic mixed precision training
            close_mosaic=10         # Disables mosaic augmentation in last 10 epochs for stability
        ) 
        # inspect_training_weights(self.model, 1)
        return results

    def validate(self):
        """
        Validate the model
        """
        model = self.model
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

    def export_model(self, format='onnx'):
        """
        Export the model to specified format
        """
        model = self.model
        print(f"\nExporting model to {format}...")
        try:
            model.export(format=format)
        except Exception as e:
            print(f"Error exporting model: {str(e)}")
            raise

    def inference(self, image_path, output_dir, conf_threshold=0.25):
        """
        Run inference on a single image or directory of images
        """        
        model = self.model
        # Handle both single images and directories
        image_paths = []
        if os.path.isfile(image_path):
            image_paths = [image_path]
        else:
            image_paths = [str(p) for p in Path(image_path).glob('*') 
                        if p.suffix.lower() in ['.jpg', '.jpeg', '.png']]
        
        all_results = {}

        for img_path in image_paths:
            # Run inference
            results = model.predict(
                source=img_path,
                conf=conf_threshold,
                show=True,  # Display the annotated image
                save=True,   # Save the results
                project=output_dir,
                name='predictions'
            )

            # Extract results for this image
            img_results = []
            for r in results[0].boxes:
                result = {
                    'confidence': float(r.conf.item()),
                    'bbox': r.xyxy[0].tolist(),  # Convert tensor to list
                }
                if hasattr(r, 'cls'):
                    result['class'] = int(r.cls.item())
                img_results.append(result)
                
            all_results[os.path.basename(img_path)] = img_results
            
            print(f"Processed {img_path}")
            print(f"Found {len(results[0].boxes)} objects")
        
        # Save results to JSON
        results_file = output_dir / 'predictions' / 'results.json'
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=4)
        
        print(f"\nResults saved to: {output_dir}")
        print(f"- Annotated images: {output_dir}/predictions/")
        print(f"- JSON results: {results_file}")

        return results


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train YOLOv8 for pool table segmentation')
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='batch size')
    parser.add_argument('--img-size', type=int, default=640, help='image size')
    parser.add_argument('--weights', type=str, default='yolov8x-seg.pt', help='initial weights path')
    parser.add_argument('--device', type=str, default=None, help='device (mps, cuda, or cpu)')
    args = parser.parse_args()

    print("PyTorch device availability:")
    print(f"MPS (Apple Silicon): {torch.backends.mps.is_available()}")
    print(f"CUDA (NVIDIA): {torch.cuda.is_available()}")

    # Update runs_dir path construction
    project_dir = '/Users/ekelley/src/Pocket-Finder/runs'
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
    # model = trainer.setup_model(args.weights)

    trainer.inspect_training_weights("start")
    
    # Train model
    results = trainer.train()
    
    # Validate model
    val_results = trainer.validate()
    
    # Export model
    trainer.export_model()
    
    print("\nTraining completed successfully!")
    print(f"Results saved in: {trainer.runs_dir}")        

if __name__ == "__main__":
    main()
