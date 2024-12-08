from ultralytics import YOLO
import cv2
import os
from pathlib import Path
import json

class PoolTableInference():
    def __init__(self, model_path = './First-working-copy/weights/best.pt', 
                 image_path = './hotel_photos',
                 output_dir='./hotel_results/', 
                 conf_threshold=0.5,
                 save_negative = False):
        self.model_path = model_path
        self.image_path = image_path
        self.output_dir = output_dir
        self.conf_threshold = conf_threshold

    def run_inference(self, model_path, image_path, output_dir, conf_threshold=0.5, save_negative=False):
        """
        Run classification inference on a single image or directory of images
        """
        # Load the model
        model = YOLO(model_path)
        print('Successfully loaded the model weights.')

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
                save=True,   # Save the results
                project= os.path.dirname(output_dir),
                name=os.path.basename(output_dir),
                exist_ok=True 
            )

            # Extract classification results
            result = {
                'class_name': results[0].names[results[0].probs.top1],  # Get class name
                'confidence': float(results[0].probs.top1conf),  # Get confidence
                'class_index': int(results[0].probs.top1)  # Get class index
            }

            if not save_negative and result['class_name'] == 'no_pool_table': # already marked via conf_threshold
                # We don't want to save negatice, and confident no pool table
                os.remove(os.path.join(output_dir,os.path.basename(img_path)))
                print(f'Removing negative photo: {os.path.join(output_dir,os.path.basename(img_path))}')


            all_results[os.path.basename(img_path)] = result
            
            print(f"\nProcessed {img_path}")
            print(f"Prediction: {result['class_name']} ({result['confidence']:.2f} confidence)")
        
        # Save results to JSON
        results_file = os.path.join(output_dir, 'results.json')
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=4)
        
        print(f"\nResults saved to: {output_dir}")
        print(f"- JSON results: {results_file}")

        return all_results


if __name__ == "__main__":
    # model_path = './Users/mouse/src/PocketFinder/yolov8x-seg.pt' # base yolo 
    model_path = './First-working-copy/weights/best.pt'
    image_path = './hotel_photos'
    # image_path = './my_own_pics'
    save_path = './hotel_results'

    engine = PoolTableInference()
    results = engine.run_inference(model_path=model_path, image_path=image_path, output_dir=save_path )