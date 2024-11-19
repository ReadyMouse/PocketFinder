# PocketFinder
Automated Finder for Pocket billiards venues. 

Challenges: 
1. Ambiguity in text
2. Limited pixels on object

Finding bars, resturants, and hotels with pool tables is particuarly hard problem because these places often don't list billiards or pool tables as an amentiy on their websites. Open Street Map (OSM) does have a "sport=billiards" tag that can be added to Points of Interest, however without a OSM-enthustist pool player in the area, venues are rarely mapped.  

Text: Sometimes Google reviewers will include a note in their feedback, however there is high ambiguity between "pool tables", "pool", and "table". Eg "I really enjoyed playing in the pool, and got a table right away at the resturant." Rarely will a reviewer use "billiards" for a dive bar, unless it's a pool hall or higher-scale establishment.  

Images: Scanning the photos, pool tables often show as small slivers in the back of full-room shots, making identifying the billiards table difficult. 

-> Text Processing: TODO

-> Image Processing: This repo approaches the problem using an image segmentation model of a pre-trained YOLOv8, fine-tuned on pool tables from OpenImages.  

The developers of PocketFinder would love for users to add the pool tables found to OSM with the "sport=billiards" tag. Query through Overpass Turbo API, or phone apps.  
https://www.openstreetmap.org/
https://overpass-turbo.eu/#


## Setting up a Running Environment
Set up a working environmnet based off the requirements.txt. Using conda or miniconda. 

Recommended order: 
```bash
conda create --name pool python=3.10
conda activate pool
conda install pytorch pandas
pip install -r requirements.txt
```

## Downloading the pool table images from openImages (and no pool tables)

```bash
python3 openimages-pool-table-image-downloader.py 
python3 negative-dataset-downloader.py 
```

(Optional) Check the images downloaded properly 

```bash 
python3 openimages-pool-checker.py 
```

Convert Openimages to YOLO format 

```bash
python3 openimages-to-yolo-converter.py # (soon to be depricated)
ython3 openimages-to-yolo-general.py
```

## Fine-tuning YOLO for pool table recognition

```bash
python3 fine-tune-yolo-mps.py
```

## Inference on New Single Images  
(Note paths are hardcoded and in progress.)

```bash
python3 inference.py
```

Note: Sections of this code was generated with Claude AI. 