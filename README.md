# PocketFinder
Automated Finder for Pocket billiards venues. 

## Step 0: Set up a working environmnet based off the requirements.txt

## Step 1: Download the pool table images from openImages 

```bash
python3 openimages-pool-table-image-downloader.py 
```

## Step 2: (Optional) Check the images downloaded properly 

```bash 
python3 openimages-pool-checker.py 
```

## Step 3: Convert Openimages to YOLO format 

```bash
python3 openimages-to-yolo-converter.py
```

