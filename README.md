# grabcut_refine

Refine silhouettes produced by mask RCNN using the grabcut algorithm.

## Installation

Install the following dependencies:
	opencv
	matplotlib



## Usage

The input data requried is the sequence frames and the related RCNN silhouettes.


Use the following command to refine the silhouettes:

```
python refine_grabcut.py $IMAGE_FOLDER $SILHOUETTE_FOLDER $OUTPUT_FOLDER
```

Note: This is set up to work with the CASIA dataset which has the following file structure, _subject/sequence/angle/frames.png_. To use a different dataset either ensure it has the same file structure or modify line 65 of _refine_grabcut.py_ to chnage the output folder to have the correct format. 

