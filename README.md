# Trackable
**Disclaimer: This project is meant for research purpose only.**

## Overview
Tracking humans in a hallway.

 - The project deals with tracking humans in a narrow hallway under different lighting conditions.
  - Unlike other [MOT](https://motchallenge.net/) models, we aim to
   track people without any training, that means, all tracking is done online.
   - There are many state-of-the-art models and architectures,
   that have been a large source of our inspiration (they are all listed under references below).
   - The videos were obtained from [this link.](http://www.santhoshsunderrajan.com/datasets.html#hfh_tracking)

## Getting Started

It is suggested that you create a virtual environment.
The requirements are given below, we've also included a requirements.txt

```
torchvision==0.6.0a0+82fd1c8
torch==1.5.0
dlib==19.19.0
numpy==1.18.2
scipy==1.4.1
opencv_contrib_python==4.2.0.34
```

## Project Structure
The directory structure is as follows:

 - Core: This contains the core modules for the project.
	- Detector:
		- This uses the YOLOv3 model to detect humans after every N frames.
		- Once the object is detected, the matching algorithm is used to see if the object is already being tracked, and thus avoiding re-identification.
		- If the object is a new object (or if re-ID fails), the next available ID is given to the new (or re-IDed) object.
	- Trackable:
		- This is the trackable object which contains a ID and other data regarding the object.
	- Tracker:
		- This implements the correlation tracker (from dlib) along with centroid tracking.
		- It updates the position of the trackable object in each frame.
	- Matcher:
		- This contains the matching function which returns a score of how similar two objects are.
		- The score is the weighted average of correlation score, Bhattacharyya score (obtained from their histograms) and the cosine similarity from the feature extractor.
	- Feature Extractor:
		- This contains the ResNet model (from PyTorch), which is used to perform feature extraction and return a cosine similarity score.
- Utils: This contains helper functions and constants used by the core module.

## License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## References
- [Multi-Object Tracking with dlib.](https://www.pyimagesearch.com/2018/10/29/multi-object-tracking-with-dlib/)
- [DeepSort.](https://github.com/nwojke/deep_sort)
- [Tracking without bells and whistles.](https://arxiv.org/pdf/1903.05625.pdf)
- [Person re-ID.](https://github.com/layumi/Person_reID_baseline_pytorch)
- [Kalman Filter.](https://www.bzarg.com/p/how-a-kalman-filter-works-in-pictures/)
