# Road extraction from satellite images

**Intent** This is a class project for a course at [Klagenfurt University](https://www.aau.at).

## Project goal
The project aims to extract the roads from a satellite image of a rural
or urban area. It focuses on using image processing techniques, rather 
than AI approaches.

## Usage
Currently all functions assume the image files to be contained in a subdirectory called "data".

## TODO
* Write a callable segmentation function that returns the clustered image
* Write a post-processing function to extract the road network
* Write a function that combines both processes

Further extensions
* Speed up the clustering process
* Refine the clustering/segmentation
* Make the output more visually appealing

## Track record

* (Jana) Wrote inital segmentation tool based on Gaussian mixtures
* (Arke) Some speed optimizations for the segmetnation process


## Segmentation

## Building extraction
The building_detection function allows us to locate and highlight buildings on images. 

First, we apply a bilateral filter, various open/close/dilate morphological operations & canny edge detection to obtain candidate building contours.

Next, we remove structures that are too long (potential streets), too small (noise) or within other structures to filter out real building contours.

Then, we distinguish between rectangular/trapezoid that can be bound by clean, 4 edge boxes, and uniquely shaped buildings to facilitate representation.

Lastly, we display an accuracy score demonstrating how many buildings our code was able to detect in the image.

Various parameters allow us to fine-tune the model to specific images and maximize our accuracy score.

This model is optimized for high quality, high luminosity images of rural neighborhoods with a modern road system (tarmac) that is aligned with the borders of the image, well aligned, bright colored and sufficiently distanced houses, and as little shadows as possible (images ideally taken at noon).

Possible improvements include 

* rotating the image to detect houses at and angle that havent been detected yet.

* distinguish between infrastructure such as parkings/pools/hangars and habitable buildings.

* detect buildigs that are lower contrast

* eliminate shadows

* distinguish between roads, railroads, rivers, ...

* detect individual houses from groups of houses packed closely together

* delimitate the compound belonging to each house using hierarchy information


## Road extraction
