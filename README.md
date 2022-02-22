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
