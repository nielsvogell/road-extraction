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
The segmentation (segment) function divides the image into 5 clusters by color using the gaussian mixture clustering algorithm.<br>
Based on the average color of each cluster, it gets assigned a label: "road", "building", "background".<br>
The mask for the "building" label can be used for the Building Extraction, and the mask for the "road" label for the Road Extraction.
* run show_color_evaluation() to see segmentation result
* run test_segment() to see masks of labels

### Possible improvements
* faster clustering algorithm
* finding optimal number of clusters for gaussian mixture (using 5 right now)

## Building extraction
The building_detection function allows us to locate and highlight buildings on images. It follows these steps 

1. First, we apply a bilateral filter, various open/close/dilate morphological operations & canny edge detection to obtain candidate building contours.
2. Next, we remove structures that are too long (potential streets), too small (noise) or within other structures to filter out real building contours.
3. Then, we distinguish between rectangular/trapezoid that can be bound by clean, 4 edge boxes, and uniquely shaped buildings to facilitate representation.
4. Lastly, we display an accuracy score demonstrating how many buildings our code was able to detect in the image.

Various parameters allow us to fine-tune the model to specific images and maximize our accuracy score.

This model is optimized for high quality, high luminosity images of rural neighborhoods with a modern road system (tarmac) that is aligned with the borders of the image, well aligned, bright colored and sufficiently distanced houses, and as little shadows as possible (images ideally taken at noon).

### Possible improvements include 

* rotating the image to detect houses at and angle that haven't been detected yet.

* distinguish between infrastructure such as parks/pools/hangars and habitable buildings.

* detect buildings that are lower contrast

* eliminate shadows

* distinguish between roads, railroads, rivers, ...

* detect individual houses from groups of houses packed closely together

* delimit the compound belonging to each house using hierarchy information


## Road extraction
The road extraction was inspired by a book chapter by Jin, et al.[^jin2012]. The paper discribes a three step algorithm.
1. Segment the image using a homogeneity histogram to detect suitable thresholds
2. Thin the mask for clusters assigned to roads using an algorithm by Wang et al.[^wang1989]
3. Extract intersections and prune unwanted dangling ends.

We implemented our own [segmentation approach](#segmentation) instead of the first step, since it works better.
For thinning we also deviated from the suggested algorithm, mainly because that algorithm was written long before 
OpenCV (2011) and NumPy (2006, or 1995) were written. The basic ideas are however similar:

1. Find the contour of the road mask
2. Remove contour pixels, if there are interior pixels beside them
3. Iterate through 1 and 2 until nothing can be removed

This process leaves us with a one or two pixel wide road network. Since it ideally never opens up a line, this approach 
preserves the topology. It is slightly less robust than the literature approach (according to the paper), but it runs
significantly faster due to NumPy's fast array operations. We implemented and tested a variant of the suggested approach
to final adjustments, but until now the implementation does not fully work (unfortunately there are ambiguities in 
the description of the algorithm in the original paper and Jin et al. did not comment on the implementation).

The next step of the road extraction approach would refine the road network. It is however dependent on the network 
lines to never exceed two pixel width, to detect intersection points. Until the thinning algorithm can be improved, we 
remain with the unprocessed extracted road network.

The results however are promising and allow for a reasonably good extraction of the basic road network. There are a few
issues with the approach that we can hint at:
1. Gaps in the segmented road mask (caused for instance by shadows, bridges or canopy) severely limit the performance of
   the algorithm. Since the algorithm mostly preserves topology, any hole induces a loop and any gap prevents the 
   connection of stretches of road.
2. Driveways are easily detected as roads. They cause the mask to have small bumps towards the side. The symmetry of the
   thinning algorithm causes these bumps to create slight curves, which makes the extracted road zigzag over the actual
   road area.
3. Removing noise through morphological operations on the road mask also removes small details that might be of 
   interest. For instance, a small patch of grass on a roundabout or a strip between lanes may vanish, which potentially
   removes important features of a road.

To further improve our approach, we suggest a look into the following ideas:
1. Roads are mostly straight forward because they conform with how we want to use cars. That means that mostly roads
   do not abruptly stop or start and end in the middle of nowhere. One could use this assumption to reconnect road 
   components that may have become disconnected through segmentation or thinning.
2. For the same reason, roads do not zigzag randomly. Once intersection points can be extracted, one could straighten
   roads by drawing a smooth curve between the intersection points, using the road mask as a constraint on the amount
   of sideways deviation.

## References
[^wang1989]: 
    P. S. P. Wang and Y. Y. Zhang,
    “A Fast and Flexible Thinning Algorithm",
    *IEEE Trans. Comput.*,
    vol. 38, no. 5, pp. 741–745, 1989

[^jin2012]: 
    H. Jin, M. Miska, E. Chung, M. Li, and Y. Feng,
    "Road Feature Extraction from High Resolution Aerial Images Upon Rural Regions Based on Multi-Resolution Image Analysis 
    and Gabor Filters." 
    In: *Remote Sensing - Advanced Techniques and Platforms*, B. Escalante, Ed. InTech, 2012
