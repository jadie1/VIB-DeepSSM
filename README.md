# VIB-DeepSSM
Implementation of "From Images to Probabilistic Anatomical Shapes: A Deep Variational Bottleneck Approach"

## Supershapes Example
An example ([run_experiment.py](run_experiment.py)) is demonstrated using a synthetic dataset called "Supershapes". This data was generated using the [Shape Cohort Generation](http://sciinstitute.github.io/ShapeWorks/6.3/notebooks/getting-started-with-shape-cohort-generation.html) package from [ShapeWorks](http://sciinstitute.github.io/ShapeWorks/6.3/index.html). 

A set of 1200 supershape meshes were generated with between 3 and 7 lobes.

The input images were generated from the meshes using the [Shape Cohort Generation](http://sciinstitute.github.io/ShapeWorks/6.3/notebooks/getting-started-with-shape-cohort-generation.html) package and blurred various degrees between 1 and 8. 

The training points were optimized using [ShapeWorks Incremental Optimization](http://sciinstitute.github.io/ShapeWorks/6.3/use-cases/multistep/incremental_supershapes.html).

The data is provided via data loaders, 1000 image/point pairs were randomly selected for training, 100 for validation, and 100 for testing. The z-dimension is selected to be 5 via PCA on the training particles preserving 95% of the variability. 
