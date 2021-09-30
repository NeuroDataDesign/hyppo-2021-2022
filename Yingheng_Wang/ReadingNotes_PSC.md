# Mapping Population-based structural connectomes

## Summary

PSC simultaneously characterizes a large number of white matter bundles within and across different subjects by registering different subjects’ brains based on coarse cortical parcellations, compressing the bundles of each connection, and extracting novel connection weights.

A robust tractography algorithm and streamline post-processing techniques, including **dilation of gray matter regions, streamline cutting, and outlier streamline removal** are applied to improve the robustness of the extracted structural connectomes.

The developed PSC framework can be used to reproducibly extract binary networks, weighted networks and streamline-based brain connectomes.

## Reasons using PSC

The structural connectome consists of grouped white matter (WM) trajectories that connect
different brain regions, representing a comprehensive diagram of neural connections. To date, dMRI is the only noninvasive technique useful for
estimating WM trajectories and water diffusivity along these trajectories in vivo. It has been widely used to quantify WM integrity and WM abnormalities associated with brain disorders.

At the population level, to quantify variations in the diffusion connectomes and local WM changes of healthy and diseased brains, there
are roughly *three broad analytical methods: (i) standard region-based
analysis; (ii) voxel-based analysis, and (iii) tract-specific analysis.* 

**The region-based method** often parcellates the brain into regions of interest (ROIs) that have anatomical meaning and studies the statistical properties of each region. Although it is
convenient to focus on specific regions, it suffers from the difficulty in
identifying meaningful regions in WM, particularly among the long
curved structures common in fiber tracts.

**The voxel-based analysis**
spatially normalizes brain images across subjects and performs statistical analysis at each voxel. One of the most popular voxel-based methods is the Tract-Based Spatial Statistics (TBSS), which is based on the projection of fractional anisotropy (FA) maps of individual subjects onto a common mean FA tract skeleton. The voxel-based methods are limited due to their reliance on existing registration methods that lack the ability to explicitly model the underlying architecture of WM fibers, including the neural systems and circuits affected, in the registration process.

Compared to the region- and voxel-based methods, **tract-specific analysis** provides several desirable outputs. It can visualize specific WM bundles, quantitatively analyze the geometry of WM bundles, and analyze the diffusion properties along WM bundles.  ***One of the most challenging tasks in this approach is to efficiently use the whole-brain tractography data to construct reproducible population-based structural
connectome maps, while effectively accounting for variation across subjects within and between populations.***

**The tract-specific approaches**
can be naturally grouped into two categories: *fiber clustering-based WM analysis* and *parcellation-based connectome
analysis*. Although both method types perform segmentation
of the WM bundles, they have different goals.

(1) The advantage of **fiber clustering-based methods** is that they can use the shape, size and location of streamlines (also referred to as fiber curves
or fiber tracts) to identify anatomically defined WM tracts, and study the WM integrity along these tracts. However, such methods ***heavily depend on
the choice of clustering method and that of the similarity metric for comparing streamlines.*** Also, ***they usually consider only part of the whole-brain fiber curves and may result in the loss of valuable information.***

(2) In contrast, **parcellation-based methods** can utilize the ***whole-brain fiber curves*** and produce an adjacency $V \times V$ matrix $A_i$, where $V$ is the number of ROIs and can vary from tens to hundreds according to the cortical parcellation methods. The $(u, v)$-th element of $A_i$ represents a measure of the strength of connection between regions $u$ and $v$. For a specific pair of ROIs, the most popular connectivity strength is an indicator (range of $0–1$) of whether there is any streamline connecting them so that standard
graph analysis may be applied. However, the use of ***such a binary connectivity matrix leads to an enormous loss of information such that all geometric and diffusivity information along the WM bundles is discarded.***

### Strength of PSC

PSC can utilize the geometric information of
streamlines, including shape, size and location, for a better
parcellation-based connectome analysis. 