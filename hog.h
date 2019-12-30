#pragma once
/** @file hog.h
** @brief Histogram of Oriented Gradients (@ref hog)
** @author Andrea Vedaldi
**/

/*
Copyright (C) 2007-12 Andrea Vedaldi and Brian Fulkerson.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#ifndef VL_HOG_H
#define VL_HOG_H
#include <string.h>
#include <assert.h>
#include <stdlib.h>
#include <math.h>

/* The following stuff was copied over from other VLFeat
files, to make this file self-sustained: */

typedef unsigned long long vl_size;
typedef int vl_bool;
typedef long long vl_index;
typedef unsigned long long vl_uindex;
//#define VL_EXPORT extern "C"
#define VL_EXPORT
#define VL_TRUE 1
#define VL_FALSE 0
#define VL_PI 3.141592653589793
#define VL_INLINE static //__inline

/** @brief Compute the minimum between two values
** @param x value
** @param y value
** @return the minimum of @a x and @a y.
**/
#define VL_MIN(x,y) (((x)<(y))?(x):(y))

/** @brief Compute the maximum between two values
** @param x value.
** @param y value.
** @return the maximum of @a x and @a y.
**/
#define VL_MAX(x,y) (((x)>(y))?(x):(y))

/** @brief Floor and convert to integer
** @param x argument.
** @return Similar to @c (int) floor(x)
**/
VL_INLINE long int
vl_floor_f(float x)
{
	long int xi = (long int)x;
	if (x >= 0 || (float)xi == x) return xi;
	else return xi - 1;
}

/** @brief Round
** @param x argument.
** @return @c lround(x)
** This function is either the same or similar to C99 @c lround().
**/
VL_INLINE long int
vl_round_d(double x)
{
	return lround(x); // Note: stripped some optimization/ifdef logic from it
}

/* End copied stuff */

enum VlHogVariant_ { VlHogVariantDalalTriggs, VlHogVariantUoctti };

typedef enum VlHogVariant_ VlHogVariant;

struct VlHog_
{
	VlHogVariant variant;
	vl_size dimension;
	vl_size numOrientations;
	vl_bool transposed;
	vl_bool useBilinearOrientationAssigment;

	/* left-right flip permutation */
	vl_index * permutation;

	/* glyphs */
	float * glyphs;
	vl_size glyphSize;

	/* helper vectors */
	float * orientationX;
	float * orientationY;

	/* buffers */
	float * hog;
	float * hogNorm;
	vl_size hogWidth;
	vl_size hogHeight;
};

typedef struct VlHog_ VlHog;

VL_EXPORT VlHog * vl_hog_new(VlHogVariant variant, vl_size numOrientations, vl_bool transposed);
VL_EXPORT void vl_hog_delete(VlHog * self);
VL_EXPORT void vl_hog_process(VlHog * self,
	float * features,
	float const * image,
	vl_size width, vl_size height, vl_size numChannels,
	vl_size cellSize);

VL_EXPORT void vl_hog_put_image(VlHog * self,
	float const * image,
	vl_size width, vl_size height, vl_size numChannels,
	vl_size cellSize);

VL_EXPORT void vl_hog_put_polar_field(VlHog * self,
	float const * modulus,
	float const * angle,
	vl_bool directed,
	vl_size width, vl_size height, vl_size cellSize);

VL_EXPORT void vl_hog_extract(VlHog * self, float * features);
VL_EXPORT vl_size vl_hog_get_height(VlHog * self);
VL_EXPORT vl_size vl_hog_get_width(VlHog * self);


VL_EXPORT void vl_hog_render(VlHog const * self,
	float * image,
	float const * features,
	vl_size width,
	vl_size height);

VL_EXPORT vl_size vl_hog_get_dimension(VlHog const * self);
VL_EXPORT vl_index const * vl_hog_get_permutation(VlHog const * self);
VL_EXPORT vl_size vl_hog_get_glyph_size(VlHog const * self);

VL_EXPORT vl_bool vl_hog_get_use_bilinear_orientation_assignments(VlHog const * self);
VL_EXPORT void vl_hog_set_use_bilinear_orientation_assignments(VlHog * self, vl_bool x);


/** @file hog.c
** @brief Histogram of Oriented Gradients (HOG) - Definition
** @author Andrea Vedaldi
**/

/*
Copyright (C) 2007-12 Andrea Vedaldi and Brian Fulkerson.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/


/**

<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->
@page hog Histogram of Oriented Gradients (HOG) features
@author Andrea Vedaldi
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->

@ref hog.h implements the Histogram of Oriented Gradients (HOG) features
in the variants of Dalal Triggs @cite{dalal05histograms} and of UOCTTI
@cite{felzenszwalb09object}. Applications include object detection
and deformable object detection.

- @ref hog-overview
- @ref hog-tech

<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->
@section hog-overview Overview
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->

HOG is a standard image feature used, among others, in object detection
and deformable object detection. It decomposes the image into square cells
of a given size (typically eight pixels), compute a histogram of oriented
gradient in each cell (similar to @ref sift), and then renormalizes
the cells by looking into adjacent blocks.

VLFeat implements two HOG variants: the original one of Dalal-Triggs
@cite{dalal05histograms} and the one proposed in Felzenszwalb et al.
@cite{felzenszwalb09object}.

In order to use HOG, start by creating a new HOG object, set the desired
parameters, pass a (color or grayscale) image, and read off the results.

@code
VlHog * hog = vl_hog_new(VlHogVariantDalalTriggs, numOrientations, VL_FALSE) ;
vl_hog_put_image(hog, image, height, width, numChannels, cellSize) ;
hogWidth = vl_hog_get_width(hog) ;
hogHeight = vl_hog_get_height(hog) ;
hogDimenison = vl_hog_get_dimension(hog) ;
hogArray = vl_malloc(hogWidth*hogHeight*hogDimension*sizeof(float)) ;
vl_hog_extract(hog, hogArray) ;
vl_hog_delete(hog) ;
@endcode

HOG is a feature array of the dimension returned by ::vl_hog_get_width,
::vl_hog_get_height, with each feature (histogram) having
dimension ::vl_hog_get_dimension. The array is stored in row major order,
with the slowest varying dimension beying the dimension indexing the histogram
elements.

The number of entries in the histogram as well as their meaning depends
on the HOG variant and is detailed later. However, it is usually
unnecessary to know such details. @ref hog.h provides support for
creating an inconic representation of a HOG feature array:

@code
glyphSize = vl_hog_get_glyph_size(hog) ;
imageHeight = glyphSize * hogArrayHeight ;
imageWidth = glyphSize * hogArrayWidth ;
image = vl_malloc(sizeof(float)*imageWidth*imageHeight) ;
vl_hog_render(hog, image, hogArray) ;
@endcode

It is often convenient to mirror HOG features from left to right. This
can be obtained by mirroring an array of HOG cells, but the content
of each cell must also be rearranged. This can be done by
the permutation obtaiend by ::vl_hog_get_permutation.

Furthermore, @ref hog.h suppots computing HOG features not from
images but from vector fields ::vl_

<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->
@section hog-tech Technical details
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->

HOG divdes the input image into square cells of size @c cellSize,
fitting as many cells as possible, filling the image domain from
the upper-left corner down to the right one. For each row and column,
the last cell is at least half contained in the image.
More precisely, the number of cells obtained in this manner is:

@code
hogWidth = (width + cellSize/2) / cellSize ;
hogHeight = (height + cellSize/2) / cellSize ;
@endcode

Then the image gradient @f$ \nabla \ell(x,y) @f$
is computed by using central difference (for colour image
the channel with the largest gradient at that pixel is used).
The gradient @f$ \nabla \ell(x,y) @f$ is assigned to one of @c 2*numOrientations orientation in the
range @f$ [0,2\pi) @f$ (see @ref hog-conventions for details).
Contributions are then accumulated by using bilinear interpolation
to four neigbhour cells, as in @ref sift.
This results in an histogram  @f$h_d@f$ of dimension
2*numOrientations, called of @e directed orientations
since it accounts for the direction as well as the orientation
of the gradient. A second histogram @f$h_u@f$ of undirected orientations
of half the size is obtained by folding @f$ h_d @f$ into two.

Let a block of cell be a @f$ 2\times 2 @f$ sub-array of cells.
Let the norm of a block be the @f$ l^2 @f$ norm of the stacking of the
respective unoriented histogram. Given a HOG cell, four normalisation
factors are then obtained as the inverse of the norm of the four
blocks that contain the cell.

For the Dalal-Triggs variant, each histogram @f$ h_d @f$ is copied
four times, normalised using the four different normalisation factors,
the four vectors are stacked, saturated at 0.2, and finally stored as the descriptor
of the cell. This results in a @c numOrientations * 4 dimensional
cell descriptor. Blocks are visited from left to right and top to bottom
when forming the final descriptor.

For the UOCCTI descriptor, the same is done for both the undirected
as well as the directed orientation histograms. This would yield
a dimension of @c 4*(2+1)*numOrientations elements, but the resulting
vector is projected down to @c (2+1)*numOrientations elements
by averaging corresponding histogram dimensions. This was shown to
be an algebraic approximation of PCA for descriptors computed on natural
images.

In addition, for the UOCTTI variant the l1 norm of each of the
four l2 normalised undirected histograms is computed and stored
as additional four dimensions, for a total of
@c 4+3*numOrientations dimensions.

<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->
@subsection hog-conventions Conventions
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->

The orientation of a gradient is expressed as the angle it forms with the
horizontal axis of the image. Angles are measured clock-wise (as the vertical
image axis points downards), and the null angle corresponds to
an horizontal vector pointing right. The quantized directed
orientations are @f$ \mathrm{k} \pi / \mathrm{numOrientations} @f$, where
@c k is an index that varies in the ingeger
range @f$ \{0, \dots, 2\mathrm{numOrientations} - 1\} @f$.

Note that the orientations capture the orientation of the gradeint;
image edges would be oriented at 90 degrees from these.

**/

/* ---------------------------------------------------------------- */
/** @brief Create a new HOG object
** @param variant HOG descriptor variant.
** @param numOrientations number of distinguished orientations.
** @param transposed wether images are transposed (column major).
** @return the new HOG object.
**
** The function creates a new HOG object to extract descriptors of
** the prescribed @c variant. The angular resolution is set by
** @a numOrientations, which specifies the number of <em>undirected</em>
** orientations. The object can work with column major images
** by setting @a transposed to true.
**/

//extern "C" {
//#include "hog.c"
//}

/* VL_HOG_H */
#endif