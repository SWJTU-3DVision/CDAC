A Cross-Dimension Annotations Method for 3D Structural Facial Landmark Extraction
====

Background
-------
This repository is for [A Cross-Dimension Annotations Method for 3D Structural Facial Landmark Extraction](https://onlinelibrary.wiley.com/doi/full/10.1111/cgf.13895) (COMPUTER GRAPHICS forum),

by Xun Gong , Ping Chen, Zhemin Zhang, Ke Chen, Yue Xiang and Xin Li from Southwest Jiaotong University.

citation
-------
If you find our work useful in your research, please consider to cite:

    @inproceedings{gong19,
        author={Xun Gong , Ping Chen, Zhemin Zhang, Ke Chen, Yue Xiang and Xin Li},
        title={A Cross-Dimension Annotations Method for 3D Structural Facial Landmark Extraction},
        booktitle={COMPUTER GRAPHICS forum},   
        year={2019},   
    }
Contents
-------
* [Introduction](#Introduction)
* [Requirements](#Requirements)
* [demo](#demo)

Introduction
-------
Recent methods for 2D facial landmark localization perform well on close-to-frontal faces, but 2D landmarks are insufficient
to represent 3D structure of a facial shape. For applications that require better accuracy, such as facial motion capture and
3D shape recovery, 3DA-2D (2D Projections of 3D Facial Annotations) is preferred. Inferring the 3D structure from a single
image is an ill-posed problem whose accuracy and robustness are not always guaranteed. This paper aims to solve accurate 2D
facial landmark localization and the transformation between 2D and 3DA-2D landmarks. One way to increase the accuracy is
to input more precisely annotated facial images. The traditional cascaded regressions cannot effectively handle large or noisy
training data sets. In this paper, we propose a Mini-Batch Cascaded Regressions (MBCR) method that can iteratively train
a robust model from a large data set. Benefiting from the incremental learning strategy and a small learning rate, MBCR is
robust to noise in training data. We also propose a new Cross-Dimension Annotations Conversion (CDAC) method to map
facial landmarks from 2D to 3DA-2D coordinates and vice versa. The experimental results showed that CDAC combined with
MBCR outperforms the-state-of-the-art methods in 3DA-2D facial landmark localization. Moreover, CDAC can run efficiently
at up to 110 fps on a 3.4 GHz-CPU workstation. Thus, CDAC provides a solution to transform existing 2D alignment methods
into 3DA-2D ones without slowing down the speed.
Recent methods for 2D facial landmark localization perform well on close-to-frontal faces, but 2D landmarks are insufficient
to represent 3D structure of a facial shape. For applications that require better accuracy, such as facial motion capture and
3D shape recovery, 3DA-2D (2D Projections of 3D Facial Annotations) is preferred. Inferring the 3D structure from a single
image is an ill-posed problem whose accuracy and robustness are not always guaranteed. This paper aims to solve accurate 2D
facial landmark localization and the transformation between 2D and 3DA-2D landmarks. One way to increase the accuracy is
to input more precisely annotated facial images. The traditional cascaded regressions cannot effectively handle large or noisy
training data sets. In this paper, we propose a Mini-Batch Cascaded Regressions (MBCR) method that can iteratively train
a robust model from a large data set. Benefiting from the incremental learning strategy and a small learning rate, MBCR is
robust to noise in training data. We also propose a new Cross-Dimension Annotations Conversion (CDAC) method to map
facial landmarks from 2D to 3DA-2D coordinates and vice versa. The experimental results showed that CDAC combined with
MBCR outperforms the-state-of-the-art methods in 3DA-2D facial landmark localization. Moreover, CDAC can run efficiently
at up to 110 fps on a 3.4 GHz-CPU workstation. Thus, CDAC provides a solution to transform existing 2D alignment methods
into 3DA-2D ones without slowing down the speed.
Recent methods for 2D facial landmark localization perform well on close-to-frontal faces, but 2D landmarks are insufficient
to represent 3D structure of a facial shape. For applications that require better accuracy, such as facial motion capture and
3D shape recovery, 3DA-2D (2D Projections of 3D Facial Annotations) is preferred. Inferring the 3D structure from a single
image is an ill-posed problem whose accuracy and robustness are not always guaranteed. This paper aims to solve accurate 2D
facial landmark localization and the transformation between 2D and 3DA-2D landmarks. One way to increase the accuracy is
to input more precisely annotated facial images. The traditional cascaded regressions cannot effectively handle large or noisy
training data sets. In this paper, we propose a Mini-Batch Cascaded Regressions (MBCR) method that can iteratively train
a robust model from a large data set. Benefiting from the incremental learning strategy and a small learning rate, MBCR is
robust to noise in training data. We also propose a new Cross-Dimension Annotations Conversion (CDAC) method to map
facial landmarks from 2D to 3DA-2D coordinates and vice versa. The experimental results showed that CDAC combined with
MBCR outperforms the-state-of-the-art methods in 3DA-2D facial landmark localization. Moreover, CDAC can run efficiently
at up to 110 fps on a 3.4 GHz-CPU workstation. Thus, CDAC provides a solution to transform existing 2D alignment methods
into 3DA-2D ones without slowing down the speed.

Requirements
-------
    1.opencv2.4.13
demo 
-------
Baidu Cloud:https://pan.baidu.com/s/1VSCAuMHHopy6Szf6nxHe3w
