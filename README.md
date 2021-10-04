# [Multiple Pairwise Ranking Networks for Personalized Video Summarization](https://www.yongliangyang.net/docs/multiRanker_iccv21.pdf)

This repository provides PyTorch implementations for Multi-ranker paper published in ICCV 2021.

This code is based on [DR-DSN](https://github.com/KaiyangZhou/pytorch-vsumm-reinforce) and [VASNet](https://github.com/ok1zjf/VASNet) implementations.

## Pairwise Ranking Model for Global Summarization (Standard ranker)

Standard ranker learns a ranking function that associates high ranking scores to important video segments so that a summary can be built by selecting the top-ranked segments.

<p align="center">
<img src="docs/ranker.jpg" width="300" />
</p>

## Multiple Pairwise Ranking Model for Personalized Summarization (Multi-ranker)

Given the number of preferences <img src="https://latex.codecogs.com/svg.latex?\large&space;P" title="\large P" />, Multi-ranker learns a set of <img src="https://latex.codecogs.com/svg.latex?\large&space;P" title="\large P" /> sub-rankers that are jointly trained so the local summaries conform with the preferences and the global summary max-aggregates the sub-rankers' scores. 

<p align="center">
<img src="docs/multi_ranker.jpg" width="500" />
</p>

## Datasets

### [TVSum](https://github.com/yalesong/tvsum)

TVSum dataset is a collection of 50 YouTube videos grouped into 10 categories. Each video is split into a set of 2 second-long shots. 20 users are asked to rate how important each shot is, compared to other shots from the same video in order to build 20 reference summaries. The GT summary for each video is defined as the mean of the corresponding 20 reference summaries.

### SumMe

### FineGym

## Prerequisites

## Getting Started

### Installation

### Standard ranker training

### Standard ranker evaluation

### Multi-ranker training

### Multi-ranker evaluation

### Plots & Results

## Citation

## Poster and Supplementary Material
