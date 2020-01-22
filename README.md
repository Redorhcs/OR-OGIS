# Choosing an Optimal Route using Fairfax OpenData
R. T. Schroder & N. Van Nostrand
## Introduction

> What's the fastest way to get from A to B in Fairfax County? What if we included point C?"

Routing is a critical service in today's world. People are always looking to get from A to B in the fastest way possible. Our project looks to explore how routing works in the digital age, and see if hand calculations can still hold up to the machine.

## Code Samples

> You've gotten their attention in the introduction, now show a few code examples. So they get a visualization and as a bonus, make them copy/paste friendly.
### Get the closest nodes by using Euclidean Distance
```
pos0_i = np.argmin(
    np.sum((nodes[:, ::-1] - pos0)**2, axis=1))
pos1_i = np.argmin(
    np.sum((nodes[:, ::-1] - pos1)**2, axis=1))
```
### Get shortest path using Dijkstra's algorthim
```
path = nx.shortest_path(
    sg,
    source=tuple(nodes[pos0_i]),
    target=tuple(nodes[pos1_i]),
    weight='distance')
len(path)
```

## Installation

1. [Install Git](http://git-scm.com/downloads)
2. [Download and install Anaconda (2019.3)](https://anaconda.com/download)
3. Open Terminal
4. Create the Cookbook conda environment `conda env create -f environment.yml`
5. Activate the environment `activate cookbook`
6. Launch Jupyter `jupyter notebook`
