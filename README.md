# Demo for Volume Raycasting using ModernGL

A simple Volume Raycasting demo using PyQT5 and ModernGL. The demo requires OpenGL >=4.1 and has been tested on OSX (planned Linux/Win)

Author: Ulrich Eck https://github.com/ulricheck

Code for OpenGL Camera Control copied from: https://github.com/pyqtgraph/pyqtgraph/blob/develop/pyqtgraph/opengl/GLViewWidget.py

Code and example data for Volume Raycasting copied from: https://github.com/toolchainX/Volume_Rendering_Using_GLSL

## Dependencies

- Python >=3.5
- PyQt5 (>=5.6)
- PyOpenGL
- numpy
- ModernGL wrapper [docs](https://moderngl.readthedocs.io/)
- Pyrr Math library [docs](http://pyrr.readthedocs.io/en/latest/info_contributing.html)


## Installation

```
git checkout https://github.com/ulricheck/ModernGL-Volume-Raycasting-Example.git
cd ModernGL-Volume-Raycasting-Example
pip3 install -r requirements.txt
```

## Running the demo

```
python3 volume_raycasting_example.py
```

This will open an opengl window


## Further Reading

- [A simple and flexible volume rendering framework for graphics-hardware-based raycasting](http://dl.acm.org/citation.cfm?id=2386498)
- [Acceleration Techniques for GPU-based Volume Rendering](http://dl.acm.org/citation.cfm?id=1081432.1081482})
- [Real-time Volume Rendering for High Quality Visualization in Augmented Reality](http://far.in.tum.de/pub/kutter2008amiarcs/)