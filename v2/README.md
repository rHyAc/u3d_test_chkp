

## Setup

### 0. Install [Miniconda](https://docs.anaconda.com/free/miniconda/)

### 1. Install dependencies:
```
conda env create -f env.yml
```

*Note1*: env includes PyTorch 1.8.1 with CUDA 11.1

*Note2*: neural-renderer-pytorch may need to be installed manually 

```
git clone https://github.com/daniilidis-group/neural_renderer.git
cd neural_renderer
modify the code according to https://github.com/daniilidis-group/neural_renderer/pull/110/commits/378989dcdae659c7021efda91ea20bf99cedf102
python setup.py install
```

### 2. Download checkpoint.pth from releases page and put it into 'ckp' dir


## Data
put face images in demo/images


## Run
```
python -m demo.demo --input demo/images/ --result demo/results/ --checkpoint ckp/checkpoint.pth
```

