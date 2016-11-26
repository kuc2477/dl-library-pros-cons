# dl-library-pros-cons
Summarized Pros and Cons of popular Deep Learning libraries and simple decision heuristic from *[Lecture 12: Deep Learning libraries](https://www.youtube.com/watch?v=XgFlBsl0Lq4&index=11&list=PLlJy-eBtNFt6EuMxFYRiNRS07MCWN5UIA)* of [Stanford 2016 Winter CS231n](http://cs231n.github.io/).


# Caffe

### Features
- Blob, Layer, Net, Solver are 4 main classes
- Uses Protocol Buffers for defining all types of Caffe instances in .proto files.
- Use python interface if you don't want just using pretrained models.

### Pros / Cons
- (+) Good for feed forward networks
- (+) Good for finetuning existing models
- (+) Train models without any code
- (+) Python interface for doing something extra (not just using pretrained models)
- (-) Need to write C++ / CUDA for new GPU layers
- (-) No Model parallelism
- (-) Not good for RNN
- (-) Cumbersome for big networks (GoogLeNet, ResNet)

### Workflow
1. Converting data: LMDB(recommended) or HDF5
2. Data layers: ImageDataLayer, WindowDataLayer, HDF5Layer
3. Define net: Define network in .proto file.
4. Define net (finetuning): Change layer name if you want to finetune from the very scratch.
5. Define solver
6. Train
7. Or just use model zoo


# Torch

### Features
- Used in DeepMind and Facebook, Tweeter
- Only Module is a main class for both network and layers
- Unlike numpy, GPU is just a datatype. Everything is plain tensors.
- Use nn, cunn, nngraph, optim package
- Writing new Module is easy.
- Tons of fresh Modules and Loss Functions updated quite often.
- Use container Module to compose your networks.
- Use nngraph package to build complex Modules that combine their inputs in complex ways
- Use loadcaffe to load pretrained Caffe model (only works for certain types of networks) 
- Lots of other useful packages: torch,cudnn, torch-hdf5, lua-cjson, cltorch, clnn, torch-autograd, fbcunn (FFT conv, Multi GPU, Data Parallelism + Model Parallelism)

### Pros / Cons
- (-) Lua
- (-) Less plug-and-play than Caffe
- (+) Lots of modular pices that are easy to combine
- (+) Easy to write your own Module type and run on GPU (Everything is just tensors)
- (+) Data parallelism + Model parallelism
- (+) Most of libraries are written in pure Lua.
- (+) Lots of pretrained models
- (-) Not great for RNN

### Workflow
1. Preprocess data: Use Python script to dump data to HDF5
2. Train model in Lua / Torch: Read and train from HDF5 file, save trained model to your disk.
3. Use trained model. Often with evaluation script.


# Theano

### Features
- Theano function, which is in effect a computational graph, compiles down to GPU code or sometimes even down to native code. It also optimizes your computational graph in behind the scenes.
- Symbolic differentiation. You can go crazy with this.
- Shared variables. Embeds parameters into your computational graph so that everything can be done in GPU.
- Conditionals and loops into your computational graph.
- Experimental Model parallelism and Data parallelism (platoon)
- High level wrappers: Lasagne and Keras
- Pretrained models: Lasagne model zoo, AlexNet

### Pros / Cons
- (+) Python + numpy
- (+) Computational graph is nice abstraction
- (+) RNN fits nicely with the computational graph
- (-) Somewhat low-level
- (+) High level wrappers ease the pain
- (-) Unhelpful error messages
- (-) Model parallelism is experimental
- (-) Large models can have long compile times
- (-) Much magic done behind the scenes than Torch
- (-) Patchy support for pretrained models.


# TensorFlow

### Features
- Written from the ground up by engineers, not by academic research labs.
- Simillar to Theano: All about computational graph
- Easy visualization: TensorBoard
- Data parallelism + Model Parallelism
- Distributed training
- Few pretrained models

### Pros / Cons
- (+) Python + numpy
- (+) Computational graph like Theano.
- (+) RNN fits nicely with the computational graph
- (+) Much faster compile time than Theano
- (+) More convenient than Theano
- (+) TensorBoard for visualization
- (+) Data parallelism + Model parallelism
- (+) Distributed training
- (-) Much magic done behind the scenes than Torch
- (-) Not many pretrained models

# Which One To Use?
- Feature extraction & Finetuning pretrained models: **Caffe**
- Complex use of pretrained models: (**Lasagne** / **Keras**) or **Torch**
- Writing your own layers: **Torch**
- RNN: **Theano** or **TensorFlow**
- Huge model with Distributed / Parallel training: **TensorFlow**
