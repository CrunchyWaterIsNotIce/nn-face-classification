# NN-Face-Classification 🥸

This project explores the fundamentals of machine learning through the creation of an neural network system from scratch using only *Numpy* and *Pandas* — no external ML frameworks. 

> Initially, I had created a folder on my local device that archived all my research and experiments towards the basics of neuron input and outputs, added biases and impact of weights, activation influence of sigmoids, and finally the importance of forward and backward propagation. Furthermore, understanding **__all__** the math behind backward propagation with the use of partial derivitives and gradients vectors in minimizing the loss.

Though creating this repository allowed me to then experiment with more complex datasets and different activation and loss functions like ReLus and Binary Cross Entropy which helped my neural network work more efficiently.

## Data
The image data collected was generated from cannguyen275's repository [isFace](https://github.com/cannguyen275/isFace/tree/master)'s image data and used to be fed into my scratch neural network. Each image was then greyscaled and condensed into 32x32 before being vector normalized into a .csv dataset. 

> ***The .csv dataset isn't pushed into the repository due to size constraints.***

## Results
Forward: ReLu in hidden → Sigmoid in output

Loss: BSE

| Type | Layer Structure | Epochs | Learning Rate | Accuracy |
| --- | --- | --- | --- | --- |
| 1 Layer | 1024 → 64 → 1 | 50 | 0.0005 | 87.459% |
| 2 Layers, Funneled | 1024 → 128 → 64 → 1 | 50 | 0.0005 | 89.43% |
| 3 Layers, Funneled | 1024 → 256 → 128 → 64 → 1 | 50 | 0.0005 | 90.23% |


The model was also deployed using a webcam input to classify real-time video frames as face or non-face, after live preprocessing. (grayscale, resize, normalization) However, because the image data centered all the faces, face detection is only limited to the position and orienation of the real face captured.
