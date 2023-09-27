With activation functions like ReLU, a network cannot learn information of the derivative of a function. With tanh, softplus, and specially cos and sin, it can.  
Setting the weights to 6 may be better than initializing them randomly. Also setting the bias to 30.
Use ADAM optimizer.
SIRENs can be used to store images and reconstruct them. They also can be used to inpaint images of which we don't know the entirety of the pixels.
[Source](https://arxiv.org/pdf/2006.09661.pdf).


Generating synthetic data with google earth is a good idea if there are no available datasets. 
Height can be obtained from multi-view stereo or encoder-decoder networks if you have a ground-truth, or ImMPS if you don't have it.
Use MSELoss (own), L1, SSIM or Learned Perceptual Image Patch Similarity LPIPS losses.
[Source](https://arxiv.org/pdf/2205.08908.pdf)


