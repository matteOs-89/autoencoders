# Autoencoders
Autoencoders are powerful deep learning algorithms which have close similarities to Generative Adversarial Networks (GAN) in terms of implementation and use, However, unlike GAN, Autoencoders use one neural network model to achieve the desired result whereas GAN uses two. Autoencoders are classed as unsupervised learning as data labels are not required.

A complete Autoencoder network works by compressing its input features and passing it to its a reduced dimension/latent space called “Code” and then replicating the input feature data and feeding it to its output layer.

 Autoencoder architecture has three functional components that work together to produce the desired output result. These components are:

•	Encoder: An Encoder takes in the input data, then reads and extracts the important features of the data to compress and feed it to Code.

•	Code: The latent space where all important reduced features of the input data are stored and ready to be fed to the Decoder.

•	Decoder: The Decoder’s job is to reconstruct the data collected from the Code in a structured manner for output. In some cases, it generates new data.

Some reasons why Autoencoders are used include:

•	Denoising of images;
•	Dimensionality reduction;
•	Data compression;
•	Feature Extraction;
•	Image generation; and 
•	Image colourization

## Types of Autoencoders:

•	“Vanilla” Autoencoder:  The Neural network architecture used for this model is constructed with fully connected feedforward Encoder and Decoder components where a Code layer sits between them.

<img width="305" alt="Screenshot 2023-01-04 at 23 03 08" src="https://user-images.githubusercontent.com/111536571/210670271-9f33b00d-a8f6-4f19-84b6-3a9a70b098b5.png">


•	Variational Autoencoder: The problem with the “Vanilla” Autoencoders is that the latent space that the input data is applied to, is mostly discrete (not continuous) this also makes back-propagation very difficult, this problem results in difficult interpolation between data points when trying to generate new data or images. To prevent this, a new form of Autoencoder was created called Variational Autoencoder. Variational Autoencoders are generative models, they generate new data close to their input data. The architecture of Variational Autoencoder is very similar to that of the Vanilla Autoencoder stated above, however with Variational Autoencoder the Encoder input is fed into the latent space as a distribution. The Encoder distribution is regularised during training providing continuous distribution, to ensure that its latent space has good properties making it easier for better interpolation of the data points, which helps us to generate new data. 

Reparameterization trick is used as part of the Variational Autoencoder, and this allows the model to generate new data from a normal distribution. As the distribution is continuous, this makes back-propagation possible during training. This also allows the model to learn better and minimise overfitting.

Another difference between Variational Autoencoders and Autoencoders is the loss function. In Variational Autoencoder, KL Divergence is added to its loss function to measure the similarities of its distributions, and during learning it aims to minimise the difference to ensure that samples taken are from the same distribution.

<img width="193" alt="Screenshot 2023-01-04 at 23 09 06" src="https://user-images.githubusercontent.com/111536571/210670239-d237c714-28b6-40c4-9155-0cf0370fb5ea.png">



•	Convolutional Autoencoders: The procedures used in this type of Autoencoders are like that of the other two stated above, however the structure of this model is built using convolutional neural networks for its Encoder component and transposed convolution for the Decoder. This type of Autoencoder is used for image generation and image segmentation.

<img width="195" alt="Screenshot 2023-01-04 at 23 11 07" src="https://user-images.githubusercontent.com/111536571/210670193-eb95f446-1644-4cb4-b81a-6f7f1083895b.png">



## Dataset

The MNIST dataset is a famous dataset that is preinstalled in Pytorch and was created in 1998. The dataset is made up of thousands of samples of handwritten numbers (from 1 to 9) from high school students and employees of the United States Census Bureau.

The MNIST dataset will be used to experiment the performance of the above models in order to minimise researcher bias. 


### IMAGES GENERATED USING AUTOENCODER 

Top two rows are original images
Bottom two rows are test reconstructed images

#### 15 Epochs
<img width="152" alt="Screenshot 2023-01-04 at 23 26 26" src="https://user-images.githubusercontent.com/111536571/210668966-c07ba1f8-6ed6-4241-87c1-d55ad32b40c9.png">

#### 20 Epochs
<img width="152" alt="Screenshot 2023-01-04 at 23 29 09" src="https://user-images.githubusercontent.com/111536571/210669282-dc513322-e451-4328-9cc3-6ae4f9ac9b9a.png">


#### AUTOENCODER TRAINING LOSS PLOT

<img width="426" alt="Screenshot 2023-01-04 at 03 14 35" src="https://user-images.githubusercontent.com/111536571/210667864-06ccff55-8c34-45cd-9eba-6341a3d73d9a.png">

The training was stopped after 30 epochs. The plot shows that the network’s gradient descent had not reached its global minimum, and perhaps with more training epochs a better performance could be reached.

### IMAGES GENERATED USING VARIATIONAL AUTOENCODER 

Top two rows are original images
Bottom two rows are test reconstructed images

#### 10 Epochs
<img width="156" alt="Screenshot 2023-01-04 at 23 32 20" src="https://user-images.githubusercontent.com/111536571/210669641-00e2c8ec-d87a-47d2-8660-a8ae1fa0afbe.png">

#### 19 Epochs

<img width="156" alt="Screenshot 2023-01-04 at 23 34 37" src="https://user-images.githubusercontent.com/111536571/210669869-9fa0b8a9-5a68-4bc9-b482-3a677f39a833.png">


### IMAGES GENERATED USING CONVOLUTIONAL AUTOENCODER 
Top two rows are original images
Middle two rows are images reconstructed during training
Bottom two rows are test reconstructed images 

 #### 5 epochs
<img width="154" alt="Screenshot 2023-01-04 at 23 18 42" src="https://user-images.githubusercontent.com/111536571/210668190-49064310-006c-4c3d-bba2-3e08e809c4c1.png">
 
 #### 10 Epochs
 
 <img width="154" alt="Screenshot 2023-01-04 at 23 22 59" src="https://user-images.githubusercontent.com/111536571/210668615-38fa3399-fc0b-459f-ab68-fcf927304500.png">


#### CONVOLUTIONAL TRAINING LOSS PLOT

<img width="538" alt="Screenshot 2023-01-04 at 17 02 05" src="https://user-images.githubusercontent.com/111536571/210667993-4c6e7715-809b-4e0c-afbe-4019ac83b744.png">

The plot shows that the Convolutional Autoencoder learned a lot quicker and with less training epochs (10) than the Autoencoder model and minimising the loss much faster, result to better reconstructed images.

By Eseosa Jesuorobo
