# Network Architecture Componentization

The objective is the design of reusable neural block in PyTorch to be assembled to produce complex network architectures.

The automation of creating deep convolutional generative adversarial networks (DCGAN) by inferring the configuration of generator from the discriminator. We will use the ubiquituous real vs. fake images detection scenario for our GAN model. 

This post does not dwell in details into generative adversarial networks or convolutional networks. It focuses on automating the configuration of some of their components. It is assumed the reader has some basic understanding of convolutional neural networks and Pytorch library.

# The challenge
For those not familiar with GANs..... 
GANs are unsupervised learning models that discover patterns in data and use those patterns to generate new samples (data augmentation) that are almost indistinguishable from the original data. GANs are part of the generative models family along with variational auto-encoders or MLE. The approach reframes the problem as a supervised learning problem using two adversarial networks:
Generator model trained to generate new samples
Discriminator model that attempts to classify the new samples as real (from the original dataset) or fake (generated)
Please refer to the reference section to learn more about generative adversarial networks.

Designing and configuring the generator and discriminator of a generative adversarial networks (GAN) or the encoder and decoder layers of a variational convolutional auto-encoders (VAE) can be a very tedious and repetitive task. 
Actually some of the steps can be fully automated knowing that the generative network of the convolutional GAN for example can be configured as the mirror (or inversion) of the discriminator using a de-convolutional network. The same automation technique applies to the instantiation of a decoder of a VAE given an encoder.

# Functional representation of a simple deep convolutional GAN
Neural component reusability is key to generate a de-convolutional network from a convolutional network. To this purpose we **break down** a neural network into computational blocks.

## Convolutional neural blocks
At the highest level, a generative adversarial network is composed of at least two neural networks: A generator and a discriminator.
These two neural networks can be broken down into neural block or group of PyTorch modules: *hidden layer, batch normalization, regularization, pooling mode and activation function*. Let's consider a discriminator built using a convolutional neural network followed by a fully connected (restricted Boltzmann machine) network. The PyTorch modules associated with any given layer are assembled as a neural block class.

A PyTorch modules of the convolutional neural block are:
- **Conv2d**: Convolutional layer with input, output channels, kernel, stride and padding
- **BatchNorm2d**: Batch normalization module
- **MaxPool2d** Pooling layer
- **ReLu**, **Sigmoid**, ... Activation functions


## Representation of a convolutional neural block
The constructor for the neural block initializes all its parameters and its modules in the proper oder. For the sake of simplicity, regularization elements such as drop-out (bagging of sub-network) is omitted.

```
class ConvNeuralBlock(nn.Module):
  def __init__(self,
      in_channels: int,
      out_channels: int,
      kernel_size: int,
      stride: int,
      padding: int,
      batch_norm: bool,
      max_pooling_kernel: int,
      activation: nn.Module,
      bias: bool,
      is_spectral: bool = False):
    
   super(ConvNeuralBlock, self).__init__()

   Initialize the input and output channels
   self.in_channels = in_channels
   self.out_channels = out_channels
   self.is_spectral = is_spectral
   modules = []
   
   # Create a 2 dimension convolution layer
   conv_module = nn.Conv2d(  # 
       self.in_channels,
       self.out_channels,
       kernel_size=kernel_size,
       stride=stride,
       padding=padding,
       bias=bias)

   # If this is a spectral norm block
   if self.is_spectral:        
     conv_module = nn.utils.spectral_norm(conv_module)
     modules.append(conv_module)
        
   # Batch normalization
   if batch_norm:               
     modules.append(nn.BatchNorm2d(self.out_channels))
     
   # Activation function
   if activation is not None: 
     modules.append(activation)
        
   if max_pooling_kernel > 0:  # Pooling module
     modules.append(nn.MaxPool2d(max_pooling_kernel))
   
   self.modules = tuple(modules)
```

We considering the case of a generative model for images. The first step (1) is to initialize the number of input and output channels, then create the 2-dimension convolution (2), a batch normalization module (3) an activation function (4) and finally a Max  pooling module (5). The spectral norm regularization (6) is optional.
The convolutional neural network is assembled from convolutional and feedback forward neural blocks, in the following build method.

```
`class ConvModel(NeuralModel):
  def __init__(self,                    
       model_id: str,
       input_size: int,  # 1 Number of input and output units
       output_size: int,
       # 2- PyTorch convolutional modules
       conv_model: nn.Sequential,
       dff_model_input_size: int = -1,
       # 3- PyTorch fully connected
       dff_model: nn.Sequential = None):
        
   super(ConvModel, self).__init__(model_id)
   self.input_size = input_size
   self.output_size = output_size
   self.conv_model = conv_model
   self.dff_model_input_size = dff_model_input_size
   self.dff_model = dff_model
   
  @classmethod
  def build(cls,
      model_id: str,
      conv_neural_blocks: list,  
      dff_neural_blocks: list) -> NeuralModel:
            
     # Initialize the input and output size for the convolutional layer
   input_size = conv_neural_blocks[0].in_channels
   output_size = conv_neural_blocks[len(conv_neural_blocks) - 1].out_channels

     # 4 Generate the model from the sequence of conv. neural blocks
   conv_modules = [conv_module for conv_block in conv_neural_blocks
         for conv_module in conv_block.modules]
   conv_model = nn.Sequential(*conv_modules)

     # 6 If a fully connected RBM is included in the model ..
   if dff_neural_blocks is not None and not is_vae:
     dff_modules = [dff_module for dff_block in dff_neural_blocks
        for dff_module in dff_block.modules]
         
     dff_model_input_size = dff_neural_blocks[0].output_size
     dff_model = nn.Sequential(*tuple(dff_modules))
   else:
     dff_model_input_size = -1
     dff_model = None
      
  return cls(
     model_id, 
     conv_dimension, 
     input_size, 
     output_size, 
     conv_model,
     dff_model_input_size, 
     dff_model)
 ```

The default **constructor** (1) initializes the number of input/output channels, the PyTorch modules for the convolutional layers (2) and the fully connected layers (3).
The class method, **build**, instantiate the convolutional model from the convolutional neural blocks and feed forward neural blocks. It initializes the size of input and output layers from the first and last neural blocks (4), generate the PyTorch convolutional modules (5) and fully-connected layers modules (6) from the neural blocks.
Next we build the de-convolutional neural network from the convolutional blocks.

## Inverting a convolutional block
The process to build a GAN is as follow:
1. Specify components (PyTorch modules) for each convolutional layer 
2. Assemble these modules into a convolutional neural block
3. Create a generator and discriminator network by aggregating the blocks
4. Wire the generator and discriminator to product a fully functional GAN
5. The goal is create a builder for generating the de-convolutional network implementing the GAN generator from the convolutional network defined in the previous section. 
6. The first step is to extract the de-convolutional block from an existing convolutional block


## Auto-generation of de-convolutional block from a convolutional block
The default constructor for the neural block of a de-convolutional network defines all the key parameters used in the network except the pooling module (not needed). The following code snippet illustrates the instantiation of a De convolutional neural block using the convolution parameters such as number of input, output channels, kernel size, stride and passing, batch normalization and activation function. 

```
class DeConvNeuralBlock(nn.Module):
  def __init__(self,
       in_channels: int,
       out_channels: int,
       kernel_size: int,
       stride: int,
       padding: int,
       batch_norm: bool,
       activation: nn.Module,
       bias: bool) -> object:
    super(DeConvNeuralBlock, self).__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels
    modules = []
             
    # Two dimension de-convolution layer
    de_conv = nn.ConvTranspose2d(
      self.in_channels,
      self.out_channels,
      kernel_size=kernel_size,
      stride=stride,
      padding=padding,
      bias=bias)
   # Add the deconvolution block
   modules.append(de_conv)

   # Add the batch normalization, if defined
   if batch_norm:         
     modules.append(nn.BatchNorm2d(self.out_channels))
   # Add activation
   modules.append(activation)
   self.modules = modules
```
*Note* that the de-convolution block does have any pooling capabilities

The class method, **auto_build**, takes a convolutional neural block, number of input and output channels and an optional activation function to generate a de-convolutional neural block of type **DeConvNeuralBlock**. The number of input and output channels in the output deconvolution layer is computed in the private method __resize

```
@classmethod
def auto_build(cls,
    conv_block: ConvNeuralBlock,
    in_channels: int,
    out_channels: int = None,
    activation: nn.Module = None) -> nn.Module:
    
  # Extract the parameters of the source convolutional block
  kernel_size, stride, padding, batch_norm, activation = \
     DeConvNeuralBlock.__resize(conv_block, activation)

  # Override the number of input_tensor channels 
  # for this block if defined
  next_block_in_channels = in_channels 
    if in_channels is not None \
    else conv_block.out_channels

  # Override the number of output-channels for 
  # this block if specified
  next_block_out_channels = out_channels 
    if out_channels is not None \
    else conv_block.in_channels
    
  return cls(
        conv_block.conv_dimension,
        next_block_in_channels,
        next_block_out_channels,
        kernel_size,
        stride,
        padding,
        batch_norm,
        activation,
        False)
```

## Sizing de-convolutional layers
The next task consists of computing the size of the component of the de-convolutional block from the original convolutional block. 

```
@staticmethod
def __resize(
  conv_block: ConvNeuralBlock,
  updated_activation: nn.Module) -> (int, int, int, bool, nn.Module):
  conv_modules = list(conv_block.modules)
    
  # 1- Extract the various components of the 
  # convolutional neural block
  _, batch_norm, activation = DeConvNeuralBlock.__de_conv_modules(conv_modules)
  # 2- override the activation function for the 
  # output layer, if necessary
  if updated_activation is not None:
    activation = updated_activation
    
    # 3- Compute the parameters for the de-convolutional 
    # layer, from the conv. block
    kernel_size, _ = conv_modules[0].kernel_size
    stride, _ = conv_modules[0].stride
    padding = conv_modules[0].padding

 return kernel_size, stride, padding, batch_norm, activation
```

The **__resize** method extracts the PyTorch modules for the de-convolutional layers from the original convolutional block (1), adds the activation function to the block (2) and finally initialize the parameters of the de-convolutional (3).

The helper method,  **__de_conf_modules**, extracts the PyTorch modules related to the convolutional layer, batch normalization module and activation function for the de-convolution from the PyTorch modules of the convolution.

```
@staticmethod
def __de_conv_modules(conv_modules: list) -> \
        (torch.nn.Module, torch.nn.Module, torch.nn.Module):
  activation_function = None
  deconv_layer = None
  batch_norm_module = None

  # 4- Extract the PyTorch de-convolutional modules 
  # from the convolutional ones
  for conv_module in conv_modules:
    if DeConvNeuralBlock.__is_conv(conv_module):
       deconv_layer = conv_module
    elif DeConvNeuralBlock.__is_batch_norm(conv_module):
       batch_norm_moduled = conv_module
    elif DeConvNeuralBlock.__is_activation(conv_module):
       activation_function = conv_module
  return deconv_layer, batch_norm_module, activation_function
```


## De-convolutional layers
As expected, the formula to computed the size of the output of a de-convolutional layer is the mirror image of the formula for the output size of the convolutional layer.

## Assembling the de-convolutional network
Finally, de-convolutional model, of type **DeConvModel**  is created using the sequence of PyTorch module, **de_conv_model**. Once again, the default constructor (1) initializes the size of the input layer (2) and output layer (3) and load the PyTorch modules, de_conv_modules, for all de-convolutional layers.

```
class DeConvModel(NeuralModel, ConvSizeParams):
  def __init__(self,            # 1 - Default constructor
           model_id: str,
           input_size: int,     # 2 - Size first layer
           output_size: int,    # 3 - Size output layer
           de_conv_modules: torch.nn.Sequential):
    super(DeConvModel, self).__init__(model_id)
    self.input_size = input_size
    self.output_size = output_size
    self.de_conv_modules = de_conv_modules


  @classmethod
  def build(cls,
      model_id: str,
      conv_neural_blocks: list,  # 4- Input to the builder
      in_channels: int,
      out_channels: int = None,
      last_block_activation: torch.nn.Module = None) -> NeuralModel:
    
    de_conv_neural_blocks = []

    # 5- Need to reverse the order of convolutional neural blocks
    list.reverse(conv_neural_blocks)

    # 6- Traverse the list of convolutional neural blocks
    for idx in range(len(conv_neural_blocks)):
       conv_neural_block = conv_neural_blocks[idx]
       new_in_channels = None
       activation = None
       last_out_channels = None

        # 7- Update num. input channels for the first 
        # de-convolutional layer
       if idx == 0:
           new_in_channels = in_channels
        
        # 8- Defined, if necessary the activation 
        # function for the last layer
       elif idx == len(conv_neural_blocks) - 1:
         if last_block_activation is not None:
           activation = last_block_activation
         if out_channels is not None:
          last_out_channels = out_channels

        # 9- Apply transposition to the convolutional block
      de_conv_neural_block = DeConvNeuralBlock.auto_build(
           conv_neural_block,
           new_in_channels,
           last_out_channels,
            activation)
      de_conv_neural_blocks.append(de_conv_neural_block)
        
       # 10- Instantiate the Deconvolutional network 
       # from its neural blocks
   de_conv_model = DeConvModel.assemble(
       model_id, 
       de_conv_neural_blocks)
     
   del de_conv_neural_blocks
   return de_conv_model
```

The alternate constructor, build, creates and configures the de-convolutional model from the convolutional blocks **conv_neural_blocks** (4). 
The order of the de-convolutional layers requires the list of convolutional blocks to be reversed (5).  For each block of the convolutional network (6), the method updates the number of input channels from the number of input channels of the first layer (7).
The method updates the activation function for the output layer (8) and weaves the de-convolutional blocks (9)
Finally, the de-convolutional neural network is assembled from these blocks (10).

```
@classmethod
def assemble(cls, model_id: str, de_conv_neural_blocks: list):
    input_size = de_conv_neural_blocks[0].in_channels
    output_size = de_conv_neural_blocks[len(de_conv_neural_blocks) - 1].out_channels
   
   # 11- Generate the PyTorch convolutional modules used by the default constructor
    conv_modules = tuple([conv_module for conv_block in de_conv_neural_blocks
                          for conv_module in conv_block.modules 
                          if conv_module is not None])
    de_conv_model = torch.nn.Sequential(*conv_modules)
    return cls(model_id, input_size, output_size, de_conv_model)
```

The **assemble** method constructs the final de-convolutional neural network from the blocks **de_conv_neural_blocks** by aggregating the PyTorch modules associated with each block (11).

# Environment
- Python 3.8
- PyTorch 1.7.2

 #References
- A Gentle Introduction to Generative Adversarial Networks https://machinelearningmastery.com/what-are-generative-adversarial-networks-gans/
- Deep learning Chap 9 Convolutional networks - I. Goodfellow, Y. Bengio, A. Courville - 2017 - MIT Press Cambridge MA 
- PyTorch www.pytorch.org
- Tutorial: DCGAN in PyTorch

