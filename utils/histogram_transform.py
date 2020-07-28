# Torch histogram transform taken from the pull request:
# https://github.com/pytorch/vision/pull/796/commits/15429dfdc48e3299632ed1128b30eb4fa319e7e7


import numpy as np
import torch
import torchvision


class HistogramTransform(object):
    """
    Transforms the distribution of the input tensor to match that
    of the list of template histograms corresponding to each channel.
    
    A template historgram must be set initially. 
    Args:
        tensor (numpy.ndarray):
            Image to transform; the histogram is computed over the flattened
            array
        noise_range (float): Default is 0. A uniform noise ranged between 
            (-noise_range, +noise_range) will be added to pixels randomly.
    Returns:
        histogram transformed tensor: 
            The output tensor type matches the input, either numpy.ndarray or torch.Tensor.
    """


    def __init__(self, template_histograms):
        '''
        Args:
            template_histograms: 
                A list of template histograms. 
                Each template histogram must consist of the tuple 
                (counts (numpy.ndarray), bins (numpy.ndarray)).
                template_histograms is a list of numpy.histogram outputs,
                each corresponding to each channel of the input tensor to be transformed.
                If 1 channel, still feed as a list, i.e. [(counts, bin)].
                Example:
                Assuming img is made by ToTensor(some pil image) and has 3 (RGB) channels, one can get the histogram as such:
                histR = np.histogram(img[0].numpy().ravel(), bins = 256, range = [0, 1])
                histG = np.histogram(img[1].numpy().ravel(), bins = 256, range = [0, 1])
                histB = np.histogram(img[2].numpy().ravel(), bins = 256, range = [0, 1])
        '''

        self.template_histograms = template_histograms
        self.num_channels = len(template_histograms)


    def __call__(self, tensor, noise_range = 0, dtype = torch.float32):
        """
        Transforms the distribution of the input tensor to match that
        of the template histogram. If a list of histograms is provided
        and it maches the number of channels of the input tensor, each 
        channel will be transformed with the corresponding histogram. 
        
        This funciton utilises histogram_tranform_1D for an easier user interface. 
        
        Args:
            tensor (numpy.ndarray):
                Image to transform; the histogram is computed over the 
                flattened array for each channel.
            noise_range (float): Default is 0. A uniform noise ranged between 
                (-noise_range, +noise_range) will be added to pixels randomly.
        Returns:
            histogram transformed tensor: 
                The output tensor type matches the input, either numpy.ndarray or torch.Tensor.
        """
        tensorType       = type(tensor)
        tensor           = np.asanyarray(tensor)

        channels = []
        for c, templateHisto in enumerate(self.template_histograms):
            channels.append(self.histogram_transform_1D(tensor[c], templateHisto, noise_range = noise_range))

        transformed_tensor = np.asanyarray(channels)

        # Convert to the original type 
        if tensorType == torch.Tensor:
            transformed_tensor = torch.tensor(transformed_tensor, dtype=dtype)


        return transformed_tensor


    # Core of the computation, to be used by histogram_transform method internally
    def histogram_transform_1D(self, tensor, template_histogram, noise_range = 0):
        """
        Transforms the distribution of the input tensor to match that
        of the template histogram.
        
        Input tensor will be flattened, transformed, and rearranged 
        to the original shape.
        Mainly intended for call by class functions. 
        
        Args:
            tensor (numpy.ndarray): Image to transform; the histogram is computed 
                over the flattened array.
            template_histogram (tubple of (numpy.ndarray, numpy.ndarray)): 
                The template histogram consisiting of a tuple of (counts, bins). 
                See (the output of) numpy.histogram. 
            noise_range (float): Default is 0. A uniform noise ranged between 
                (-noise_range, +noise_range) will be added to pixels randomly. 
        Returns:
            histogram transformed array (numpy.ndarray):
                The transformed output tensor/image that maches the input
        """
        # === Template Histogram ===
        # t_... stands for template_
        t_counts, t_bins = template_histogram
        # t_bin_idx not required

        # Take the cumsum of the counts and normalize by the number of pixels to
        # Get the empirical cumulative distribution functions
        # (maps value --> quantile)
        t_quantiles      = np.cumsum(t_counts).astype(np.float32)
        t_quantiles     /= t_quantiles[-1]

        # === Input Tensor ===
        # Convert to flattened numpy array
        tensor           = np.asanyarray(tensor)
        originalShape    = tensor.shape
        tensor           = tensor.ravel()

        # Get counts, bins, and corresponding bin indices for each tensor value
        counts, bins     = np.histogram(tensor, bins = t_bins)
        bin_idx          = np.searchsorted(t_bins[:-2], tensor)

        # See comments for t_quantiles
        quantiles      = np.cumsum(counts).astype(np.float32)
        quantiles     /= quantiles[-1]

        # === Histogram Transformation ===
        # interpolate linearly to find the pixel values in the template image
        # that corresponds most closely to the quantiles for the input tensor
        interp_t_values  = np.interp(quantiles, t_quantiles, t_bins[:-1])
        tensor_transformed  = interp_t_values[bin_idx]

        noise = np.random.uniform(
                low=-noise_range, 
                high=+noise_range, 
                size=(len(tensor_transformed))
            )
        tensor_transformed += noise
        tensor_transformed  = np.maximum(tensor_transformed, min(t_bins))

        return tensor_transformed.reshape(originalShape)
