
import torch

class Normalize:
    """
    A normalization transform that standardizes input data using the provided mean and standard deviation.

    This class can be used as a callable to normalize input samples, and also provides an inverse transformation
    to recover the original data.

    Attributes:
        mean (torch.Tensor): Mean values for each feature.
        std (torch.Tensor): Standard deviation values for each feature.
    """

    def __init__(self, mean, std):
        """
        Initialize the Normalize transform.

        Args:
            mean (array-like): Mean values for each feature.
            std (array-like): Standard deviation values for each feature.
        """
        self.mean = torch.tensor(mean, dtype=torch.float64)
        self.std = torch.tensor(std, dtype=torch.float64)

    def __call__(self, sample):
        """
        Normalize the input sample.

        Args:
            sample (array-like): Input sample to normalize.

        Returns:
            torch.Tensor: Normalized sample.
        """
        sample = torch.tensor(sample, dtype=torch.float64)
        return (sample - self.mean) / self.std

    def inverse_transform(self, sample):
        """
        Inverse the normalization transformation.

        Args:
            sample (array-like): Normalized sample to invert.

        Returns:
            torch.Tensor: Original (unnormalized) sample.
        """
        sample = torch.tensor(sample, dtype=torch.float64)
        return sample * self.std + self.mean
