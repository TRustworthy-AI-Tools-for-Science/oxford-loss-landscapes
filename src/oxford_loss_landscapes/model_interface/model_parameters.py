"""
Basic linear algebra operations as defined on the parameter sets of entire models.

We can think of these list as a single vectors consisting of all the individual
parameter values. The functions in this module implement basic linear algebra
operations on such lists.

The operations defined in the module follow the PyTorch convention of appending
the '__' suffix to the name of in-place operations.
"""

import copy
import math
import numpy as np
import torch
import torch.nn

class ModelParameters:
    """
    A ModelParameters object is an abstract view of a model's optimizable parameters as a tensor. This class
    enables the parameters of models of the same 'shape' (architecture) to be operated on as if they were 'real'
    tensors. A ModelParameters object cannot be converted to a true tensor as it is potentially irregularly
    shaped.
    """

    def __init__(self, parameters: list):
        if not isinstance(parameters, list) or not all(isinstance(p, torch.Tensor) for p in parameters):
            raise AttributeError('Argument to ModelParameter is not a list of torch.Tensor objects.')

        self.parameters = parameters

    def __len__(self) -> int:
        """
        Returns the number of model layers within the parameter tensor.
        :return: number of layer tensors
        """
        return len(self.parameters)

    def numel(self) -> int:
        """
        Returns the number of elements (i.e. individual parameters) within the tensor.
        Note that this refers to individual parameters, not layers.
        :return: number of elements in tensor
        """
        return sum(p.numel() for p in self.parameters)

    def __getitem__(self, index) -> torch.nn.Parameter:
        """
        Returns the tensor of the layer at the given index.
        :param index: layer index
        :return: tensor of layer
        """
        return self.parameters[index]

    def __eq__(self, other: 'ModelParameters') -> bool:
        """
        Compares this parameter tensor for equality with the argument tensor, using the == operator.
        :param other: the object to compare to
        :return: true if equal
        """
        if not isinstance(other, ModelParameters) or len(self) != len(other):
            return False
        else:
            return all(torch.equal(p_self, p_other) for p_self, p_other in zip(self.parameters, other.parameters))

    def __add__(self, other: 'ModelParameters') -> 'ModelParameters':
        """
        Constructively returns the result of addition between this tensor and another.
        :param other: other to add
        :return: self + other
        """
        return ModelParameters([self[idx] + other[idx] for idx in range(len(self))])

    def __radd__(self, other: 'ModelParameters') -> 'ModelParameters':
        """
        Constructively returns the result of addition between this tensor and another.
        :param other: model parameters to add
        :return: other + self
        """
        return self.__add__(other)

    def add_(self, other: 'ModelParameters'):
        """
        In-place addition between this tensor and another.
        :param other: model parameters to add
        :return: none
        """
        with torch.no_grad():
            for idx in range(len(self)):
                self.parameters[idx] += other[idx]

    def __sub__(self, other: 'ModelParameters') -> 'ModelParameters':
        """
        Constructively returns the result of subtracting another tensor from this one.
        :param other: model parameters to subtract
        :return: self - other
        """
        return ModelParameters([self[idx] - other[idx] for idx in range(len(self))])

    def __rsub__(self, other: 'ModelParameters') -> 'ModelParameters':
        """
        Constructively returns the result of subtracting this tensor from another one.
        :param other: other to subtract from
        :return: other - self
        """
        return self.__sub__(other)

    def sub_(self, vector: 'ModelParameters'):
        """
        In-place subtraction of another tensor from this one.
        :param vector: other to subtract
        :return: none
        """
        with torch.no_grad():
            for idx in range(len(self)):
                self.parameters[idx] -= vector[idx]

    def __mul__(self, scalar) -> 'ModelParameters':
        """
        Constructively returns the result of multiplying this tensor by a scalar.
        :param scalar: scalar to multiply by
        :return: self * scalar
        """
        return ModelParameters([self[idx] * scalar for idx in range(len(self))])

    def __rmul__(self, scalar) -> 'ModelParameters':
        """
        Constructively returns the result of multiplying this tensor by a scalar.
        :param scalar: scalar to multiply by
        :return: scalar * self
        """
        return self.__mul__(scalar)

    def mul_(self, scalar):
        """
        In-place multiplication of this tensor by a scalar.
        :param scalar: scalar to multiply by
        :return: none
        """
        with torch.no_grad():
            for idx in range(len(self)):
                self.parameters[idx] *= scalar

    def __truediv__(self, scalar) -> 'ModelParameters':
        """
        Constructively returns the result of true-dividing this tensor by a scalar.
        :param scalar: scalar to divide by
        :return: scalar / self
        """
        return ModelParameters([self[idx] / scalar for idx in range(len(self))])

    def truediv_(self, scalar):
        """
        In-place true-division of this tensor by a scalar.
        :param scalar: scalar to divide by
        :return: none
        """
        with torch.no_grad():
            for idx in range(len(self)):
                self.parameters[idx] /= scalar

    def __floordiv__(self, scalar) -> 'ModelParameters':
        """
        Constructively returns the result of floor-dividing this tensor by a scalar.
        :param scalar: scalar to divide by
        :return: scalar // self
        """
        return ModelParameters([self[idx] // scalar for idx in range(len(self))])

    def floordiv_(self, scalar):
        """
        In-place floor-division of this tensor by a scalar.
        :param scalar: scalar to divide by
        :return: none
        """
        with torch.no_grad():
            for idx in range(len(self)):
                self.parameters[idx] //= scalar

    def __matmul__(self, other: 'ModelParameters') -> 'ModelParameters':
        """
        Constructively returns the result of tensor-multiplication of this tensor by another tensor.
        :param other: other tensor
        :return: self @ tensor
        """
        raise NotImplementedError()

    def dot(self, other: 'ModelParameters') -> float:
        """
        Returns the vector dot product of this ModelParameters vector and the given other vector.
        :param other: other ModelParameters vector
        :return: dot product of self and other
        """
        param_products = []
        for idx in range(len(self.parameters)):
            param_products.append((self.parameters[idx] * other.parameters[idx]).sum().item())
        return sum(param_products)

    def model_normalize_(self, ref_point: 'ModelParameters', order=2):
        """
        In-place model-wise normalization of the tensor.
        :param ref_point: use this model's norm, if given
        :param order: norm order, e.g. 2 for L2 norm
        :return: none
        """
        eps = 1e-12
        self_norm = self.model_norm(order)
        if self_norm <= eps:
            return
        scale = ref_point.model_norm(order) / self_norm
        with torch.no_grad():
            for parameter in self.parameters:
                parameter *= scale

    def layer_normalize_(self, ref_point: 'ModelParameters', order=2):
        """
        In-place layer-wise normalization of the tensor.
        :param ref_point: use this model's layer norms, if given
        :param order: norm order, e.g. 2 for L2 norm
        :return: none
        """
        eps = 1e-12
        with torch.no_grad():
            for layer_idx, parameter in enumerate(self.parameters, 0):
                self_norm = self.layer_norm(layer_idx, order)
                if self_norm > eps:
                    parameter *= (ref_point.layer_norm(layer_idx, order) / self_norm)

    def filter_normalize_(self, ref_point: 'ModelParameters', order=2):
        """
        In-place filter-wise normalization of the tensor.
        Handles 0-D scalar parameters, 1-D bias/vector parameters, and
        2-D+ weight tensors safely.
        :param ref_point: use this model's filter norms, if given
        :param order: norm order, e.g. 2 for L2 norm
        :return: none
        """
        eps = 1e-12  # Small epsilon to prevent division by zero

        with torch.no_grad():
            for l in range(len(self.parameters)):
                param = self.parameters[l]
                ref_param = ref_point.parameters[l]
                # normalize 0-D scalar parameters (e.g. MLFF scale/shift params)
                if param.ndim == 0:
                    self_norm = param.abs().item()
                    if self_norm > eps:
                        param *= (ref_param.abs().item() / self_norm)
                # normalize 1-D bias/vector parameters
                elif param.ndim == 1:
                    ref_norm = ref_param.norm(order).item()
                    self_norm = param.norm(order).item()
                    if self_norm > eps:
                        param *= (ref_norm / self_norm)
                # normalize 2-D+ weight tensors per filter (dim 0)
                else:
                    for f in range(param.shape[0]):
                        ref_filter_norm = ref_point.filter_norm((l, f), order)
                        self_filter_norm = self.filter_norm((l, f), order)
                        if self_filter_norm > eps:
                            param[f] *= ref_filter_norm / self_filter_norm

    def model_norm(self, order=2) -> float:
        """
        Returns the model-wise L-norm of the tensor.
        :param order: norm order, e.g. 2 for L2 norm
        :return: L-norm of tensor
        """
        # L-n norm of model where we treat the model as a flat vector
        return math.pow(sum([
            torch.pow(torch.abs(layer), order).sum().item()
            for layer in self.parameters
        ]), 1.0 / order)

    def layer_norm(self, index, order=2) -> float:
        """
        Returns the layer-wise L-norm of the specified parameter tensor.
        :param order: norm order, e.g. 2 for L2 norm
        :param index: layer index
        :return: L-norm of layer
        """
        return math.pow(torch.pow(torch.abs(self.parameters[index]), order).sum().item(), 1.0 / order)

    def filter_norm(self, index, order=2) -> float:
        """
        Returns the L-norm of a specific filter within a layer.
        :param order: norm order, e.g. 2 for L2 norm
        :param index: tuple with layer index and filter index
        :return: L-norm of filter
        """
        return math.pow(torch.pow(torch.abs(self.parameters[index[0]][index[1]]), order).sum().item(), 1.0 / order)

    def as_numpy(self) -> np.ndarray:
        """
        Returns the tensor as a flat numpy array.
        :return: a numpy array
        """
        return np.concatenate([p.detach().cpu().numpy().flatten() for p in self.parameters])

    def _get_parameters(self) -> list:
        """
        Returns a reference to the internal parameter data in whatever format used by the source model.
        :return: reference to internal parameter data
        """
        return self.parameters


def rand_u_like(example_vector: ModelParameters) -> ModelParameters:
    """
    Create a new ModelParameters object of size and shape compatible with the given
    example vector, such that the values in the ModelParameter are uniformly distributed
    in the range [0,1].
    :param example_vector: defines by example the size and shape the new vector will have
    :return: new vector with uniformly distributed values
    """
    new_vector = []

    for param in example_vector:
        new_vector.append(torch.rand(size=param.size(), dtype=param.dtype, device=param.device))

    return ModelParameters(new_vector)


def rand_n_like(example_vector: ModelParameters) -> ModelParameters:
    """
    Create a new ModelParameters object of size and shape compatible with the given
    example vector, such that the values in the ModelParameter are normally distributed
    as N(0,1).
    :param example_vector: defines by example the size and shape the new vector will have
    :return: new vector with normally distributed values
    """
    new_vector = []

    for param in example_vector:
        new_vector.append(torch.randn(size=param.size(), dtype=param.dtype, device=param.device))

    return ModelParameters(new_vector)


def orthogonal_to(vector: ModelParameters) -> ModelParameters:
    """
    Create a new ModelParameters object of size and shape compatible with the given
    example vector, such that the two vectors are orthogonal.
    :param vector: original vector
    :return: new vector orthogonal to original vector
    """
    new_vector = rand_n_like(vector)
    norm_sq = math.pow(vector.model_norm(2), 2)
    if norm_sq > 1e-24:
        new_vector = new_vector - (new_vector.dot(vector) / norm_sq) * vector
    return new_vector


def add(vector_a: ModelParameters, vector_b: ModelParameters) -> ModelParameters:
    return vector_a + vector_b


def sub(vector_a: ModelParameters, vector_b: ModelParameters) -> ModelParameters:
    return vector_a - vector_b


def mul(vector: ModelParameters, scalar) -> ModelParameters:
    return vector * scalar


def truediv(vector: ModelParameters, scalar) -> ModelParameters:
    return vector / scalar


def floordiv(vector: ModelParameters, scalar) -> ModelParameters:
    return vector // scalar


def filter_normalize(tensor: ModelParameters, ref_point: ModelParameters, order=2) -> ModelParameters:
    new_tensor = copy.deepcopy(tensor)
    new_tensor.filter_normalize_(ref_point, order=order)
    return new_tensor


def layer_normalize(tensor: ModelParameters, ref_point: ModelParameters, order=2) -> ModelParameters:
    new_tensor = copy.deepcopy(tensor)
    new_tensor.layer_normalize_(ref_point, order=order)
    return new_tensor


def model_normalize(tensor: ModelParameters, ref_point: ModelParameters, order=2) -> ModelParameters:
    new_tensor = copy.deepcopy(tensor)
    new_tensor.model_normalize_(ref_point, order=order)
    return new_tensor
