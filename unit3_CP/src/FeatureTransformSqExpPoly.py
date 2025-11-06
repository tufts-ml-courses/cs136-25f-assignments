'''
SqExpPoly Feature Transform

Summary
-------
Feature transform that takes 1D input only

Transform concatenates polynomial features and sq exp basis features

Example input: [x]
Example output (order = 1, num_bases=0):
    [1, x]
Example output (order = 1, num_bases=3):
    [1, x, k(x,-B), k(x, 0), k(x,+B)]
    where k is the squared exponential kernel with provided length_scale

API
---
Follows an sklearn-like "transform" interface
* __init__(<<options>>) : constructor
* transform(x) : transform provided data in array x
'''

import numpy as np

from sklearn.gaussian_process.kernels import RBF

class SqExpPolyTransform:

    def __init__(self, input_dim=1, input_feature_names=None,
          poly_order=0,
          num_bases=0,
          length_scale=1.0,
          min_val=-1,
          max_val=+1):
        ''' Create FeatureTransform 
        '''
        if poly_order < 0:
            raise ValueError("Polynomial order cannot be negative")
        if num_bases < 0:
            raise ValueError("Number of basis locations cannot be negative")
        if input_dim < 1 or input_dim > 1:
            raise ValueError(
                "Input dimension must be exactly 1")
        self.input_dim = int(input_dim)
        self.poly_order = int(poly_order)
        self.num_bases = int(num_bases)
        self.output_dim = 1 + self.num_bases + self.poly_order
        self.length_scale = float(length_scale)
        self.min_val = min_val
        self.max_val = max_val
        if input_feature_names is None:
            self.input_feature_names = ['x%2d' %
                                        a for a in range(self.input_dim)]
        else:
            self.input_feature_names = [str(a) for a in input_feature_names]

    def get_basis_locations(self):
        B = int(self.num_bases)
        if B < 1:
            return np.empty(0)
        locs_B = np.linspace(self.min_val, self.max_val, B)
        return locs_B

    def get_feature_size(self):
        return self.output_dim
        
    def get_feature_names(self):
        ''' Get list of string names, one for each transformed feature

        Examples
        --------
        >>> tfm = SqExpPolyTransform(input_feature_names=['a'],
        ...     input_dim=1, poly_order=1, num_bases=2)
        >>> tfm.get_feature_names()
        ['bias', 'a^1', 'k(a,-1.000)', 'k(a, 1.000)']
        '''
        feat_names = ['bias']
        for P in range(1, self.poly_order+1):
            feat_names += map(lambda s: '%s^%d' % (s,P),
                              self.input_feature_names)
        locs_B = self.get_basis_locations()
        for b in range(self.num_bases):
            feat_names += map(lambda s: 'k(%s,% .3f)' % (s,locs_B[b]),
                              self.input_feature_names)
        return feat_names

    def transform(self, x_ND):
        ''' Perform feature transformation on raw input measurements

        Examples
        --------
        >>> x_ND = np.arange(-2, 2.001, 1)[:, np.newaxis]
        >>> x_ND
        array([[-2.],
               [-1.],
               [ 0.],
               [ 1.],
               [ 2.]])
        >>> tfm0 = SqExpPolyTransform(poly_order=1, num_bases=0)
        >>> tfm0.transform(x_ND)
        array([[ 1., -2.],
               [ 1., -1.],
               [ 1.,  0.],
               [ 1.,  1.],
               [ 1.,  2.]])
        >>> tfm1 = SqExpPolyTransform(poly_order=1, num_bases=5,
        ...     length_scale=0.3, min_val=-2.2, max_val=+2.2)
        >>> phi_NM = tfm1.transform(x_ND)
        >>> print(np.array2string(phi_NM, precision=2, suppress_small=True))
        [[ 1.   -2.    0.8   0.01  0.    0.    0.  ]
         [ 1.   -1.    0.    0.95  0.    0.    0.  ]
         [ 1.    0.    0.    0.    1.    0.    0.  ]
         [ 1.    1.    0.    0.    0.    0.95  0.  ]
         [ 1.    2.    0.    0.    0.    0.01  0.8 ]]
        '''
        N, D = x_ND.shape
        if not self.input_dim == D:
            raise ValueError(
                "Mismatched input dimension. Expected %d but received %d" % (self.input_dim, D))
        assert D == 1
        phi_NM = np.zeros((N, self.output_dim), dtype=x_ND.dtype)

        phi_NM[:, 0] = 1.0
        for p in range(1, self.poly_order+1):
            phi_NM[:, p] = np.pow(x_ND[:,0], p)

        locs_B1 = self.get_basis_locations()[:,np.newaxis]
        k = RBF(length_scale=self.length_scale)
        phi_NM[:, 1+self.poly_order:] = k(x_ND, locs_B1)

        return phi_NM
