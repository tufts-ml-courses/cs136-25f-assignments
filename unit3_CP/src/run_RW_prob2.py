import numpy as np
import pandas as pd
import scipy.stats
import scipy.special 
import os
import itertools

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
sns.set_context("notebook")

from FeatureTransformSqExpPoly import SqExpPolyTransform
from RandomWalkSampler import RandomWalkSampler
from util_softplus import softplus

def calc_joint_log_pdf(z_D, t_N, phi_NM, w_prior, v_prior):
    ''' Compute the log pdf of all random variables in model

    Args
    ----
    z_D : 1D array, shape (D,)
        Parameter vector, representing values of all hidden random variables
    t_N : 1D array, shape (N,)
        Output values observed, representing random variable t.
    phi_NM : 2D array, shape (N, M)
        Features for each example in training set. Assumed known and fixed.
    w_prior : dict
    v_prior : dict

    Returns
    -------
    logpdf : float real scalar
        Log probability density function value at provided input
    '''
    N, M = phi_NM.shape

    # Unpack vector z into constituent parts
    w_M = z_D[:M]
    v_M = z_D[M:]

    log_pdf_t = 0.0 # TODO compute likelhood log p(t_1, ... t_N | w, v)
    # HINT: use provided function unpack_mean_N_and_stddev_N

    log_pdf_w = 0.0 # TODO compute prior log p(w)
    
    log_pdf_v = 0.0 # TODO compute prior log p(v)

    return log_pdf_t + log_pdf_w + log_pdf_v

def calc_score(list_of_z_D, phi_RM, t_R):
    ''' Calculate per-example score averaged over provided test set of size R

    Args
    ----
    list_of_z_D : list of ndarray
        List of samples of parameters, assumed to be from target posterior
    phi_RM : 2D array, shape (R, M)
        Feature vectors for each of the examples in test set of size R
    t_R : 1D array, shape (R,)
        Output values for each of the examples in test set of size R

    Returns
    -------
    score : float
        Per-example log pdf of all t values in test set
        using Monte-Carlo approximation to marginal likelihood
    '''
    S = len(list_of_z_D)
    for ss in range(S):
        z_ss_D = list_of_z_D[ss]
        # Compute score formula for ss-th sample (see CP3 instructions)
        # Hint 1: Use unpack_mean_N_and_stddev_N
        # Hint 2: verify dimensionality makes sense at every step
    # TODO aggregate across all S samples
    # Hint: use scipy.special.logsumexp to be numerically stable
    return 0.0 # TODO FIXME


def unpack_mean_N_and_stddev_N(z_D, phi_NM):
    ''' Obtain mean and stddev of Gaussian likelihood for r.v. t

    Args
    ----
    z_D : 1D array, shape (D,)
        Compact vector representing all parameters of our model
    phi_NM : 2D array, shape (N,M)
        Each row represents a provided feature vector we want prediction for

    Returns
    -------
    mean_N : 1D array, shape (N,)
        Mean of r.v. t_N at each provided feature vector
    stddev_N : 1D array, shape (N,)
        Standard deviation of r.v. t_N at each provided feature vector
    '''
    M = z_D.size // 2
    w_M = z_D[:M]
    v_M = z_D[M:]
    mean_N = np.dot(phi_NM, w_M)
    # Small positive constant ensures learning doesn't push stddev to 0.0
    stddev_N = 0.0001 + softplus(np.dot(phi_NM, v_M))
    return mean_N, stddev_N


def load_bird_data(
        data_dir='../bird_data/',
        xcol = 'days_since_0401',
        ycol = 'bird_density',
        standardize_x=True):
    ''' Load data for training / evaluating regression models

    Returns
    -------
    x_tr_N1
    t_tr_N
    x_test_R1
    t_test_R
    '''
    tr_df = pd.read_csv(os.path.join(data_dir, "density_per_day_train.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "density_per_day_test.csv"))

    # Unpack input/output pairs x and t
    x_tr_N1 = tr_df[xcol].values[:,np.newaxis]
    t_tr_N = tr_df[ycol]

    x_test_R1 = test_df[xcol].values[:,np.newaxis]
    t_test_R = test_df[ycol]

    # Standardize x values using train set mean/stddev
    if standardize_x:
        mx = np.mean(x_tr_N1)
        sx = np.std(x_tr_N1)
        x_tr_N1 = (x_tr_N1 - mx) / sx
        x_test_R1 = (x_test_R1 - mx) / sx
    return (x_tr_N1, t_tr_N, x_test_R1, t_test_R)

def show_posterior_predictive_with_data(
        z_keep_list, x_tr_N1, t_tr_N, tfm,
        G=201,
        y_label='bird density',
        x_label='day of year (standardized around Apr 1)',
        random_state=np.random,
        lo_prctile=2.5,
        hi_prctile=97.5,
        extrapolation_ratio=0.3,
        transparency_level=0.2):
    ''' Visualize posterior predictive overlayed on train data

    Args
    ----
    z_keep_list : list of ndarray
        List of parameter vectors that are posterior samples
    x_tr_N1 : array, shape (N, 1)
        Training set feature vectors (before phi transform)
    t_tr_N : array, shape (N,)
        Training set output values
    tfm : feature transformer
    G : int
        Number of x values in grid to use to visualize predictions
    '''
    xmax = x_tr_N1.max()
    xmin = x_tr_N1.min()
    xbuf = extrapolation_ratio * float(xmax - xmin)
    xgrid_G1 = np.linspace(xmin - xbuf, xmax + xbuf, G)[:,np.newaxis]
    phi_GM = tfm.transform(xgrid_G1)
    M = phi_GM.shape[1]
    list_of_t_G = []
    for ss, z_ss_D in enumerate(z_keep_list):
        # Form posterior predictive distribution's parameters
        mean_G, stddev_G = unpack_mean_N_and_stddev_N(z_ss_D, phi_GM)
        # Sample from posterior predictive
        t_ss_G = mean_G + random_state.randn(G) * stddev_G
        list_of_t_G.append(t_ss_G)
    t_SG = np.vstack(list_of_t_G)

    fig_handle = plt.plot(x_tr_N1[:,0], t_tr_N, 'k.')

    # Estimate mean and plot it as a line
    that_G = t_SG.mean(axis=0)
    plt.plot(xgrid_G1[:,0], that_G, 'b-', linewidth=2)
    # Estimate 5-95% interval
    plt.fill_between(xgrid_G1[:,0],
        np.percentile(t_SG, lo_prctile, axis=0),
        np.percentile(t_SG, hi_prctile, axis=0),
        color='b', alpha=transparency_level)
    plt.ylim([-10, 70]); plt.yticks([0, 10, 20, 30, 40, 50, 60])
    plt.ylabel(y_label)
    plt.xticks([-2., -1., 0, +1., +2.])
    plt.xlabel(x_label)
    return fig_handle

if __name__ == '__main__':
    B = 1000 # Num Burnin samples
    S = 1000 # Num Keep samples

    data_dir = os.path.abspath(os.path.join("..", "bird_data"))
    if not os.path.exists(data_dir):
        raise FileNotFoundError("Cannot find expected directory:", data_dir)
    x_tr_N1, t_tr_N, x_test_R1, t_test_R = load_bird_data(data_dir=data_dir)

    order_base_list = [(0,0)] # TODO FIXME also try (2,5)
    seed_list = [101, 202]

    for (order, num_bases), random_state in itertools.product(
            order_base_list, seed_list):

        tfm = SqExpPolyTransform(
            poly_order=order, num_bases=num_bases,
            min_val=-2.5, max_val=2.5, length_scale=0.333)
        phi_tr_NM = tfm.transform(x_tr_N1)
        N, M = phi_tr_NM.shape

        # Build list of random walk proposals
        # Each one should have high stddev at one entry, zero at others
        # This way each proposal tries to change one thing at a time
        # More likely to be accepted more often
        D = 2 * M
        list_of_rw_stddev_D = [np.zeros(D) for _ in range(D)]
        for d in range(D):
            list_of_rw_stddev_D[d][d] = 0.9

        # Set priors
        w_bias_mean = 10.0
        w_prior = {
            'mean': np.hstack([w_bias_mean, np.zeros(M-1)]),
            'stddev': 3.0,
        }
        v_bias_mean = 5.0
        v_prior = {
            'mean': np.hstack([v_bias_mean, np.zeros(M-1)]),
            'stddev': 3.0,
        }

        # Create a function to calculate the target logpdf up to additive const
        # Will provide this into RandomWalkSampler
        def calc_tilde_logpdf(z_D):
            return calc_joint_log_pdf(
                z_D, t_tr_N, phi_tr_NM, w_prior, v_prior)

        # Create initial parameters to fit data "reasonably"
        # 1) Estimate of w using ML 
        what_M = np.linalg.solve(
            0.001 * np.eye(M) + np.matmul(phi_tr_NM.T, phi_tr_NM),
            np.matmul(phi_tr_NM.T, t_tr_N))
        # 2) Estimate of variance under order=0 model
        vhat_bias = np.sqrt(np.var(t_tr_N))
        vhat_M = np.hstack([vhat_bias, np.zeros(M-1)])
        # 3) Concat these estimates into initial vector, with noise
        zinit_D = np.hstack([what_M, vhat_M]) + (
            0.05 * np.random.RandomState(random_state).randn(D))

        # Launch sampler
        walker = RandomWalkSampler(
            calc_tilde_logpdf,
            list_of_rw_stddev_D,
            random_state=random_state,
            )
        z_keep_list, info = walker.draw_samples(zinit_D,
            n_burnin_samples=B,
            n_keep_samples=S)

        # Assess on test data from 2018
        phi_test_RM = tfm.transform(x_test_R1)
        test_logpdf_score = calc_score(z_keep_list, phi_test_RM, t_test_R)
        # Also assess on train data
        tr_logpdf_score = calc_score(z_keep_list, phi_tr_NM, t_tr_N)

        # Print useful message
        print("### SqExpPolyTransform(order=%d, num_bases=%d) | Seed %d" % (
                order, num_bases, random_state))
        print("Sampler settings:")
        for d, rw_stddev_D in enumerate(list_of_rw_stddev_D):
            print("  rw_stddev_D for dim %d: %s" % (d, str(rw_stddev_D)))
        print("Summary of sampled values:")
        print("       %7s %7s %7s %7s" % (
            "mean", "2.5th", "97.5th", "accept%"))
        z_SD = np.vstack(z_keep_list)
        for d in range(D):
            print("dim %d: % 7.2f % 7.2f % 7.2f %7.1f" % (d, 
                np.mean(z_SD[:,d]),
                np.percentile(z_SD[:,d], 2.5),
                np.percentile(z_SD[:,d], 97.5),
                100*info['did_accept_SP'][B:,d].mean(axis=0)))
        print("Results:")
        print("  train logpdf score: % .6f" % tr_logpdf_score)
        print("  test logpdf score: % .6f" % test_logpdf_score)

        # Create figure for 2a and save it
        plt.figure() # create fresh figure
        show_posterior_predictive_with_data(
            z_keep_list, x_tr_N1, t_tr_N, tfm,
            random_state=walker.random_state)
        fpath = "fig2a_order%d_numbases%d_seed%d.png" % (
            order, num_bases, random_state)
        plt.savefig(fpath, bbox_inches='tight', pad_inches=0)
        print("Prediction visuals saved to %s" % fpath)
