"""
Originally this diagnostics code was for gwr model. M. Naser Lessani made some minor changes to the codes according to SGWR model.
"""
__author__ = "Taylor Oshan tayoshan@gmail.com"

import numpy as np

from spglm.family import Gaussian, Poisson, Binomial


def get_AICc(sgwr):
    """
    Get AICc value

    Gaussian: p61, (2.33), Fotheringham, Brunsdon and Charlton (2002)

    GWGLM: AICc=AIC+2k(k+1)/(n-k-1), Nakaya et al. (2005): p2704, (36)

    """
    n = sgwr.n
    k = sgwr.tr_S
    if isinstance(sgwr.family, Gaussian):
        aicc = -2.0 * sgwr.llf + 2.0 * n * (k + 1.0) / (n - k - 2.0)

    elif isinstance(sgwr.family, (Poisson, Binomial)):
        aicc = get_AIC(sgwr) + 2.0 * k * (k + 1.0) / (n - k - 1.0)

    return aicc


def get_AIC(sgwr):
    """
    Get AIC calue

    Gaussian: p96, (4.22), Fotheringham, Brunsdon and Charlton (2002)

    GWGLM:  AIC(G)=D(G) + 2K(G), where D and K denote the deviance and the effective
    number of parameters in the model with bandwidth G, respectively.

    """
    k = sgwr.tr_S
    y = sgwr.y
    mu = sgwr.mu
    if isinstance(sgwr.family, Gaussian):
        aic = -2.0 * sgwr.llf + 2.0 * (k + 1)
    elif isinstance(sgwr.family, (Poisson, Binomial)):
        aic = np.sum(sgwr.family.resid_dev(y, mu) ** 2) + 2.0 * k
    return aic


def get_BIC(sgwr):
    """
    Get BIC value

    Gaussian: p61 (2.34), Fotheringham, Brunsdon and Charlton (2002)
    BIC = -2log(L)+klog(n)

    GWGLM: BIC = dev + tr_S * log(n)

    """
    n = sgwr.n  
    k = sgwr.tr_S
    y = sgwr.y
    mu = sgwr.mu
    if isinstance(sgwr.family, Gaussian):
        bic = -2.0 * sgwr.llf + (k + 1) * np.log(n)
    elif isinstance(sgwr.family, (Poisson, Binomial)):
        bic = np.sum(sgwr.family.resid_dev(y, mu) ** 2) + k * np.log(n)
    return bic


def get_CV(sgwr):
    """
    Get CV value

    Gaussian only

    Methods: p60, (2.31) or p212 (9.4)
    Fotheringham, A. S., Brunsdon, C., & Charlton, M. (2002).
    Geographically weighted regression: the analysis of spatially varying relationships.
    Modification: sum of residual squared is divided by n according to GWR4 results

    """
    aa = sgwr.resid_response.reshape((-1, 1)) / (1.0 - sgwr.influ)
    cv = np.sum(aa ** 2) / sgwr.n
    return cv


def corr(cov):
    invsd = np.diag(1 / np.sqrt(np.diag(cov)))
    cors = np.dot(np.dot(invsd, cov), invsd)
    return cors
