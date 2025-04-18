"""
Authors revised GWR code based on Similarity and Geographically Weighted Regression manuscript.

The code for this manuscript is developed on the basis of the main GWR code that was written by 
(Taylor Oshan). And the process of bandwidth identification and selection and diagnostics 
is similar to traditional GWR.

"""

import copy
from typing import Optional
import numpy as np
import numpy.linalg as la
from scipy.stats import t
from scipy.special import factorial
from itertools import combinations as combo
from spglm.family import Gaussian, Binomial, Poisson
from spglm.glm import GLM, GLMResults
from spglm.iwls import iwls, _compute_betas_gwr
from spglm.utils import cache_readonly
from diagnostics import get_AIC, get_AICc, get_BIC, corr
from kernels import *
from summary import *
import multiprocessing as mp
from sklearn.metrics.pairwise import cosine_similarity
from mgwr.sel_bw import Sel_BW
from scipy.spatial.distance import pdist, squareform
from scipy.spatial.distance import cdist
import numba

class SGWR(GLM):
    def __init__(self, coords, y, X, bw, sim_data, bt_value,data, variables, family=Gaussian(), offset=None,
                 sigma2_v1=True, kernel='bisquare', fixed=False, constant=True,
                 spherical=False, hat_matrix=False):

        GLM.__init__(self, y, X, family, constant=constant)
        self.constant = constant
        self.sigma2_v1 = sigma2_v1
        self.coords = np.array(coords)
        self.bw = bw
        self.kernel = kernel
        self.fixed = fixed
        if offset is None:
            self.offset = np.ones((self.n, 1))
        else:
            self.offset = offset * 1.0
        self.fit_params = {}
        self.points = None
        self.exog_scale = None
        self.exog_resid = None
        self.P = None
        self.spherical = spherical
        self.hat_matrix = hat_matrix
        self.sim_data = sim_data  
        self.bt_value = bt_value
        self.data = data
        self.variables = variables

    def _build_wi(self, i, bw):
        """ This section calculate the weight matrix based on data similarity. We have (i) obeservation and we need to estimate
        its similarity with the entire dataset """
        
        similarity_matrices = []
        for var in self.variables:
            data_var = np.array(self.data[var].values.reshape(-1,1))
            dist_mat = pairwise_distances(data_var[i:i+1], data_var)
            similarity_matrices.append(dist_mat)
        combined = np.sum(similarity_matrices, axis=0)
        combined = np.divide(combined, len(similarity_matrices))
        w2 = np.exp(-combined**2)
        w2 = np.squeeze(w2) 
        
        if bw == np.inf:
            w1 = np.ones((self.n))
            
            alpha_ = self.bt_value 
            beta_ = 1 - alpha_  

            W_combined = alpha_ * w1 + beta_ * w2
            
            wi = W_combined / W_combined.max()
            
        try:
            w1 = Kernel(i, self.coords, bw, fixed=self.fixed,
                        function=self.kernel, points=self.points,
                        spherical=self.spherical).kernel
            
            
            alpha_ = self.bt_value  
            beta_ = 1 - alpha_  
            
            W_combined = alpha_ * w1 + beta_ * w2
            
            wi = W_combined / W_combined.max()
            
                
        except BaseException:
            raise  # TypeError('Unsupported kernel function  ', kernel)  
        return wi

    def _local_fit(self, i):
        """
        Local fitting at location i.
        
        """
        wi = self._build_wi(i, self.bw).reshape(-1, 1)  
        
        if isinstance(self.family, Gaussian):
            betas, inv_xtx_xt = _compute_betas_gwr(self.y, self.X, wi)
            predy = np.dot(self.X[i], betas)[0]
            resid = self.y[i] - predy
            influ = np.dot(self.X[i], inv_xtx_xt[:, i])
            w = 1
            
        elif isinstance(self.family, (Poisson, Binomial)):
            
            rslt = iwls(self.y, self.X, self.family, self.offset, None,
                        self.fit_params['ini_params'], self.fit_params['tol'],
                        self.fit_params['max_iter'], wi=wi)
            inv_xtx_xt = rslt[5]
            w = rslt[3][i][0]
            influ = np.dot(self.X[i], inv_xtx_xt[:, i]) * w
            predy = rslt[1][i]
            resid = self.y[i] - predy
            betas = rslt[0]

        if self.fit_params['lite']:
            return influ, resid, predy, betas.reshape(-1)
        else:
            
            Si = np.dot(self.X[i], inv_xtx_xt).reshape(-1) 
            tr_STS_i = np.sum(Si * Si * w * w)
            CCT = np.diag(np.dot(inv_xtx_xt, inv_xtx_xt.T)).reshape(-1)
            if not self.hat_matrix:
                Si = None
            
            return influ, resid, predy, betas.reshape(-1), w, Si, tr_STS_i, CCT

    def fit(self, ini_params=None, tol=1.0e-5, max_iter=20, solve='iwls',
            lite=False, pool=None):
        self.fit_params['ini_params'] = ini_params
        self.fit_params['tol'] = tol
        self.fit_params['max_iter'] = max_iter
        self.fit_params['solve'] = solve
        self.fit_params['lite'] = lite
        
        if solve.lower() == 'iwls':

            if self.points is None:
                m = self.y.shape[0]
            else:
                m = self.points.shape[0]
            
            if pool:
                rslt = pool.map(self._local_fit,range(m))  #parallel using mp.Pool
            else:
                rslt = map(self._local_fit, range(m))  #sequential
            
            rslt_list = list(zip(*rslt))
            influ = np.array(rslt_list[0]).reshape(-1, 1)
            resid = np.array(rslt_list[1]).reshape(-1, 1)
            params = np.array(rslt_list[3]) ### the Coef for each IDVs 
            
            if lite:
                return SGWRResultsLite(self, resid, influ, params)
            else:
                predy = np.array(rslt_list[2]).reshape(-1, 1)
                w = np.array(rslt_list[-4]).reshape(-1, 1)
                if self.hat_matrix:
                    S = np.array(rslt_list[-3])
                else:
                    S = None
                tr_STS = np.sum(np.array(rslt_list[-2]))
                CCT = np.array(rslt_list[-1]) ### shows the colinearity between IDVs
                return SGWRResults(self, params, predy, S, CCT, influ, tr_STS,w)


    def predict(self, points, P, exog_scale=None, exog_resid=None,
                fit_params={}):
        if (exog_scale is None) & (exog_resid is None):
            train_gwr = self.fit(**fit_params)
            self.exog_scale = train_gwr.scale
            self.exog_resid = train_gwr.resid_response
        elif (exog_scale is not None) & (exog_resid is not None):
            self.exog_scale = exog_scale
            self.exog_resid = exog_resid
        else:
            raise InputError('exog_scale and exog_resid must both either be'
                             'None or specified')
        self.points = points
        if self.constant:
            P = np.hstack([np.ones((len(P), 1)), P])
            self.P = P
        else:
            self.P = P
        gwr = self.fit(**fit_params)

        return gwr

    @cache_readonly
    def df_model(self):
        return None

    @cache_readonly
    def df_resid(self):
        return None


class SGWRResults(GLMResults):
    def __init__(self, model, params, predy, S, CCT, influ, tr_STS=None,
                 w=None):
        GLMResults.__init__(self, model, params, predy, w)
        self.offset = model.offset
        if w is not None:
            self.w = w
        self.predy = predy
        self.S = S
        self.tr_STS = tr_STS
        self.influ = influ
        self.CCT = self.cov_params(CCT, model.exog_scale)
        self._cache = {}

    @cache_readonly
    def W(self):
        W = np.array(
            [self.model._build_wi(i, self.model.bw) for i in range(self.n)])
        return W

    @cache_readonly
    def resid_ss(self):
        if self.model.points is not None:
            raise NotImplementedError('Not available for SGWR prediction')
        else:
            u = self.resid_response.flatten()
        return np.dot(u, u.T)

    @cache_readonly
    def scale(self, scale=None):
        if isinstance(self.family, Gaussian):
            scale = self.sigma2
        else:
            scale = 1.0
        return scale

    def cov_params(self, cov, exog_scale=None):
        if exog_scale is not None:
            return cov * exog_scale
        else:
            return cov * self.scale

    @cache_readonly
    def tr_S(self):
        """
        trace of S (hat) matrix
        """
        return np.sum(self.influ)

    @cache_readonly
    def ENP(self):  ### Effective number of parameters 
        if self.model.sigma2_v1:
            return self.tr_S
        else:
            return 2 * self.tr_S - self.tr_STS

    @cache_readonly
    def y_bar(self):
        """
        weighted mean of y
        """
        if self.model.points is not None:
            n = len(self.model.points)
        else:
            n = self.n
        off = self.offset.reshape((-1, 1))
        arr_ybar = np.zeros(shape=(self.n, 1))
        for i in range(n):
            w_i = np.reshape(self.model._build_wi(i, self.model.bw), (-1, 1))
            sum_yw = np.sum(self.y.reshape((-1, 1)) * w_i)
            arr_ybar[i] = 1.0 * sum_yw / np.sum(w_i * off)
        return arr_ybar

    @cache_readonly
    def TSS(self): ### Total Sum of Squares ()
        if self.model.points is not None:
            n = len(self.model.points)
        else:
            n = self.n
        TSS = np.zeros(shape=(n, 1)) ## set the size of this array equal to the number of observations (n)
        for i in range(n): ### iterate over the input observations (n)
            TSS[i] = np.sum(
                np.reshape(self.model._build_wi(i, self.model.bw),
                           (-1, 1)) * (self.y.reshape(
                               (-1, 1)) - self.y_bar[i])**2)
        return TSS

    @cache_readonly
    def RSS(self):
        
        if self.model.points is not None:  ### based on exogenous residual 
            n = len(self.model.points)
            resid = self.model.exog_resid.reshape((-1, 1))
        else: ## based on resid_response method
            n = self.n
            resid = self.resid_response.reshape((-1, 1))
        RSS = np.zeros(shape=(n, 1))  ### first set the size of the array equal to the number of defined observations
        for i in range(n):  ### iterate over all input observations
            RSS[i] = np.sum(
                np.reshape(self.model._build_wi(i, self.model.bw),
                           (-1, 1)) * resid**2)
        return RSS

    @cache_readonly
    def localR2(self):
        if isinstance(self.family, Gaussian):  ### check if the family is Gaussian then implement 
            return (self.TSS - self.RSS) / self.TSS
        else: ### else not applicable in this model
            raise NotImplementedError('Only applicable to Gaussian')

    @cache_readonly
    def sigma2(self):
        if self.model.sigma2_v1:
            return (self.resid_ss / (self.n - self.tr_S))
        else:
            # could be changed to SWSTW - nothing to test against
            return self.resid_ss / (self.n - 2.0 * self.tr_S + self.tr_STS)

    @cache_readonly
    def std_res(self):
        return self.resid_response.reshape(
            (-1, 1)) / (np.sqrt(self.scale * (1.0 - self.influ)))

    @cache_readonly
    def bse(self):
        return np.sqrt(self.CCT)

    @cache_readonly
    def cooksD(self):
        return self.std_res**2 * self.influ / (self.tr_S * (1.0 - self.influ))

    @cache_readonly
    def deviance(self):
        off = self.offset.reshape((-1, 1)).T
        y = self.y
        ybar = self.y_bar
        if isinstance(self.family, Gaussian):
            raise NotImplementedError(
                'deviance not currently used for Gaussian')
        elif isinstance(self.family, Poisson):
            dev = np.sum(
                2.0 * self.W * (y * np.log(y / (ybar * off)) -
                                (y - ybar * off)), axis=1)
        elif isinstance(self.family, Binomial):
            dev = self.family.deviance(self.y, self.y_bar, self.W, axis=1)
        return dev.reshape((-1, 1))

    @cache_readonly
    def resid_deviance(self):
        if isinstance(self.family, Gaussian):
            raise NotImplementedError(
                'deviance not currently used for Gaussian')
        else:
            off = self.offset.reshape((-1, 1)).T
            y = self.y
            ybar = self.y_bar
            global_dev_res = ((self.family.resid_dev(self.y, self.mu))**2)
            dev_res = np.repeat(global_dev_res.flatten(), self.n)
            dev_res = dev_res.reshape((self.n, self.n))
            dev_res = np.sum(dev_res * self.W.T, axis=0)
            return dev_res.reshape((-1, 1))

    @cache_readonly
    def pDev(self):
        if isinstance(self.family, Gaussian):
            raise NotImplementedError('Not implemented for Gaussian')
        else:
            return 1.0 - (self.resid_deviance / self.deviance)

    @cache_readonly
    def adj_alpha(self):
        alpha = np.array([.1, .05, .001])
        pe = self.ENP
        p = self.k
        return (alpha * p) / pe

    def critical_tval(self, alpha=None):
        n = self.n
        if alpha is not None:
            alpha = np.abs(alpha) / 2.0
            critical = t.ppf(1 - alpha, n - 1)
        else:
            alpha = np.abs(self.adj_alpha[1]) / 2.0
            critical = t.ppf(1 - alpha, n - 1)
        return critical

    def filter_tvals(self, critical_t=None, alpha=None):
        n = self.n
        if critical_t is not None:
            critical = critical_t
        else:
            critical = self.critical_tval(alpha=alpha)

        subset = (self.tvalues < critical) & (self.tvalues > -1.0 * critical)
        tvalues = self.tvalues.copy()
        tvalues[subset] = 0
        return tvalues

    @cache_readonly
    def df_model(self):
        return self.n - self.tr_S

    @cache_readonly
    def df_resid(self):
        return self.n - 2.0 * self.tr_S + self.tr_STS

    @cache_readonly
    def normalized_cov_params(self):
        return None

    @cache_readonly
    def resid_pearson(self):
        return None

    @cache_readonly
    def resid_working(self):
        return None

    @cache_readonly
    def resid_anscombe(self):
        return None

    @cache_readonly
    def pearson_chi2(self):
        return None

    @cache_readonly
    def llnull(self):
        return None

    @cache_readonly
    def null_deviance(self):
        return self.family.deviance(self.y, self.null)

    @cache_readonly
    def global_deviance(self):
        deviance = np.sum(self.family.resid_dev(self.y, self.mu)**2)
        return deviance

    @cache_readonly
    def D2(self):
        """
        Percentage of deviance explanied. Equivalent to 1 - (deviance/null deviance)
        """
        D2 = 1.0 - (self.global_deviance / self.null_deviance)
        return D2

    @cache_readonly
    def R2(self):
        """
        Global r-squared value for a Gaussian model.
        """
        if isinstance(self.family, Gaussian):
            return self.D2
        else:
            raise NotImplementedError('R2 only for Gaussian')

    @cache_readonly
    def adj_D2(self):
        """
        Adjusted percentage of deviance explanied.
        """
        adj_D2 = 1 - (1 - self.D2) * (self.n - 1) / (self.n - self.ENP - 1)
        return adj_D2

    @cache_readonly
    def adj_R2(self):
        """
        Adjusted global r-squared for a Gaussian model.
        """
        if isinstance(self.family, Gaussian):
            return self.adj_D2
        else:
            raise NotImplementedError('adjusted R2 only for Gaussian')

    @cache_readonly
    def aic(self):
        return get_AIC(self)

    @cache_readonly
    def aicc(self):
        return get_AICc(self)

    @cache_readonly
    def bic(self):
        return get_BIC(self)

    @cache_readonly
    def pseudoR2(self):
        return None

    @cache_readonly
    def adj_pseudoR2(self):
        return None

    @cache_readonly
    def pvalues(self):
        return None

    @cache_readonly
    def conf_int(self):
        return None

    @cache_readonly
    def use_t(self):
        return None

    def get_bws_intervals(self, selector, level=0.95):
        try:
            import pandas as pd
        except ImportError:
            return

        #Get AICcs and associated bw from the last iteration of back-fitting and make a DataFrame
        aiccs = pd.DataFrame(list(zip(*selector.sel_hist))[1],columns=["aicc"])
        aiccs['bw'] = list(zip(*selector.sel_hist))[0]
        #Sort DataFrame by the AICc values
        aiccs = aiccs.sort_values(by=['aicc'])
        #Calculate delta AICc
        d_aic_ak = aiccs.aicc - aiccs.aicc.min()
        #Calculate AICc weights
        w_aic_ak = np.exp(-0.5*d_aic_ak) / np.sum(np.exp(-0.5*d_aic_ak))
        aiccs['w_aic_ak'] = w_aic_ak/np.sum(w_aic_ak)
        #Calculate cum. AICc weights
        aiccs['cum_w_ak'] = aiccs.w_aic_ak.cumsum()
        #Find index where the cum weights above p-val
        index = len(aiccs[aiccs.cum_w_ak < level]) + 1
        #Get bw boundaries
        interval = (aiccs.iloc[:index,:].bw.min(),aiccs.iloc[:index,:].bw.max())
        return interval


    def local_collinearity(self):
        x = self.X
        w = self.W
        nvar = x.shape[1]
        nrow = len(w)
        if self.model.constant:
            ncor = (((nvar - 1)**2 + (nvar - 1)) / 2) - (nvar - 1)
            jk = list(combo(range(1, nvar), 2))
        else:
            ncor = (((nvar)**2 + (nvar)) / 2) - nvar
            jk = list(combo(range(nvar), 2))
        corr_mat = np.ndarray((nrow, int(ncor)))
        if self.model.constant:
            vifs_mat = np.ndarray((nrow, nvar - 1))
        else:
            vifs_mat = np.ndarray((nrow, nvar))
        vdp_idx = np.ndarray((nrow, nvar))
        vdp_pi = np.ndarray((nrow, nvar, nvar))

        for i in range(nrow):
            wi = self.model._build_wi(i, self.model.bw)
            sw = np.sum(wi)
            wi = wi / sw
            tag = 0

            for j, k in jk:
                corr_mat[i, tag] = corr(np.cov(x[:, j], x[:, k],
                                               aweights=wi))[0][1]
                tag = tag + 1

            if self.model.constant:
                corr_mati = corr(np.cov(x[:, 1:].T, aweights=wi))
                vifs_mat[i, ] = np.diag(
                    np.linalg.solve(corr_mati, np.identity((nvar - 1))))

            else:
                corr_mati = corr(np.cov(x.T, aweights=wi))
                vifs_mat[i, ] = np.diag(
                    np.linalg.solve(corr_mati, np.identity((nvar))))

            xw = x * wi.reshape((nrow, 1))
            sxw = np.sqrt(np.sum(xw**2, axis=0))
            sxw = np.transpose(xw.T / sxw.reshape((nvar, 1)))
            svdx = np.linalg.svd(sxw)
            vdp_idx[i, ] = svdx[1][0] / svdx[1]
            phi = np.dot(svdx[2].T, np.diag(1 / svdx[1]))
            phi = np.transpose(phi**2)
            pi_ij = phi / np.sum(phi, axis=0)
            vdp_pi[i, :, :] = pi_ij

        local_CN = vdp_idx[:, nvar - 1].reshape((-1, 1))
        VDP = vdp_pi[:, nvar - 1, :]

        return corr_mat, vifs_mat, local_CN, VDP

    def spatial_variability(self, selector, n_iters=1000, seed=None):
        temp_sel = copy.deepcopy(selector)
        temp_gwr = copy.deepcopy(self.model)

        if seed is None:
            np.random.seed(5536)
        else:
            np.random.seed(seed)

        fit_params = temp_gwr.fit_params
        search_params = temp_sel.search_params
        kernel = temp_gwr.kernel
        fixed = temp_gwr.fixed

        if self.model.constant:
            X = self.X[:, 1:]
        else:
            X = self.X

        init_sd = np.std(self.params, axis=0)
        SDs = []

        try:
            from tqdm.auto import tqdm  # if they have it, let users have a progress bar
        except ImportError:
            def tqdm(x, desc=''):  # otherwise, just passthrough the range
                return x

        for x in tqdm(range(n_iters), desc='Testing'):
            temp_coords = np.random.permutation(self.model.coords)
            temp_sel.coords = temp_coords
            temp_bw = temp_sel.search(**search_params)
            temp_gwr.bw = temp_bw
            temp_gwr.coords = temp_coords
            temp_params = temp_gwr.fit(**fit_params).params
            temp_sd = np.std(temp_params, axis=0)
            SDs.append(temp_sd)

        p_vals = (np.sum(np.array(SDs) > init_sd, axis=0) / float(n_iters))
        return p_vals

    @cache_readonly
    def predictions(self):
        P = self.model.P
        if P is None:
            raise TypeError('predictions only avaialble if predict'
                            'method is previously called on SGWR model')
        else:
            predictions = np.sum(P * self.params, axis=1).reshape((-1, 1))
        return predictions

    def summary(self, as_str: bool = False) -> Optional[str]:
        
        return summaryModel(self) + summaryGLM(self) + summarySGWR(self)
    

class SGWRResultsLite(object):
    def __init__(self, model, resid, influ, params):
        self.y = model.y
        self.family = model.family
        self.n = model.n
        self.influ = influ
        self.resid_response = resid
        self.model = model
        self.params = params

    @cache_readonly
    def tr_S(self):
        return np.sum(self.influ)

    @cache_readonly
    def llf(self):
        return self.family.loglike(self.y, self.mu)

    @cache_readonly
    def mu(self):
        return self.y - self.resid_response

    @cache_readonly
    def predy(self):
        return self.y - self.resid_response

    @cache_readonly
    def resid_ss(self):
        u = self.resid_response.flatten()
        return np.dot(u, u.T)
    
    
    
