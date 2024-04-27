import math
import os.path

import numpy.linalg as la
import pandas as pd
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler
import copy
from typing import Optional
from scipy.stats import t
from itertools import combinations as combo
from spglm.glm import GLM, GLMResults
from spglm.iwls import iwls, _compute_betas_gwr
from spglm.utils import cache_readonly
from diagnostics import get_AIC, get_AICc, get_BIC, corr
from summary import *
from scipy.spatial.distance import cdist
from scipy import linalg
from sklearn.metrics import mean_absolute_error, mean_squared_error


class ALPHA_OPT(GLM):
    def __init__(self, coords, y, X, bw, data, variables, comm, n_n, bt_value,
                 family=Gaussian(), offset=None,
                 sigma2_v1=True, kernel='bisquare', fixed=False, constant=True,
                 spherical=False, hat_matrix=False):
        """
        Initialize class
        """
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
        self.data = data
        self.variables = variables
        self.bt_value = bt_value
        self.iter = n_n
        self.comm = comm

        self.k = self.X.shape[1]

        m_m = int(math.ceil(float(len(self.iter)) / self.comm.size))
        self.x_chunk = self.iter[self.comm.rank * m_m:(self.comm.rank + 1) * m_m]

    def weight_func(self, i, bw):
        """ This section calculate the weight matrix based on data similarity. We have (i) obeservation and we need to estimate
        its similarity with the entire dataset """

        """ Calculate the weight matrix based on data similarity. """
        similarity_matrices = []
        for var in self.variables:
            data_var = np.array(self.data[var].values.reshape(-1, 1))
            dist_mat = pairwise_distances(data_var[i:i + 1], data_var)
            similarity_matrices.append(dist_mat)
        combined = np.sum(similarity_matrices, axis=0)
        combined = np.divide(combined, len(similarity_matrices))
        w2 = np.exp(-combined ** 2)
        w2 = np.squeeze(w2)
        ############# Geographically weighted
        dis = np.sqrt(np.sum((self.coords[i] - self.coords) ** 2, axis=1))
        dis = np.squeeze(dis)
        if self.fixed:
            w1 = np.exp(-0.5 * (dis / bw) ** 2).reshape(-1, 1)
            alpha_ = self.bt_value

            beta_ = 1 - alpha_

            W_combined = alpha_ * w1 + beta_ * w2
            wi = W_combined / W_combined.max()
        else:
            maxd = np.partition(dis, int(bw) - 1)[int(bw) - 1] * 1.0000001
            wegt = dis / maxd
            wegt[wegt >= 1] = 1
            w1 = ((1 - (wegt) ** 2) ** 2).reshape(-1, 1)
            w1 = np.squeeze(w1)

            ###########################
            alpha_ = self.bt_value
            beta_ = 1 - alpha_

            W_combined = alpha_ * w1 + beta_ * w2

            wi = W_combined / W_combined.max()
        return wi


    def local_fitting(self, i):
        wi = self.weight_func(i, self.bw).reshape(-1, 1)

        if isinstance(self.family, Gaussian):
            x = self.X
            y = self.y
            xT = (x * wi).T
            xtx = np.dot(xT, x)
            inv_xtx_xt = linalg.solve(xtx, xT)
            betas = np.dot(inv_xtx_xt, y)
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
            return influ, resid, predy, betas
        else:

            Si = np.dot(self.X[i], inv_xtx_xt).reshape(-1)
            tr_STS_i = np.sum(Si * Si * w * w)
            CCT = np.diag(np.dot(inv_xtx_xt, inv_xtx_xt.T)).reshape(-1)
            if not self.hat_matrix:
                Si = None

            return influ, resid, predy, betas, w, Si, tr_STS_i, CCT


    def alpha_opt_serial(self):
        influ_ = []
        resid_ = []
        predy_ = []
        betas_ = []
        w_ = []
        si_ = []
        tr_STS_i_ = []
        cct_ = []
        if self.fit_params['lite']:
            for i in self.x_chunk:
                influ, resid, predy, betass = self.local_fitting(i)
                influ_.append(influ)
                resid_.append(resid)
                predy_.append((predy))
                betas_.append(betass)
        else:
            for i in self.x_chunk:
                influ, resid, predy, betass, w, si, tr_STS_i, cct = self.local_fitting(i)
                influ_.append(influ)
                resid_.append(resid)
                predy_.append(predy)
                betas_.append(betass)
                w_.append(w)
                si_.append(si)
                tr_STS_i_.append(tr_STS_i)
                cct_.append(cct)

        influ_g = self.comm.gather(influ_, root=0)
        resid_g = self.comm.gather(resid_, root=0)
        predy_g = self.comm.gather(predy_, root=0)
        betas_g = self.comm.gather(betas_, root=0)
        w_g = self.comm.gather(w_, root=0)
        si_g = self.comm.gather(si_, root=0)
        tr_STS_i_g = self.comm.gather(tr_STS_i_, root=0)
        cct_g = self.comm.gather(cct_, root=0)

        if self.comm.rank == 0:
            # aicc2 = 0
            S = 0
            influ_lis = np.concatenate(influ_g).reshape(-1, 1)

            resdi_lis = np.concatenate(resid_g).reshape(-1, 1)

            params = np.concatenate(betas_g)
            params = np.squeeze(params, axis=-1)

            predy_lis = np.concatenate(predy_g).reshape(-1, 1)

            w_lis = np.concatenate(w_g).reshape(-1, 1)

            tr_sts = np.concatenate(tr_STS_i_g)

            CCT_lis = np.concatenate(cct_g).reshape(-1, 1)

            if self.hat_matrix:
                S = np.concatenate(si_g)
            else:
                S = None

        else:
            influ_lis = None
            resdi_lis = None
            params = None
            predy_lis = None
            w_lis = None
            tr_sts = None
            CCT_lis = None
            S = None

        influ_lis = self.comm.bcast(influ_lis, root=0)
        resdi_lis = self.comm.bcast(resdi_lis, root=0)
        params = self.comm.bcast(params, root=0)
        predy_lis = self.comm.bcast(predy_lis, root=0)
        w_lis = self.comm.bcast(w_lis, root=0)
        tr_sts = self.comm.bcast(tr_sts, root=0)
        CCT_lis = self.comm.bcast(CCT_lis, root=0)
        S = self.comm.bcast(S, root=0)

        return influ_lis, resdi_lis, predy_lis, params, w_lis, S, tr_sts, CCT_lis


    def fit_func(self, ini_params=None, tol=1.0e-5, max_iter=20, solve='iwls',
                 lite=False, pool=None):
        self.fit_params['ini_params'] = ini_params
        self.fit_params['tol'] = tol
        self.fit_params['max_iter'] = max_iter
        self.fit_params['solve'] = solve
        self.fit_params['lite'] = lite

        influ_lis, resdi_lis, predy_lis, params, w_lis, S, tr_sts, CCT_lis = self.alpha_opt_serial()

        if lite:
            return SGWRResultsLite(self, resdi_lis, influ_lis, params)
        else:
            return SGWRResults(self, params, predy_lis, S, CCT_lis, influ_lis, tr_sts, w_lis)


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
    def ENP(self):
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
    def TSS(self):
        if self.model.points is not None:
            n = len(self.model.points)
        else:
            n = self.n
        TSS = np.zeros(shape=(n, 1))
        for i in range(n):
            TSS[i] = np.sum(
                np.reshape(self.model._build_wi(i, self.model.bw),
                           (-1, 1)) * (self.y.reshape(
                    (-1, 1)) - self.y_bar[i]) ** 2)
        return TSS

    @cache_readonly
    def RSS(self):

        if self.model.points is not None:
            n = len(self.model.points)
            resid = self.model.exog_resid.reshape((-1, 1))
        else:
            n = self.n
            resid = self.resid_response.reshape((-1, 1))
        RSS = np.zeros(shape=(n, 1))
        for i in range(n):
            RSS[i] = np.sum(
                np.reshape(self.model._build_wi(i, self.model.bw),
                           (-1, 1)) * resid ** 2)

        return RSS

    @cache_readonly
    def localR2(self):
        if isinstance(self.family, Gaussian):
            return (self.TSS - self.RSS) / self.TSS
        else:
            raise NotImplementedError('Only applicable to Gaussian')

    @cache_readonly
    def sigma2(self):
        if self.model.sigma2_v1:
            return (self.resid_ss / (self.n - self.tr_S))
        else:
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
        return self.std_res ** 2 * self.influ / (self.tr_S * (1.0 - self.influ))

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
            global_dev_res = ((self.family.resid_dev(self.y, self.mu)) ** 2)
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
        deviance = np.sum(self.family.resid_dev(self.y, self.mu) ** 2)
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

        # Get AICcs and associated bw from the last iteration of back-fitting and make a DataFrame
        aiccs = pd.DataFrame(list(zip(*selector.sel_hist))[1], columns=["aicc"])
        aiccs['bw'] = list(zip(*selector.sel_hist))[0]
        # Sort DataFrame by the AICc values
        aiccs = aiccs.sort_values(by=['aicc'])
        # Calculate delta AICc
        d_aic_ak = aiccs.aicc - aiccs.aicc.min()
        # Calculate AICc weights
        w_aic_ak = np.exp(-0.5 * d_aic_ak) / np.sum(np.exp(-0.5 * d_aic_ak))
        aiccs['w_aic_ak'] = w_aic_ak / np.sum(w_aic_ak)
        # Calculate cum. AICc weights
        aiccs['cum_w_ak'] = aiccs.w_aic_ak.cumsum()
        # Find index where the cum weights above p-val
        index = len(aiccs[aiccs.cum_w_ak < level]) + 1
        # Get bw boundaries
        interval = (aiccs.iloc[:index, :].bw.min(), aiccs.iloc[:index, :].bw.max())
        return interval

    def local_collinearity(self):
        x = self.X
        w = self.W
        nvar = x.shape[1]
        nrow = len(w)
        if self.model.constant:
            ncor = (((nvar - 1) ** 2 + (nvar - 1)) / 2) - (nvar - 1)
            jk = list(combo(range(1, nvar), 2))
        else:
            ncor = (((nvar) ** 2 + (nvar)) / 2) - nvar
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
                vifs_mat[i,] = np.diag(
                    np.linalg.solve(corr_mati, np.identity((nvar - 1))))

            else:
                corr_mati = corr(np.cov(x.T, aweights=wi))
                vifs_mat[i,] = np.diag(
                    np.linalg.solve(corr_mati, np.identity((nvar))))

            xw = x * wi.reshape((nrow, 1))
            sxw = np.sqrt(np.sum(xw ** 2, axis=0))
            sxw = np.transpose(xw.T / sxw.reshape((nvar, 1)))
            svdx = np.linalg.svd(sxw)
            vdp_idx[i,] = svdx[1][0] / svdx[1]
            phi = np.dot(svdx[2].T, np.diag(1 / svdx[1]))
            phi = np.transpose(phi ** 2)
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
            from tqdm.auto import tqdm
        except ImportError:
            def tqdm(x, desc=''):
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


class SGWR:
    def __init__(self, comm, parser):
        """
        Class Initialization
        """
        self.comm = comm
        self.parser = parser
        self.X = None
        self.y = None
        self.coords = None
        self.n = None
        self.k = None
        self.iter = None
        self.minbw = None
        self.maxbw = None
        self.bw = None

        self.parse_sgwr_args()

        if self.comm.rank == 0:
            self.read_file()
            self.k = self.X.shape[1]
            self.iter = np.arange(self.n)

        self.X = comm.bcast(self.X, root=0)
        self.y = comm.bcast(self.y, root=0)
        self.coords = comm.bcast(self.coords, root=0)
        self.iter = comm.bcast(self.iter, root=0)
        self.n = comm.bcast(self.n, root=0)
        self.k = comm.bcast(self.k, root=0)

        m = int(math.ceil(float(len(self.iter)) / self.comm.size))
        self.x_chunk = self.iter[self.comm.rank * m:(self.comm.rank + 1) * m]

    def parse_sgwr_args(self):
        """
        Parsing arguments from the command line
        """
        parser_arg = self.parser.parse_args()
        self.fname = parser_arg.data
        self.fout = parser_arg.out
        self.fixed = parser_arg.fixed
        self.constant = parser_arg.constant
        self.estonly = parser_arg.estonly

        if parser_arg.bw:
            if self.fixed:
                self.bw = float(parser_arg.bw)
            else:
                self.bw = int(parser_arg.bw)

        if parser_arg.minbw:
            if self.fixed:
                self.minbw = float(parser_arg.minbw)
            else:
                self.minbw = int(parser_arg.minbw)

        if self.comm.rank == 0:
            print("-" * 60, flush=True)

    def read_file(self):
        """
        Read file from the path
        """

        input = np.genfromtxt(self.fname, dtype=float, delimiter=',', skip_header=True)
        self.y = input[:, 2].reshape(-1, 1)
        self.n = input.shape[0]

        if self.constant:
            self.X = np.hstack([np.ones((self.n, 1)), input[:, 3:]])
        else:
            self.X = input[:, 3:]
        self.coords = input[:, :2]

    def set_search_range(self):
        """
        Define the search range in golden section
        """
        if self.fixed:
            max_dist = np.max(np.array([np.max(cdist([self.coords[i]], self.coords)) for i in range(self.n)]))
            self.maxbw = max_dist * 2

            if self.minbw is None:
                min_dist = np.min(
                    np.array([np.min(np.delete(cdist(self.coords[[i]], self.coords), i)) for i in range(self.n)]))

                self.minbw = min_dist / 2

        else:
            self.maxbw = self.n

            if self.minbw is None:
                self.minbw = 40 + 2 * self.k

    def build_wi(self, i, bw):
        """
        Build the local weight matrix for location i
        """
        dist = cdist([self.coords[i]], self.coords).reshape(-1)
        if self.fixed:
            wi = np.exp(-0.5 * (dist / bw) ** 2).reshape(-1, 1)

        else:
            maxd = np.partition(dist, int(bw) - 1)[int(bw) - 1] * 1.0000001
            zs = dist / maxd
            zs[zs >= 1] = 1
            wi = ((1 - (zs) ** 2) ** 2).reshape(-1, 1)

        return wi

    def local_fit(self, i, y, X, bw, final=False):
        """
        Fit the local regression model at location i
        """
        wi = self.build_wi(i, bw)

        if final:
            xT = (X * wi).T
            xtx_inv_xt = np.dot(np.linalg.inv(np.dot(xT, X)), xT)
            betas = np.dot(xtx_inv_xt, y).reshape(-1)

            ri = np.dot(X[i], xtx_inv_xt)
            predy = np.dot(X[i], betas)
            err = y[i][0] - predy
            CCT = np.diag(np.dot(xtx_inv_xt, xtx_inv_xt.T))
            return np.concatenate(([i, err, ri[i]], betas, CCT))

        else:
            X_new = X * np.sqrt(wi)
            Y_new = y * np.sqrt(wi)
            temp = np.dot(np.linalg.inv(np.dot(X_new.T, X_new)), X_new.T)
            hat = np.dot(X_new[i], temp[:, i])
            yhat = np.sum(np.dot(X_new, temp[:, i]).reshape(-1, 1) * Y_new)
            err = Y_new[i][0] - yhat
            return err * err, hat

    def golden_section(self, a, c, function):
        """
        Golden-section search bandwidth optimization
        """
        delta = 0.38197
        b = a + delta * np.abs(c - a)
        d = c - delta * np.abs(c - a)
        opt_bw = None
        score = None
        diff = 1.0e9
        iters = 0
        dict = {}
        a_a = a
        while np.abs(diff) > 1.0e-6 and iters < 200:
            iters += 1
            if not self.fixed:
                b = np.round(b)
                d = np.round(d)

            if b in dict:
                score_b = dict[b]
            else:
                score_b = function(b)
                dict[b] = score_b
            if d in dict:
                score_d = dict[d]
            else:
                score_d = function(d)
                dict[d] = score_d

            if self.comm.rank == 0:
                if score_b <= score_d:
                    opt_bw = b
                    opt_score = score_b
                    c = d
                    d = b
                    if b <= 3 * a_a:
                        b = (a + d) / 2
                    else:
                        b = (a + d) / 4

                else:
                    opt_bw = d
                    opt_score = score_d
                    a = b
                    b = (a + d) / 2
                    d = d

                diff = score_b - score_d
                score = opt_score
            b = self.comm.bcast(b, root=0)
            d = self.comm.bcast(d, root=0)
            opt_bw = self.comm.bcast(opt_bw, root=0)
            diff = self.comm.bcast(diff, root=0)
            score = self.comm.bcast(score, root=0)

        return opt_bw

    def mpi_sgwr_fit(self, y, X, bw, final=False):
        """
        Fit the model given a bandwidth
        """
        k = X.shape[1]
        if final:
            sub_Betas = np.empty((self.x_chunk.shape[0], 2 * k + 3), dtype=np.float64)
            pos = 0

            for i in self.x_chunk:
                sub_Betas[pos] = self.local_fit(i, y, X, bw, final=True)
                pos += 1

            Betas_list = self.comm.gather(sub_Betas, root=0)

            if self.comm.rank == 0:
                data = np.vstack(Betas_list)

                RSS = np.sum(data[:, 1] ** 2)
                TSS = np.sum((y - np.mean(y)) ** 2)
                R2 = 1 - RSS / TSS
                trS = np.sum(data[:, 2])

                sigma2_v1 = RSS / (self.n - trS)

                aicc = self.compute_aicc(RSS, trS)
                data[:, -k:] = np.sqrt(data[:, -k:] * sigma2_v1)

                header = "index,residual,influ,"
                varNames = np.genfromtxt(self.fname, dtype=str, delimiter=',', names=True, max_rows=1).dtype.names[3:]
                if self.constant:
                    varNames = ['intercept'] + list(varNames)
                for x in varNames:
                    header += ("b_" + x + ',')
                for x in varNames:
                    header += ("se_" + x + ',')

            return

        sub_RSS = 0
        sub_trS = 0
        resides = 0
        for i in self.x_chunk:
            err2, hat = self.local_fit(i, y, X, bw, final=False)
            sub_RSS += err2
            sub_trS += hat
            resides += math.sqrt(err2)

        RSS_list = self.comm.gather(sub_RSS, root=0)
        trS_list = self.comm.gather(sub_trS, root=0)

        if self.comm.rank == 0:
            RSS = sum(RSS_list)
            trS = sum(trS_list)

            aicc = self.compute_aicc(RSS, trS)

            return aicc

        return

    def fit(self, y=None, X=None):
        if y is None:
            y = self.y
            X = self.X
        if self.bw:
            self.mpi_sgwr_fit(y, X, self.bw, final=True)
            return

        if self.comm.rank == 0:
            self.set_search_range()
        self.minbw = self.comm.bcast(self.minbw, root=0)
        self.maxbw = self.comm.bcast(self.maxbw, root=0)

        sgwr_func = lambda bw: self.mpi_sgwr_fit(y, X, bw)

        opt_bw = self.golden_section(self.minbw, self.maxbw, sgwr_func)
        if self.fixed:
            opt_bw = round(opt_bw, 2)

        self.opt_bw = opt_bw

        opt_alpha = self.optimal_alpah()

    def optimal_alpah(self):
        data = np.genfromtxt(self.fname, dtype=str, delimiter=',')
        x_x = data[1:, 0]  ### x coordinate
        y_y = data[1:, 1]  ### y coordiante
        # Get header
        header = data[0, 3:]

        g_x = data[1:, 3:]

        g_y = data[1:, 2].reshape(-1, 1)

        scaler = StandardScaler()
        g_x = scaler.fit_transform(g_x)
        g_y = scaler.fit_transform(g_y)

        x_x = pd.Series(x_x)
        y_y = pd.Series(y_y)

        g_coords = list(zip(x_x, y_y))
        g_coords = [(float(x), float(y)) for x, y in g_coords]

        columns = header.tolist()

        data = data[1:, 3:]
        data = pd.DataFrame(data, columns=columns)
        models = []
        aiccs = []
        bt_value = 0.5
        alph_determine = ALPHA_OPT(g_coords, g_y, g_x, self.opt_bw, data, columns, self.comm, self.iter, bt_value,
                                   kernel='gaussian')
        result = alph_determine.fit_func()
        aiccV = result.aicc
        aiccs.append(aiccV)
        models.append(result)

        ind_bt = aiccs.index(min(aiccs))

        if self.comm.rank == 0:
            selected_model = models[ind_bt]
            print('R2', selected_model.R2)
            print('Adj R2', selected_model.adj_R2)
            print('AICc', selected_model.aicc)
            rss = np.sum(selected_model.resid_response ** 2)
            print('RSS', rss)

            ### Calculate MAPE
            def calculate_mape(actual, predicted):
                return 100 * sum(abs((a - p) / a) for a, p in zip(actual, predicted)) / len(actual)

            mape = calculate_mape(g_y, selected_model.predy)
            print("MAPE:", mape[0])

            sgwr_y_prd = selected_model.predy
            # Compute MAE
            mae = mean_absolute_error(g_y, sgwr_y_prd)
            print("MAE:", mae)

            # Compute RMSE
            rmse = np.sqrt(mean_squared_error(g_y, sgwr_y_prd))
            print("RMSE:", rmse)

            ### For saving the summary table and coef
            self.save_results(header, selected_model, g_y)

    def compute_aicc(self, RSS, trS):
        """
        Compute AICc
        """
        aicc = self.n * np.log(RSS / self.n) + self.n * np.log(2 * np.pi) + self.n * (self.n + trS) / (
                self.n - trS - 2.0)

        return aicc

    def save_results(self, header, selected_model, g_y):
        """
        Save results to .csv file
        """
        if self.comm.rank == 0:
            # Summary Table
            summary_ouput = os.path.join(os.path.dirname(self.fname), "summarySGWR.txt")
            summary_SGWR = selected_model.summary()
            with open(summary_ouput, 'w') as file:
                file.write(summary_SGWR)

            # Residual values
            residuals = g_y - (selected_model.predy)

            # Coef values
            coef_value = selected_model.params
            results = np.concatenate((residuals, coef_value), axis=1)
            header_str = ','.join(header)

            header_str = 'Intercept,' + header_str
            header_str = 'Residual,' + header_str

            np.savetxt(self.fout, results, delimiter=',', header=header_str, comments='', fmt='%s')
