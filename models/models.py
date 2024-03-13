import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
import statsmodels as sms
from firthlogist import FirthLogisticRegression

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm, truncnorm
from scipy.stats import beta

from tqdm.notebook import tqdm


class QLearningModule:
    def __init__(self, *args, **kwargs):
        self.lr = kwargs.get('lr', .5)
        self.rl_temp = kwargs.get('rl_temp', 1)
        n_states, n_actions = kwargs.get(
            'n_states', 2), kwargs.get('n_actions', 2)
        self.q = np.ones((n_states, n_actions)) * kwargs.get('q0', 50)

    def value_update(self, s, a, r):
        self.q[s, a] += self.lr * (r - self.q[s, a])


class PerceptualLogit:
    def __init__(self, *args, **kwargs):
        self.perceptual_temp = kwargs.get('perceptual_temp', 1)
        self.perceptual_model = kwargs.get('perceptual_model', 'logit')
        self.x = kwargs.get('x', np.arange(-1, 1, .2))

        self.firth_fit = None
        self.logit_fit = None
        # history of forcefields
        self.hist_ff = []
        # history of rewards (destroyed or not)
        self.hist_r = []

    def predictff(self, ff1=None, ff2=None):

        model = self.perceptual_model

        if ff1 is not None and ff2 is not None:
            x = np.arange(len(self.x))
            to_select = np.array([x[self.x == ff1][0], x[self.x == ff2][0]])
        else:
            # to_select = np.arange(len(self.x))
            to_select = np.arange(len(np.unique(self.hist_ff)))
        if model == 'logit':
            try:
                self.logit_fit = sm.Logit(self.hist_r, sm.add_constant(self.hist_ff))\
                    .fit_regularized(disp=0, start_params=[0, 0])

                return self.logit_fit.predict(sm.add_constant(self.x))[to_select]
            # catch separation error and value errrors
            except (sms.tools.sm_exceptions.PerfectSeparationError,
                    ValueError, np.linalg.LinAlgError):
                # equal probability for all forcefields
                try:
                    return 1/len(to_select) * np.ones(len(to_select))
                except ZeroDivisionError:
                    return

        elif model == 'firth':
            try:
                self.firth_fit = FirthLogisticRegression(skip_ci=True, wald=False, fit_intercept=True)\
                    .fit(sm.add_constant(self.hist_ff), self.hist_r)
                return self.firth_fit.predict(sm.add_constant(np.unique(self.hist_ff)))[to_select]
            except:
                return 1/len(to_select) * np.ones(len(to_select))
        else:
            raise ValueError(
                'model must be either logit, firth, or val. Model is {}'.format(model))

    def get_intercept_and_slope(self):
        model = self.perceptual_model
        if model == 'logit':
            if self.logit_fit is None:
                self.predictff()
                if self.logit_fit is None:
                    return np.array([0, 0])
            return np.clip(self.logit_fit.params, -10, 10)
        # elif model=='linear':
            # return
        elif model == 'firth':
            return np.array(self.firth_fit.coef_)
        elif model == 'val':
            return self.val_fit
        else:
            raise ValueError('model must be either logit or linear')


class LogitRLEV(PerceptualLogit, QLearningModule):
    def __init__(self, *args, **kwargs):
        PerceptualLogit.__init__(self, *args, **kwargs)
        QLearningModule.__init__(self, *args, **kwargs)

    def make_choice(self, s, ff1, ff2):
        def logsumexp(x):
            c = x.max()
            return c + np.log(np.sum(np.exp(x - c)))

        pdestroy = 1

        if ff1 is not None and ff2 is not None:
            pdestroy = self.predictff(ff1=ff1, ff2=ff2)

        x = ((self.q[s, :] * self.rl_temp)
             * (pdestroy
             * self.perceptual_temp))

        p = np.exp(x - logsumexp(x)).round(3)

        return np.random.choice(np.arange(2), p=p)

    def learn_perceptual(self, a, r):
        self.hist_ff.append(a)
        self.hist_r.append(r)
        self.predictff()

    def learn_value(self, s, a, r):
        self.value_update(s, a, r)

    def get_params(self, s=None):
        intercept, slope = self.get_intercept_and_slope()
        return {
            # alpha logit (utf8 character symbol as key)
            'α': intercept,
            # beta logit
            'β': slope,
            # alpha qlearning
            'lr': self.lr,
            # temperature qlearning
            'rl_temp': self.rl_temp,
            # temperature logit
            'perceptual_temp': self.perceptual_temp
        }


class RLLogitLogRatio(QLearningModule, PerceptualLogit):
    def __init__(self, *args, **kwargs):
        QLearningModule.__init__(self, *args, **kwargs)
        PerceptualLogit.__init__(self, *args, **kwargs)

    def make_choice(self, s, ff1, ff2):
        def logsumexp(x):
            c = x.max()
            return c + np.log(np.sum(np.exp(x - c)))

        x1 = np.log(self.q[s, :]) * self.rl_temp
        x2 = np.log(self.predictff(ff1=ff1, ff2=ff2)) * self.perceptual_temp
        x = x1 + x2

        p = np.exp(x - logsumexp(x))

        try:
            return np.random.choice(np.arange(2), p=p)
        except ValueError as e:
            print(e)
            return np.random.choice(np.arange(2), p=[0.5, 0.5])


class NormativePerceptual:
    def __init__(self, *args, **kwargs):
        # super().__init__(*args, **kwargs)
        # slope prior
        self.slope_prior = kwargs.get('slope_prior', 2)
        # leak parameter
        self.leak = kwargs.get('leak', 0)
        # x values = forcefield values
        self.x = kwargs.get('x', np.arange(-1, 1, 10))

        # possible values for the slope
        self.slope_range = kwargs.get(
            'slope_range', np.arange(-10, 10, 0.002))

        # initialize logposterior for the slope to prior
        self.lp_slope = np.log(norm.pdf(self.slope_range, 0, self.slope_prior))

        # define logit function
        self.logit = lambda x: 1 / (1+np.exp(x))

    def perceptual_update(self, choice, destroyed):
        v = -self.slope_range*choice if destroyed else self.slope_range*choice
        # compute log likelihood
        ll = np.log(self.logit(v))
        # update log posterior
        self.lp_slope += ll
        self._apply_leak_perceptual()

    def predictff(self, ff1=None, ff2=None):
        if ff1 is None and ff2 is None:
            # predict p(destroy) for all forcefields
            to_select = np.arange(len(self.x))
        else:
            # predict p(destroy) for 2 (displayed) forcefields
            x = np.arange(len(self.x))
            to_select = np.array([x[self.x == ff1][0], x[self.x == ff2][0]])

        return self.logit(-self.get_slope()*self.x[to_select])

    def get_slope(self):
        w = np.exp(self.lp_slope-np.max(self.lp_slope))
        slope = np.sum(w*self.slope_range)/np.sum(w)
        return slope
    #

    def _apply_leak_perceptual(self):
        self.lp_slope *= 1-self.leak


class NormativeValue:
    def __init__(self, *args, **kwargs):
        # super().__init__(*args, **kwargs)
        self.sigma_prior = kwargs.get('sigma_prior', .2)
        self.std_prior = kwargs.get('std_prior', .2)
        self.leak = kwargs.get('leak', 0)
        self.nstate = kwargs.get('nstate', 2)

        # possible values for the difference between two options
        self.sigma_range = kwargs.get('sigma_range', np.arange(-1, 1, 0.002))

        # initialize logposterior for the slope to prior
        self.lp_sigma = [
            np.log(tnormpdf(0.5*(1+self.sigma_range), 0.5, self.sigma_prior))
            for _ in range(self.nstate)
        ]

    def value_update(self, s, a, r):
        v = 1-self.sigma_range if a else 1+self.sigma_range
        # compute log likelihood
        ll = np.log(tnormpdf(r, 0.5*(v), self.std_prior))
        # update log posterior
        self.lp_sigma[s] += ll
        self._apply_leak_value(s)

    def get_sigma(self, s):
        w = np.exp(self.lp_sigma[s]-np.max(self.lp_sigma[s]))
        sigma = np.sum(w*self.sigma_range)/np.sum(w)
        return sigma

    def _apply_leak_value(self, s):
        self.lp_sigma[s] *= 1-self.leak


class Normative(NormativeValue, NormativePerceptual):
    def __init__(self, *args, **kwargs):
        NormativeValue.__init__(self, *args, **kwargs)
        NormativePerceptual.__init__(self, *args, **kwargs)

        self.temp = kwargs.get('temp', 1e6)

    def make_choice(self, s, ff1, ff2):
        # default is EV decision
        p = self.predictff(ff1, ff2) if ff1 is not None else 1
        ev1, ev2 = (
            (.5 * (1+self.get_sigma(s)*(np.array([1, -1])))) * p).round(3)
        choice = np.random.random() > 1/(1+np.exp(-self.temp*(ev1-ev2)))
        return choice

    def learn_perceptual(self, a, r):
        self.perceptual_update(a, r)

    def learn_value(self, s, a, r):
        self.value_update(s, a, r)

    def get_params(self, s=None):
        return {
            # beta logit (utf8 character symbol as key)
            'β': self.get_slope(),
            # sigma (diff between two options) (utf8 character symbol as key)
            'σ': self.get_sigma(s) if s is not None
            else [self.get_sigma(s) for s in range(self.nstate)],
        }



class NormativeEV(Normative,
                  NormativeValue, NormativePerceptual):
    def ll_of_choice(self, ff1, ff2, s, a):
        p1 = self.predictff(ff1, ff2) if ff1 is not None else 1
        ev1, ev2 = (
            (.5 * (1+self.get_sigma(s)*(np.array([1, -1])))) * p1).round(3)
        p1 = 1/(1+np.exp(-self.temp*(ev1-ev2)))
        p = [p1, 1-p1]
        if p[a] == 0:
            p[a] += 1e-6
        return p[a]


class NormativeLogRatio(Normative,
                        NormativeValue, NormativePerceptual):
    def __init__(self, *args, **kwargs):
        Normative.__init__(self, *args, **kwargs)
        self.perceptual_temp = kwargs.get('perceptual_temp', 1e6)
        self.rl_temp = kwargs.get('rl_temp', 1e6)

    def make_choice(self, s, ff1, ff2):
        p = np.log(self.predictff(ff1, ff2)) if ff1 is not None else 1
        ev = np.log(
            (.5 * (1+self.get_sigma(s)*(np.array([1, -1])))).round(3)
        )
        dv = ev[0] - ev[1]
        dp = p[0] - p[1]
        x = (
            np.array([dv, dp]) *
            np.array([self.rl_temp, self.perceptual_temp])
        ).sum().round(3)
        choice = int(np.random.random() > 1/(1+np.exp(-x)))
        return choice

    def ll_of_choice(self, ff1, ff2, s, a):
        p = np.log(self.predictff(ff1, ff2)) if ff1 is not None else 1
        ev = np.log(
            (.5 * (1+self.get_sigma(s)*(np.array([1, -1])))).round(3)
        )
        dv = ev[0] - ev[1]
        dp = p[0] - p[1]
        x = (
            np.array([dv, dp]) *
            np.array([self.rl_temp, self.perceptual_temp])
        ).sum().round(3)
        p1 = 1/(1+np.exp(-x))
        p = [p1, 1-p1]
        if p[a] == 0:
            p[a] += 1e-6

        return p[a]


class RandomModel:
    def __init__(self, *args, **kwargs):
        pass

    @staticmethod
    def make_choice(s, ff1, ff2):
        return np.random.choice(np.arange(2))

    @staticmethod
    def learn_perceptual(a, r):
        pass

    @staticmethod
    def learn_value(s, a, r):
        pass

    @staticmethod
    def get_params(s=None):
        pass


def tnormpdf(x, m, s):
    x = x + np.zeros_like(m)
    p = norm.pdf(x, m, s)
    cdf_range = norm.cdf(1, m, s) - norm.cdf(0, m, s)
    p = p / (s * cdf_range)
    return p


def tnormpdf2(x, loc, std, lb=0, ub=1):
    # TODO: check why it doesn't work
    a, b = (lb-loc)/std, (ub-loc)/std
    return truncnorm(a, b, loc=loc).pdf(x)


def tnormrdn(loc, std, lb=0, ub=1, size=1):
    a, b = (lb-loc)/std, (ub-loc)/std
    x = truncnorm(a, b, loc=loc, scale=std).rvs(size=size)
    return x[0] if len(x) == 1 else x

import matlab.engine
def find_matlab_engine():
    try:
        # Try to connect to an existing MATLAB session
        eng = matlab.engine.connect_matlab()
        return eng
    except Exception as e:
        print(f"Failed to connect to an existing MATLAB session: {e}")
        return None