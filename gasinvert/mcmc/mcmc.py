import gasinvert.atmospheric_measurements as gp

import tensorflow_probability.substrates.jax as tfp
tfd = tfp.distributions
import jax.numpy as jnp
import jax
from jax import random
from jax.flatten_util import ravel_pytree
from dataclasses import dataclass
from jaxtyping import Float
import time
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd



__all__ = ['Priors', 'MALA_Within_Gibbs', 'Manifold_MALA_Within_Gibbs', 'Plots']


@dataclass
class Priors:

    # slab allocation prior
    theta: Float

    # Original scale SS
    spike_var: Float
    spike_mean: Float
    slab_var: Float
    slab_mean: Float

    # Log scale SS
    log_spike_mean: Float
    log_spike_var: Float
    log_slab_mean: Float
    log_slab_var: Float

    # Sigma squared
    sigma_squared_con: Float
    sigma_squared_rate: Float

    # Background
    mean_log_background_prior: Float
    variance_log_background_prior: Float

    # tan_gamma_H and tan_gamma_V
    tan_gamma_con: Float
    tan_gamma_rate: Float

    # b_H and b_V
    b_mean: Float
    b_var: Float





class Gibbs_samplers:

    def __init__(self, gaussianplume, data, priors):
        
        self.gaussianplume = gaussianplume
        self.data = data
        self.priors = priors



    def background_conditional_posterior(self, s, A,  sigma_squared, background_cov, key):
        """Conditional posterior for emission rate vector s"""
        covariance = 1/(1/ sigma_squared + 1/background_cov)
        a = jnp.mean((self.data - jnp.matmul(A, s.reshape(-1,1))) / sigma_squared)
        b =  jnp.exp(self.priors.mean_log_background_prior) / background_cov 
        mean = covariance *(a + b)

        return tfd.Normal(loc = mean, scale = jnp.sqrt(covariance)).sample(self.gaussianplume.sensors_settings.sensor_number, seed=key)
    


    def measurement_error_var_conditional_posterior(self, y, A, beta, s, key):
        n = self.gaussianplume.sensors_settings.sensor_number * self.gaussianplume.wind_field.number_of_time_steps
        variance = self.priors.sigma_squared_rate + (jnp.sum(jnp.square(y - beta - jnp.matmul(A, s.reshape(-1,1)))) / 2)
        shape = (n/2) + self.priors.sigma_squared_con

        return tfd.InverseGamma(concentration = shape, scale = variance).sample(1, key)[0]



    def binary_indicator_Zi_conditional_posterior(self, s, key):
        a = (jnp.square(s.reshape(-1,1) - self.priors.log_slab_mean) / (2 * self.priors.log_slab_var)) 
        numerator = jnp.power(2 * jnp.pi * self.priors.log_slab_var, -0.5) *  self.priors.theta
        denominator = numerator + (jnp.power(2 * jnp.pi * self.priors.log_spike_var , -0.5) * jnp.exp(-(jnp.square(s.reshape(-1,1) - self.priors.log_spike_mean) / (2 * self.priors.log_spike_var)) + a) * (1-self.priors.theta))
        bern_prob = numerator / denominator
        return tfd.Binomial(total_count=1, probs=bern_prob).sample(1, seed=key).squeeze()





class MWG_tools:

    def __init__(self, MMALA, grided, gaussianplume, data, log_posterior, priors, mh_params, fixed):

        self.MMALA = MMALA
        self.grided = grided
        self.gaussianplume = gaussianplume
        self.data = data
        self.log_posterior = log_posterior
        self.priors = priors
        self.mh_unflat_func = ravel_pytree(mh_params)[1]
        self.fixed = fixed

        self.binary_indicator_Zi_conditional_posterior = Gibbs_samplers(gaussianplume, data, priors).binary_indicator_Zi_conditional_posterior



    def glpi(self, x, sigma_squared, betas, ss_var, ss_mean, A):
        return jax.grad(self.log_posterior)(self.mh_unflat_func(x), sigma_squared, betas, ss_var, ss_mean, self.data, self.priors, A)
    


    def mwg_scan(self, step, Gibbs_init, MH_init, iters, r_eps):
        key = random.PRNGKey(0)
        sigma_squared, background = Gibbs_init["sigma_squared"], Gibbs_init["background"].reshape(-1,1)
        betas = jnp.repeat(background, 10_000).reshape(-1,1)
        z = self.binary_indicator_Zi_conditional_posterior(self.mh_unflat_func(MH_init)["log_s"], key)
        ss_var = jnp.where(z==0, self.priors.log_spike_var, self.priors.log_slab_var).reshape(-1,1)
        ss_mean = jnp.where(z==0, self.priors.log_spike_mean, self.priors.log_slab_mean).reshape(-1,1)
        if self.grided == True:
            A = self.gaussianplume.temporal_grided_coupling_matrix(self.fixed, jnp.exp(self.mh_unflat_func(MH_init)["log_tan_gamma_H"]), jnp.exp(self.mh_unflat_func(MH_init)["log_tan_gamma_V"]), jnp.exp(self.mh_unflat_func(MH_init)["log_b_H"]), jnp.exp(self.mh_unflat_func(MH_init)["log_b_V"]))
        else:
            A = self.gaussianplume.temporal_gridfree_coupling_matrix(self.fixed, jnp.exp(self.mh_unflat_func(MH_init)["source_x"]), jnp.exp(self.mh_unflat_func(MH_init)["source_y"]), jnp.exp(self.mh_unflat_func(MH_init)["log_tan_gamma_H"]), jnp.exp(self.mh_unflat_func(MH_init)["log_tan_gamma_V"]), jnp.exp(self.mh_unflat_func(MH_init)["log_b_H"]), jnp.exp(self.mh_unflat_func(MH_init)["log_b_V"]))
        ll, gradi = jax.value_and_grad(self.log_posterior)(self.mh_unflat_func(MH_init), sigma_squared, betas, ss_var, ss_mean, self.data, self.priors, A)
        sum_accept = 0
        z_count = jnp.zeros(len(self.mh_unflat_func(MH_init)["log_s"]))
        gradients = ravel_pytree(gradi)[0]
        new_grad_squared_sum = gradients**2
        max_dist = jnp.full(len(MH_init), r_eps)
        if self.MMALA == True:
            dt = r_eps
        else:
            dt = max_dist / jnp.sqrt(new_grad_squared_sum)
        iteration = 0
        _, states = jax.lax.scan(step, [MH_init, sigma_squared, background, ll, dt, sum_accept, z_count, new_grad_squared_sum, max_dist, iteration], jax.random.split(key, iters))

        return states[0], states[1], states[2].squeeze(), states[3], states[4], states[5], states[6], states[7], states[8]





class MALA_Within_Gibbs(gp.GaussianPlume, Priors):

    def __init__(self, grided, gaussianplume, data, log_posterior, priors, mh_params, gibbs_params, fixed):
        
        self.grided = grided
        self.gaussianplume = gaussianplume
        self.data = data
        self.log_posterior = log_posterior
        self.priors = priors
        self.mh_params = mh_params
        self.gibbs_params = gibbs_params
        self.mh_flat = ravel_pytree(mh_params)[0]
        self.mh_unflat_func = ravel_pytree(mh_params)[1]
        self.fixed = fixed

        self.background_conditional_posterior = Gibbs_samplers(gaussianplume, data, priors).background_conditional_posterior
        self.measurement_error_var_conditional_posterior = Gibbs_samplers(gaussianplume, data, priors).measurement_error_var_conditional_posterior
        self.binary_indicator_Zi_conditional_posterior = Gibbs_samplers(gaussianplume, data, priors).binary_indicator_Zi_conditional_posterior

        self.glpi = MWG_tools(False, grided, gaussianplume, data, log_posterior, priors, mh_params, fixed).glpi
        self.mwg_scan = MWG_tools(False, grided, gaussianplume, data, log_posterior, priors, mh_params, fixed).mwg_scan



    def mala_advance(self, x, dt, sigma_squared, betas, ss_var, ss_mean, A):
        return  x + 0.5*dt*ravel_pytree(self.glpi(x, sigma_squared, betas, ss_var, ss_mean, A))[0]



    def mala_rprop(self, key, x, dt, sigma_squared, betas, ss_var, ss_mean, A):
        return self.mala_advance(x, dt, sigma_squared, betas, ss_var, ss_mean, A) + jnp.sqrt(dt)*jax.random.normal(key, [len(x)])



    def mala_dprop(self, prop, x, dt, sigma_squared, betas, ss_var, ss_mean, A):
        return jnp.sum(tfd.Normal(loc=self.mala_advance(x, dt, sigma_squared, betas, ss_var, ss_mean, A), scale=jnp.sqrt(dt)).log_prob(prop))



    def mala_step(self, updates, key):
        [x, sigma_squared, background, ll, dt, sum_accept, z_count, new_grad_squared_sum, max_dist, iteration] = updates
        #          Updates          #
        betas = jnp.repeat(background, 10_000).reshape(-1,1)
        if self.grided == True:
            A = self.gaussianplume.temporal_grided_coupling_matrix(self.fixed, jnp.exp(self.mh_unflat_func(x)["log_tan_gamma_H"]), jnp.exp(self.mh_unflat_func(x)["log_tan_gamma_V"]), jnp.exp(self.mh_unflat_func(x)["log_b_H"]), jnp.exp(self.mh_unflat_func(x)["log_b_V"]))
        else:   
            A = self.gaussianplume.temporal_gridfree_coupling_matrix(self.fixed, jnp.exp(self.mh_unflat_func(x)["source_x"]), jnp.exp(self.mh_unflat_func(x)["source_y"]), jnp.exp(self.mh_unflat_func(x)["log_tan_gamma_H"]), jnp.exp(self.mh_unflat_func(x)["log_tan_gamma_V"]), jnp.exp(self.mh_unflat_func(x)["log_b_H"]), jnp.exp(self.mh_unflat_func(x)["log_b_V"]))
        z = self.binary_indicator_Zi_conditional_posterior(self.mh_unflat_func(x)["log_s"], key)
        ss_var = jnp.where(z==0, self.priors.log_spike_var, self.priors.log_slab_var).reshape(-1,1)
        ss_mean = jnp.where(z==0, self.priors.log_spike_mean, self.priors.log_slab_mean).reshape(-1,1)
        #           Gibbs           #
        sigma_squared = self.measurement_error_var_conditional_posterior(self.data, A, betas, jnp.exp(self.mh_unflat_func(x)["log_s"]), key)
        background = self.background_conditional_posterior(jnp.exp(self.mh_unflat_func(x)["log_s"]), A, sigma_squared, self.priors.variance_log_background_prior, key).reshape(-1,1)
        betas = jnp.repeat(background, 10_000).reshape(-1,1)
        #           MALA            #
        ll = self.log_posterior(self.mh_unflat_func(x), sigma_squared, betas, ss_var, ss_mean, self.data, self.priors, A)
        prop = self.mala_rprop(key, x, dt, sigma_squared, betas, ss_var, ss_mean, A)
        if self.grided == True:
            A_prop = self.gaussianplume.temporal_grided_coupling_matrix(self.fixed, jnp.exp(self.mh_unflat_func(prop)["log_tan_gamma_H"]), jnp.exp(self.mh_unflat_func(prop)["log_tan_gamma_V"]), jnp.exp(self.mh_unflat_func(prop)["log_b_H"]), jnp.exp(self.mh_unflat_func(prop)["log_b_V"]))
        else:   
            A_prop = self.gaussianplume.temporal_gridfree_coupling_matrix(self.fixed, jnp.exp(self.mh_unflat_func(prop)["source_x"]), jnp.exp(self.mh_unflat_func(prop)["source_y"]), jnp.exp(self.mh_unflat_func(prop)["log_tan_gamma_H"]), jnp.exp(self.mh_unflat_func(prop)["log_tan_gamma_V"]), jnp.exp(self.mh_unflat_func(prop)["log_b_H"]), jnp.exp(self.mh_unflat_func(prop)["log_b_V"]))
        lp = self.log_posterior(self.mh_unflat_func(prop), sigma_squared, betas, ss_var, ss_mean, self.data, self.priors, A_prop)
        a = lp - ll + self.mala_dprop(x, prop, dt, sigma_squared, betas, ss_var, ss_mean, A_prop) - self.mala_dprop(prop, x, dt, sigma_squared, betas, ss_var, ss_mean, A)
        accept = (jnp.log(jax.random.uniform(key)) < a)
        new_x, new_ll, new_A = jnp.where(accept, prop, x), jnp.where(accept, lp, ll), jnp.where(accept, A_prop, A)
        #          Updates          #
        sum_accept += jnp.min(jnp.array([1, jnp.exp(a)]))
        z_count += z
        max_dist = jnp.maximum(max_dist,  jnp.sqrt(jnp.square(new_x - self.mh_flat)))
        gradients = ravel_pytree(self.glpi(new_x, sigma_squared, betas, ss_var, ss_mean, new_A))[0]
        new_grad_squared_sum += gradients**2
        dt = max_dist / jnp.sqrt(new_grad_squared_sum + 1e-16)

        updates = [new_x, sigma_squared, background, new_ll, dt, sum_accept, z_count, new_grad_squared_sum, max_dist, iteration]

        return updates, updates



    def mala_chains(self, Gibbs_init, MH_init, iters, r_eps):
        t1 = time.time()
        mala_chains = self.mwg_scan(self.mala_step, Gibbs_init, MH_init, iters, r_eps)
        t2 = time.time()
        running_time = t2-t1
        print("Running time MALA within Gibbs: " + str(round(running_time // 60)) + " minutes " + str(round(running_time % 60)) + " seconds")

        if self.grided == True:
            MALA_within_Gibbs_traces = {
                "b_H": jnp.exp(mala_chains[0][:,0]),
                "b_V": jnp.exp(mala_chains[0][:,1]),
                "background" : mala_chains[2],
                "s": jnp.exp(mala_chains[0][:,2:2+len(self.mh_unflat_func(MH_init)["log_s"])]),
                "sigma_squared": mala_chains[1],
                "tan_gamma_H": jnp.exp(mala_chains[0][:,-2]),
                "tan_gamma_V": jnp.exp(mala_chains[0][:,-1]),
                "ll": mala_chains[3],
                "dt": mala_chains[4],
                "acceptance_rate": mala_chains[5]/jnp.arange(1,iters+1),
                "z_count": mala_chains[6],
                "new_grad_squared_sum": mala_chains[7],
                "max_dist": mala_chains[8],
            }
        else:
            MALA_within_Gibbs_traces = {
                "b_H": jnp.exp(mala_chains[0][:,0]),
                "b_V": jnp.exp(mala_chains[0][:,1]),
                "background" : mala_chains[2],
                "s": jnp.exp(mala_chains[0][:,2:2+len(self.mh_unflat_func(MH_init)["log_s"])]),
                "sigma_squared": mala_chains[1],
                "tan_gamma_H": jnp.exp(mala_chains[0][:,2+len(self.mh_unflat_func(MH_init)["log_s"])]),
                "tan_gamma_V": jnp.exp(mala_chains[0][:,2+len(self.mh_unflat_func(MH_init)["log_s"])+1]),
                "source_x": jnp.exp(mala_chains[0][:,2+len(self.mh_unflat_func(MH_init)["log_s"])+2:2+len(self.mh_unflat_func(MH_init)["log_s"])+4]),
                "source_y": jnp.exp(mala_chains[0][:,2+len(self.mh_unflat_func(MH_init)["log_s"])+4:2+len(self.mh_unflat_func(MH_init)["log_s"])+6]),
                "ll": mala_chains[3],
                "dt": mala_chains[4],
                "acceptance_rate": mala_chains[5]/jnp.arange(1,iters+1),
                "z_count": mala_chains[6],
                "new_grad_squared_sum": mala_chains[7],
                "max_dist": mala_chains[8],
            }
        return MALA_within_Gibbs_traces






class Manifold_MALA_Within_Gibbs(gp.GaussianPlume, Priors):

    def __init__(self, grided, gaussianplume, data, log_posterior, priors, mh_params, gibbs_params, fixed):
        
        self.grided = grided
        self.gaussianplume = gaussianplume
        self.data = data
        self.log_posterior = log_posterior
        self.priors = priors
        self.mh_params = mh_params
        self.gibbs_params = gibbs_params
        self.mh_flat = ravel_pytree(mh_params)[0]
        self.mh_unflat_func = ravel_pytree(mh_params)[1]
        self.fixed = fixed

        self.background_conditional_posterior = Gibbs_samplers(gaussianplume, data, priors).background_conditional_posterior
        self.measurement_error_var_conditional_posterior = Gibbs_samplers(gaussianplume, data, priors).measurement_error_var_conditional_posterior
        self.binary_indicator_Zi_conditional_posterior = Gibbs_samplers(gaussianplume, data, priors).binary_indicator_Zi_conditional_posterior

        self.glpi = MWG_tools(True, grided, gaussianplume, data, log_posterior, priors, mh_params, fixed).glpi
        self.mwg_scan = MWG_tools(True, grided, gaussianplume, data, log_posterior, priors, mh_params, fixed).mwg_scan



    def inverse_hessian(self, x, sigma_squared, betas, ss_var, ss_mean, A):
        hess = jax.jacfwd(jax.jacrev(self.log_posterior))(self.mh_unflat_func(x), sigma_squared, betas, ss_var, ss_mean, self.data, self.priors, A)
        b_H_hess = jnp.array([hess["log_b_H"]["log_b_H"]])
        b_V_hess = jnp.array([hess["log_b_V"]["log_b_V"]])
        log_s_hess = jnp.diag(hess["log_s"]["log_s"].reshape(len(self.mh_unflat_func(x)["log_s"]),len(self.mh_unflat_func(x)["log_s"])))
        tan_gamma_H_hess = jnp.array([hess["log_tan_gamma_H"]["log_tan_gamma_H"]])
        tan_gamma_V_hess = jnp.array([hess["log_tan_gamma_V"]["log_tan_gamma_V"]])
        if self.grided == False:
            source_x_hess = jnp.diag(hess["source_x"]["source_x"].reshape(len(self.mh_unflat_func(x)["source_x"]),len(self.mh_unflat_func(x)["source_x"])))
            source_y_hess = jnp.diag(hess["source_y"]["source_y"].reshape(len(self.mh_unflat_func(x)["source_y"]),len(self.mh_unflat_func(x)["source_y"])))
            hessian = jnp.diag(jnp.concatenate([b_H_hess, b_V_hess, log_s_hess, tan_gamma_H_hess, tan_gamma_V_hess, source_x_hess, source_y_hess]))
        else:
            hessian = jnp.diag(jnp.concatenate([b_H_hess, b_V_hess, log_s_hess, tan_gamma_H_hess, tan_gamma_V_hess]))
        inv_hessian = jnp.linalg.inv(hessian)

        return inv_hessian



    def sqrt_inv_hess(self, x, sigma_squared, betas, ss_var, ss_mean, A):
        inv_hessian = self.inverse_hessian(x, sigma_squared, betas, ss_var, ss_mean, A)
        # Compute the eigenvalues and eigenvectors of the inverse Hessian
        eigenvalues, eigenvectors = jnp.linalg.eigh(inv_hessian)
        # Compute the square root of the absolute value of the eigenvalues
        sqrt_eigenvalues = jnp.sqrt(jnp.abs(eigenvalues))
        # Compute the square root of the inverse Hessian
        sqrt_inv_hessian = eigenvectors @ jnp.diag(sqrt_eigenvalues) @ jnp.linalg.inv(eigenvectors)

        return sqrt_inv_hessian



    def manifold_mala_advance(self, x, dt, sigma_squared, betas, ss_var, ss_mean, A):
        return  x + 0.5*dt*jnp.matmul(self.inverse_hessian(x, sigma_squared, betas, ss_var, ss_mean, A), ravel_pytree(self.glpi(x, sigma_squared, betas, ss_var, ss_mean, A))[0])



    def manifold_mala_rprop(self, key, x, dt, sigma_squared, betas, ss_var, ss_mean, A):
        return self.manifold_mala_advance(x, dt, sigma_squared, betas, ss_var, ss_mean, A) + jnp.sqrt(dt)*self.sqrt_inv_hess(x, sigma_squared, betas, ss_var, ss_mean, A)@jax.random.normal(key, [len(x)])



    def manifold_mala_dprop(self, prop, x, dt, sigma_squared, betas, ss_var, ss_mean, A):
        return jnp.sum(tfd.Normal(loc=self.manifold_mala_advance(x, dt, sigma_squared, betas, ss_var, ss_mean, A), scale=jnp.diag(jnp.sqrt(dt)*self.sqrt_inv_hess(x, sigma_squared, betas, ss_var, ss_mean, A))).log_prob(prop))



    def manifold_mala_step(self, updates, key):
        [x, sigma_squared, background, ll, dt, sum_accept, z_count, new_grad_squared_sum, max_dist, iteration] = updates
        #          Updates          #        
        betas = jnp.repeat(background, 10_000).reshape(-1,1)
        if self.grided == True:
            A = self.gaussianplume.temporal_grided_coupling_matrix(self.fixed, jnp.exp(self.mh_unflat_func(x)["log_tan_gamma_H"]), jnp.exp(self.mh_unflat_func(x)["log_tan_gamma_V"]), jnp.exp(self.mh_unflat_func(x)["log_b_H"]), jnp.exp(self.mh_unflat_func(x)["log_b_V"]))
        else:   
            A = self.gaussianplume.temporal_gridfree_coupling_matrix(self.fixed, jnp.exp(self.mh_unflat_func(x)["source_x"]), jnp.exp(self.mh_unflat_func(x)["source_y"]), jnp.exp(self.mh_unflat_func(x)["log_tan_gamma_H"]), jnp.exp(self.mh_unflat_func(x)["log_tan_gamma_V"]), jnp.exp(self.mh_unflat_func(x)["log_b_H"]), jnp.exp(self.mh_unflat_func(x)["log_b_V"]))
        z = self.binary_indicator_Zi_conditional_posterior(self.mh_unflat_func(x)["log_s"], key)
        ss_var = jnp.where(z==0, self.priors.log_spike_var, self.priors.log_slab_var).reshape(-1,1)
        ss_mean = jnp.where(z==0, self.priors.log_spike_mean, self.priors.log_slab_mean).reshape(-1,1)

        #           Gibbs           #
        sigma_squared = self.measurement_error_var_conditional_posterior(self.data, A, betas, jnp.exp(self.mh_unflat_func(x)["log_s"]), key)
        background = self.background_conditional_posterior(jnp.exp(self.mh_unflat_func(x)["log_s"]), A, sigma_squared, self.priors.variance_log_background_prior, key).reshape(-1,1)
        betas = jnp.repeat(background, 10_000).reshape(-1,1)
        #          Manifold MALA            #
        ll = self.log_posterior(self.mh_unflat_func(x), sigma_squared, betas, ss_var, ss_mean, self.data, self.priors, A)
        prop = self.manifold_mala_rprop(key, x, dt, sigma_squared, betas, ss_var, ss_mean, A)
        if self.grided == True:
            A_prop = self.gaussianplume.temporal_grided_coupling_matrix(self.fixed, jnp.exp(self.mh_unflat_func(prop)["log_tan_gamma_H"]), jnp.exp(self.mh_unflat_func(prop)["log_tan_gamma_V"]), jnp.exp(self.mh_unflat_func(prop)["log_b_H"]), jnp.exp(self.mh_unflat_func(prop)["log_b_V"]))
        else:   
            A_prop = self.gaussianplume.temporal_gridfree_coupling_matrix(self.fixed, jnp.exp(self.mh_unflat_func(prop)["source_x"]), jnp.exp(self.mh_unflat_func(prop)["source_y"]), jnp.exp(self.mh_unflat_func(prop)["log_tan_gamma_H"]), jnp.exp(self.mh_unflat_func(prop)["log_tan_gamma_V"]), jnp.exp(self.mh_unflat_func(prop)["log_b_H"]), jnp.exp(self.mh_unflat_func(prop)["log_b_V"]))
        lp = self.log_posterior(self.mh_unflat_func(prop), sigma_squared, betas, ss_var, ss_mean, self.data, self.priors, A_prop)
        a = lp - ll + self.manifold_mala_dprop(x, prop, dt, sigma_squared, betas, ss_var, ss_mean, A_prop) - self.manifold_mala_dprop(prop, x, dt, sigma_squared, betas, ss_var, ss_mean, A)
        accept = (jnp.log(jax.random.uniform(key)) < a)
        new_x, new_ll = jnp.where(accept, prop, x), jnp.where(accept, lp, ll)
        #          Updates         #
        sum_accept += jnp.min(jnp.array([1, jnp.exp(a)]))
        z_count += z
        iteration += 1
        dt = dt * ( 1 + 0.1 * ( sum_accept / iteration - 0.574 ) )

        updates = [new_x, sigma_squared, background, new_ll, dt, sum_accept, z_count, new_grad_squared_sum, max_dist, iteration]

        return updates, updates



    def manifold_mala_chains(self, Gibbs_init, MH_init, iters, r_eps):
        t1 = time.time()
        manifold_mala_chains = self.mwg_scan(self.manifold_mala_step, Gibbs_init, MH_init, iters, r_eps)
        t2 = time.time()
        running_time = t2-t1
        print("Running time Manifold MALA within Gibbs: " + str(round(running_time // 60)) + " minutes " + str(round(running_time % 60)) + " seconds")

        if self.grided == True:
            Manifold_MALA_within_Gibbs_traces = {
                "b_H": jnp.exp(manifold_mala_chains[0][:,0]),
                "b_V": jnp.exp(manifold_mala_chains[0][:,1]),
                "background" : manifold_mala_chains[2],
                "s": jnp.exp(manifold_mala_chains[0][:,2:2+len(self.mh_unflat_func(MH_init)["log_s"])]),
                "sigma_squared": manifold_mala_chains[1],
                "tan_gamma_H": jnp.exp(manifold_mala_chains[0][:,-2]),
                "tan_gamma_V": jnp.exp(manifold_mala_chains[0][:,-1]),
                "ll": manifold_mala_chains[3],
                "dt": manifold_mala_chains[4],
                "acceptance_rate": manifold_mala_chains[5]/jnp.arange(1,iters+1),
                "z_count": manifold_mala_chains[6],
                "new_grad_squared_sum": manifold_mala_chains[7],
                "max_dist": manifold_mala_chains[8],
            }
        else:
            Manifold_MALA_within_Gibbs_traces = {
                "b_H": jnp.exp(manifold_mala_chains[0][:,0]),
                "b_V": jnp.exp(manifold_mala_chains[0][:,1]),
                "background" : manifold_mala_chains[2],
                "s": jnp.exp(manifold_mala_chains[0][:,2:2+len(self.mh_unflat_func(MH_init)["log_s"])]),
                "sigma_squared": manifold_mala_chains[1],
                "tan_gamma_H": jnp.exp(manifold_mala_chains[0][:,2+len(self.mh_unflat_func(MH_init)["log_s"])]),
                "tan_gamma_V": jnp.exp(manifold_mala_chains[0][:,2+len(self.mh_unflat_func(MH_init)["log_s"])+1]),
                "source_x": jnp.exp(manifold_mala_chains[0][:,2+len(self.mh_unflat_func(MH_init)["log_s"])+2:2+len(self.mh_unflat_func(MH_init)["log_s"])+4]),
                "source_y": jnp.exp(manifold_mala_chains[0][:,2+len(self.mh_unflat_func(MH_init)["log_s"])+4:2+len(self.mh_unflat_func(MH_init)["log_s"])+6]),
                "ll": manifold_mala_chains[3],
                "dt": manifold_mala_chains[4],
                "acceptance_rate": manifold_mala_chains[5]/jnp.arange(1,iters+1),
                "z_count": manifold_mala_chains[6],
                "new_grad_squared_sum": manifold_mala_chains[7],
                "max_dist": manifold_mala_chains[8],
            }
        return Manifold_MALA_within_Gibbs_traces






class Plots:

    def __init__(self, gaussianplume, truth):
        self.gaussianplume = gaussianplume
        self.truth = truth



    def true_source_location_emission_rate_chains(self, chains, save = False, format = "pdf"):
        plt.plot(chains["s"][:,jnp.where(self.truth[2]>0)[0]])
        plt.title("Source emission rate")
        plt.axhline(self.gaussianplume.atmospheric_state.emission_rate, color='red', linestyle='--')
        if save:
            plt.savefig("true_source_emission_rate." + format, dpi=300, bbox_inches="tight")
        return  plt.show()
    
    def true_source_location_emission_rate_density(self, chains, burn_in, save = False, format = "pdf"):
        sns.kdeplot(chains["s"][burn_in:,jnp.where(self.truth[2]>0)[0]].squeeze(), color='seagreen', label='MALA', fill=True, alpha=0.1)
        plt.axvline(self.gaussianplume.atmospheric_state.emission_rate, color='red', linestyle='--')
        plt.xlabel("Emission rate")
        plt.ylabel("Density")
        plt.title("Source emission rate")
        plt.legend()
        if save:
            plt.savefig("true source emission rate density." + format, dpi=300, bbox_inches="tight")
        return  plt.show()




    def source_location_emission_rate_chains(self, chains, source_number, save = False, format = "pdf"):
        plt.plot(chains["s"][:,source_number])
        plt.title("Source emission rate")
        plt.axhline(self.gaussianplume.atmospheric_state.emission_rate, color='red', linestyle='--')
        if save:
            plt.savefig("grid_free_source_emission_rate." + format, dpi=300, bbox_inches="tight")
        return  plt.show()
    
    def source_location_emission_rate_density(self, chains, source_number, burn_in, save = False, format = "pdf"):
        sns.kdeplot(chains["s"][burn_in:,source_number].squeeze(), color='seagreen', label='MALA', fill=True, alpha=0.1)
        plt.axvline(self.gaussianplume.atmospheric_state.emission_rate, color='red', linestyle='--')
        plt.xlabel("Emission rate")
        plt.ylabel("Density")
        plt.title("Source emission rate")
        plt.legend()
        if save:
            plt.savefig("true source emission rate density." + format, dpi=300, bbox_inches="tight")
        return  plt.show()




    def total_emission_rates_chains(self, chains, save = False, format = "pdf"):
        plt.plot(jnp.sum(chains["s"][:,:], axis=1))
        plt.title("Total Emissions Rates Over Grid")
        plt.axhline(self.gaussianplume.atmospheric_state.emission_rate, color='red', linestyle='--')
        if save:
            plt.savefig("total emission rates." + format, dpi=300, bbox_inches="tight")
        plt.show()

    def total_emission_rates_density(self, chains, burn_in, save = False, format = "pdf"):
        sns.kdeplot(jnp.sum(chains["s"][burn_in:,:], axis=1), color='seagreen', fill=True, alpha=0.1)
        plt.axvline(self.gaussianplume.atmospheric_state.emission_rate, color='red', linestyle='--')
        plt.xlabel("Total Emissions Rate")
        plt.ylabel("Density")
        plt.title("Total Emissions Rates Over Grid")
        plt.legend()
        if save:
            plt.savefig("total emission rates density." + format, dpi=300, bbox_inches="tight")
        return  plt.show()



    def zero_emission_rates_chains(self, chains, save = False, format = "pdf"):
        plt.plot(jnp.delete(chains["s"][:], jnp.where(self.truth[2]>0)[0], axis=1))
        plt.title("Zero Emissions source")
        if save:
            plt.savefig("Zero emissions source." + format, dpi=300, bbox_inches="tight")
        plt.show()



    def measurement_error_var_chains(self, chains, save = False, format = "pdf"):
        plt.plot(chains["sigma_squared"][:])
        plt.title("Sigma squared")
        plt.axhline(self.gaussianplume.sensors_settings.measurement_error_var, color='red', linestyle='--')
        if save:
            plt.savefig("sigma squared." + format, dpi=300, bbox_inches="tight")
        plt.show()

    def measurement_error_var_density(self, chains, burn_in, save = False, format = "pdf"):
        sns.kdeplot(chains["sigma_squared"][burn_in:], color='seagreen', fill=True, alpha=0.1)
        plt.axvline(self.gaussianplume.sensors_settings.measurement_error_var, color='red', linestyle='--')
        plt.xlabel("Sigma squared")
        plt.ylabel("Density")
        plt.title("Sigma squared")
        plt.legend()
        if save:
            plt.savefig("sigma squared density." + format, dpi=300, bbox_inches="tight")
        return  plt.show()



    def background_chains(self, chains, save = False, format = "pdf"):
        plt.plot(chains["background"][:])
        plt.title("Background")
        for i in range(self.gaussianplume.sensors_settings.sensor_number):
            plt.axhline(jnp.unique(self.truth[4])[i], color='red', linestyle='--')
        if save:
            plt.savefig("background." + format, dpi=300, bbox_inches="tight")
        plt.show()
    
    def background_density(self, chains, burn_in, save = False, format = "pdf"):
        sns.kdeplot(chains["background"][burn_in:], color='seagreen', fill=True, alpha=0.1)
        for i in range(self.gaussianplume.sensors_settings.sensor_number):
            plt.axvline(jnp.unique(self.truth[4])[i], color='red', linestyle='--')
        plt.xlabel("Background")
        plt.ylabel("Density")
        plt.title("Background")
        plt.legend()
        if save:
            plt.savefig("background density." + format, dpi=300, bbox_inches="tight")
        return  plt.show()



    def tan_gamma_H_chains(self, chains, save = False, format = "pdf"):
        plt.plot(chains["tan_gamma_H"][:])
        plt.title("tan_gamma_H")
        plt.axhline(jnp.tan(self.gaussianplume.atmospheric_state.horizontal_angle), color='red', linestyle='--')
        if save:
            plt.savefig("tan_gamma_H." + format, dpi=300, bbox_inches="tight")
        plt.show()

    def tan_gamma_H_density(self, chains, burn_in, save = False, format = "pdf"):
        sns.kdeplot(chains["tan_gamma_H"][burn_in:], color='seagreen', fill=True, alpha=0.1)
        plt.axvline(jnp.tan(self.gaussianplume.atmospheric_state.horizontal_angle), color='red', linestyle='--')
        plt.xlabel("tan_gamma_H")
        plt.ylabel("Density")
        plt.title("tan_gamma_H")
        plt.legend()
        if save:
            plt.savefig("tan_gamma_H density." + format, dpi=300, bbox_inches="tight")
        return  plt.show()
    


    def tan_gamma_V_chains(self, chains, save = False, format = "pdf"):
        plt.plot(chains["tan_gamma_V"][:])
        plt.title("tan_gamma_V")
        plt.axhline(jnp.tan(self.gaussianplume.atmospheric_state.vertical_angle), color='red', linestyle='--')
        if save:
            plt.savefig("tan_gamma_V." + format, dpi=300, bbox_inches="tight")
        plt.show()

    def tan_gamma_V_density(self, chains, burn_in, save = False, format = "pdf"):
        sns.kdeplot(chains["tan_gamma_V"][burn_in:], color='seagreen', fill=True, alpha=0.1)
        plt.axvline(jnp.tan(self.gaussianplume.atmospheric_state.vertical_angle), color='red', linestyle='--')
        plt.xlabel("tan_gamma_V")
        plt.ylabel("Density")
        plt.title("tan_gamma_V")
        plt.legend()
        if save:
            plt.savefig("tan_gamma_V density." + format, dpi=300, bbox_inches="tight")
        return  plt.show()
    


    def b_H_chains(self, chains, save = False, format = "pdf"):
        plt.plot(chains["b_H"][:])
        plt.title("b_H")
        plt.axhline(self.gaussianplume.atmospheric_state.downwind_power_H, color='red', linestyle='--')
        if save:
            plt.savefig("b_H." + format, dpi=300, bbox_inches="tight")
        plt.show()

    def b_H_density(self, chains, burn_in, save = False, format = "pdf"):
        sns.kdeplot(chains["b_H"][burn_in:], color='seagreen', fill=True, alpha=0.1)
        plt.axvline(self.gaussianplume.atmospheric_state.downwind_power_H, color='red', linestyle='--')
        plt.xlabel("b_H")
        plt.ylabel("Density")
        plt.title("b_H")
        plt.legend()
        if save:
            plt.savefig("b_H density." + format, dpi=300, bbox_inches="tight")
        return  plt.show()
    


    def b_V_chains(self, chains, save = False, format = "pdf"):
        plt.plot(chains["b_V"][:])
        plt.title("b_V")
        plt.axhline(self.gaussianplume.atmospheric_state.downwind_power_V, color='red', linestyle='--')
        if save:
            plt.savefig("b_V." + format, dpi=300, bbox_inches="tight")
        plt.show()

    def b_V_density(self, chains, burn_in, save = False, format = "pdf"):
        sns.kdeplot(chains["b_V"][burn_in:], color='seagreen', fill=True, alpha=0.1)
        plt.axvline(self.gaussianplume.atmospheric_state.downwind_power_V, color='red', linestyle='--')
        plt.xlabel("b_V")
        plt.ylabel("Density")
        plt.title("b_V")
        plt.legend()
        if save:
            plt.savefig("b_V density." + format, dpi=300, bbox_inches="tight")
        return  plt.show()
    


    def source_position_chains(self, chains, save = False, format = "pdf"):
        plt.plot(chains["source_x"][:], chains["source_y"][:])
        plt.title("Source Position")
        plt.scatter(self.gaussianplume.source_location.source_location_x, self.gaussianplume.source_location.source_location_y, color='red', marker='x')
        if save:
            plt.savefig("source position." + format, dpi=300, bbox_inches="tight")
        plt.show()
        
    def source_position_density(self, chains, burn_in, save = False, format = "pdf"):
        sns.kdeplot(x=chains["source_x"][burn_in:, 0], y=chains["source_y"][burn_in:, 0], cmap='viridis', fill=True, alpha=0.8)
        sns.kdeplot(x=chains["source_x"][burn_in:, 1], y=chains["source_y"][burn_in:, 1], cmap='rainbow', fill=True, alpha=0.8)
        plt.scatter(self.gaussianplume.source_location.source_location_x, self.gaussianplume.source_location.source_location_y, color='red', marker='x')
        plt.xlabel("Source x")
        plt.ylabel("Source y")
        plt.title("Source Position Density")
        if save:
            plt.savefig("source position density." + format, dpi=300, bbox_inches="tight")
        
        return plt.show()



    def emission_rates_heatmap(self, chains, burn_in, save = False, format = "pdf"):
        df = pd.DataFrame(jnp.mean(chains["s"][burn_in:,:], axis=0).reshape(len(self.gaussianplume.grid.x), len(self.gaussianplume.grid.y)))
        plt.figure(figsize=(15, 10))
        if (len(self.gaussianplume.grid.x) > 10) and (len(self.gaussianplume.grid.y) > 10):
            ax = sns.heatmap(df, cmap="viridis", xticklabels=round(len(self.gaussianplume.grid.x)/10), yticklabels=round(len(self.gaussianplume.grid.y)/10))
            ax.set_xticks(jnp.arange(0, len(self.gaussianplume.grid.x) + 1, round(len(self.gaussianplume.grid.x)/10)))
            ax.set_yticks(jnp.arange(0, len(self.gaussianplume.grid.y) + 1, round(len(self.gaussianplume.grid.y)/10)))
            ax.set_yticklabels([round(i,1) for i in jnp.arange(self.gaussianplume.grid.y_range[0], self.gaussianplume.grid.y_range[1] + self.gaussianplume.grid.dy + 1, self.gaussianplume.grid.y_range[1]/10)], rotation=0)
            ax.set_xticklabels([round(i,1) for i in jnp.arange(self.gaussianplume.grid.x_range[0], self.gaussianplume.grid.x_range[1] + self.gaussianplume.grid.dx + 1, self.gaussianplume.grid.x_range[1]/10)], rotation=0)
        else:
            ax = sns.heatmap(df, cmap="viridis")
            ax.set_yticklabels(self.gaussianplume.grid.y)
            ax.set_xticklabels(self.gaussianplume.grid.x)
        ax.invert_yaxis()
        ax.scatter(float(self.gaussianplume.source_location.source_location_x/self.gaussianplume.grid.dx), float(self.gaussianplume.source_location.source_location_y/self.gaussianplume.grid.dy), marker='.', s=10_000, color='orange')
        colorbar = ax.collections[0].colorbar
        colorbar.set_label('Parts per million (PPM)')
        plt.title("Initial Gaussian Plume")
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        if save == True:
            plt.savefig("Initial Gaussian Plume." + format, dpi=300, bbox_inches="tight")

        return plt.show()



    def layerwise_dog_step_size_chains(self, chains, save = False, format = "pdf"):
        plt.plot(chains["dt"][:])
        plt.title("Layerwise Dog Step Size")
        if save:
            plt.savefig("layerwise dog step size." + format, dpi=300, bbox_inches="tight")
        plt.show()

    def step_size_chains(self, chains, save = False, format = "pdf"):
        plt.plot(chains["dt"][:])
        plt.title("Step Size")
        if save:
            plt.savefig("step size." + format, dpi=300, bbox_inches="tight")
        plt.show()



    def samples_acceptance_rate(self, chains, save = False, format = "pdf"):
        plt.plot(chains["acceptance_rate"][:])
        plt.title("Acceptance Rate")
        if save:
            plt.savefig("acceptance rate." + format, dpi=300, bbox_inches="tight")
        plt.show()



    def spike_slab_allocation(self, chains, save = False, format = "pdf"):
        plt.plot(chains["z_count"][:])
        plt.title("Spike Slab Allocation")
        if save:
            plt.savefig("spike slab allocation." + format, dpi=300, bbox_inches="tight")
        plt.show()



    def log_posterior_plot(self, chains, save = False, format = "pdf"):
        plt.plot(chains["ll"][:])
        plt.title("Log Posterior")
        if save:
            plt.savefig("log posterior." + format, dpi=300, bbox_inches="tight")
        plt.show()


    
