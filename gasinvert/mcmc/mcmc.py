import gasinvert.atmospheric_measurements as gp

import tensorflow_probability.substrates.jax as tfp
tfd = tfp.distributions
import jax.numpy as jnp
import jax
from jax import random
from jax.flatten_util import ravel_pytree
from dataclasses import dataclass
from jaxtyping import Float



__all__ = ['Priors', 'MALA_Within_Gibbs']

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



class MALA_Within_Gibbs(gp.GaussianPlume, Priors):

    def __init__(self, gaussianplume, data, log_posterior, priors, mh_params, gibbs_params, fixed):
        
        self.gaussianplume = gaussianplume
        self.data = data
        self.log_posterior = log_posterior
        self.priors = priors
        self.mh_params = mh_params
        self.gibbs_params = gibbs_params
        self.mh_flat = ravel_pytree(mh_params)[0]
        self.mh_unflat_func = ravel_pytree(mh_params)[1]
        self.fixed = fixed



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


    def glpi(self, x, sigma_squared, betas, ss_var, ss_mean):
        return jax.grad(self.log_posterior)(self.mh_unflat_func(x), sigma_squared, betas, ss_var, ss_mean, self.data, self.priors)


    def advance(self, x, dt, sigma_squared, betas, ss_var, ss_mean):
        return  x + 0.5*dt*ravel_pytree(self.glpi(x, sigma_squared, betas, ss_var, ss_mean))[0]

    def rprop(self, key, x, dt, sigma_squared, betas, ss_var, ss_mean):
        return self.advance(x, dt, sigma_squared, betas, ss_var, ss_mean) + jnp.sqrt(dt)*jax.random.normal(key, [len(x)])

    def dprop(self, prop, x, dt, sigma_squared, betas, ss_var, ss_mean):
        return jnp.sum(tfd.Normal(loc=self.advance(x, dt, sigma_squared, betas, ss_var, ss_mean), scale=jnp.sqrt(dt)).log_prob(prop))



    def step(self, updates, key):
        [x, ll, dt, ss_var, sum_accept, z_count, new_grad_squared_sum, max_dist, a, ss_mean, A, sigma_squared, background, betas] = updates
        #           Gibbs           #
        sigma_squared = self.measurement_error_var_conditional_posterior(self.data, A, betas, jnp.exp(self.mh_unflat_func(x)["log_s"]), key)
        background = self.background_conditional_posterior(jnp.exp(self.mh_unflat_func(x)["log_s"]), A, sigma_squared, self.priors.variance_log_background_prior, key).reshape(-1,1)
        betas = jnp.repeat(background, 1_000).reshape(-1,1)
        z = self.binary_indicator_Zi_conditional_posterior(self.mh_unflat_func(x)["log_s"], key)
        #           MALA            #
        ll = self.log_posterior(self.mh_unflat_func(x), sigma_squared, betas, ss_var, ss_mean, self.data, self.priors)
        prop = self.rprop(key, x, dt, sigma_squared, betas, ss_var, ss_mean)
        lp = self.log_posterior(self.mh_unflat_func(prop), sigma_squared, betas, ss_var, ss_mean, self.data, self.priors)
        a = lp - ll + self.dprop(x, prop, dt, sigma_squared, betas, ss_var, ss_mean) - self.dprop(prop, x, dt, sigma_squared, betas, ss_var, ss_mean)
        accept = (jnp.log(jax.random.uniform(key)) < a)
        new_x, new_ll = jnp.where(accept, prop, x), jnp.where(accept, lp, ll)
        #          Updates         #
        A_new = self.gaussianplume.temporal_coupling_matrix(self.fixed, jnp.exp(self.mh_unflat_func(new_x)["log_tan_gamma_H"]), jnp.exp(self.mh_unflat_func(new_x)["log_tan_gamma_V"]), jnp.exp(self.mh_unflat_func(new_x)["log_b_H"]), jnp.exp(self.mh_unflat_func(new_x)["log_b_V"]))
        sum_accept += jnp.min(jnp.array([1, jnp.exp(a)]))
        z_count += z
        ss_var = jnp.where(z==0, self.priors.log_spike_var, self.priors.log_slab_var).reshape(-1,1)
        ss_mean = jnp.where(z==0, self.priors.log_spike_mean, self.priors.log_slab_mean).reshape(-1,1)
        max_dist = jnp.maximum(max_dist,  jnp.sqrt(jnp.square(new_x - self.mh_flat)))
        gradients = ravel_pytree(self.glpi(new_x, sigma_squared, betas, ss_var, ss_mean))[0]
        new_grad_squared_sum += gradients**2
        dt = max_dist / jnp.sqrt(new_grad_squared_sum + 1e-16)

        updates = [new_x, new_ll, dt, ss_var, sum_accept, z_count, new_grad_squared_sum, max_dist, a, ss_mean, A_new, sigma_squared, background, betas]

        return updates, updates


    def mh_gibbs_scan(self, Gibbs_init, MH_init, iters, r_eps):
        key = random.PRNGKey(0)
        sigma_squared, background = Gibbs_init["sigma_squared"], Gibbs_init["background"].reshape(-1,1)
        betas = jnp.repeat(background, 1_000).reshape(-1,1)
        z = self.binary_indicator_Zi_conditional_posterior(self.mh_unflat_func(MH_init)["log_s"], key)
        ss_var = jnp.where(z==0, self.priors.log_spike_var, self.priors.log_slab_var).reshape(-1,1)
        ss_mean = jnp.where(z==0, self.priors.log_spike_mean, self.priors.log_slab_mean).reshape(-1,1)
        ll, gradi = jax.value_and_grad(self.log_posterior)(self.mh_unflat_func(MH_init), sigma_squared, betas, ss_var, ss_mean, self.data, self.priors)
        A = self.gaussianplume.temporal_coupling_matrix(self.fixed, jnp.exp(self.mh_unflat_func(MH_init)["log_tan_gamma_H"]), jnp.exp(self.mh_unflat_func(MH_init)["log_tan_gamma_V"]), jnp.exp(self.mh_unflat_func(MH_init)["log_b_H"]), jnp.exp(self.mh_unflat_func(MH_init)["log_b_V"]))
        sum_accept = 0
        z_count = jnp.zeros(36)
        gradients = ravel_pytree(gradi)[0]
        new_grad_squared_sum = gradients**2
        max_dist = jnp.full(len(MH_init), r_eps)
        dt = max_dist / jnp.sqrt(new_grad_squared_sum)
        a = 0
        _, states = jax.lax.scan(self.step, [MH_init, ll, dt, ss_var, sum_accept, z_count, new_grad_squared_sum, max_dist, a, ss_mean, A, sigma_squared, background, betas], jax.random.split(key, iters))

        return states[0], states[1], states[2], states[4], states[5], states[8], states[6], states[7], states[11], states[12].squeeze()
    

