import gasinvert.atmospheric_measurements as gp
import gasinvert.mcmc as mcmc

from jax.flatten_util import ravel_pytree
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import jax.numpy as jnp
from jax import config
import seaborn as sns
import numpy as np
import time
import jax
import tensorflow_probability.substrates.jax as tfp
tfd = tfp.distributions


config.update("jax_enable_x64", True)
params = {
        'legend.fontsize': '15',
        'axes.labelsize': '20',
        'axes.titlesize': '30',
        'xtick.labelsize':'x-large',
        'ytick.labelsize':'x-large',
        'figure.figsize': '10, 7',
        }
pylab.rcParams.update(params)



############################################################
# 1. Simulating Atmospheric Conditions and Sensors locations
############################################################

# Make a grid
grid = gp.Grid(
    x_range = (jnp.array(0.0), jnp.array(40.0)), 
    y_range = (jnp.array(0.0), jnp.array(40.0)),
    z_range= (jnp.array(0.0), jnp.array(0.0)), # change it to cube
    dx = jnp.array(1),
    dy = jnp.array(1),
    dz = jnp.array(1),
)

# Choose a source location
source_location = gp.SourceLocation(
    source_location_x = jnp.array(22),
    source_location_y = jnp.array(22),
    source_location_z = jnp.array(1.0),
)

# Define the wind field
wind_field = gp.WindField(
    initial_wind_speed = jnp.array(10.0),
    initial_wind_direction = jnp.array(-135.0),
    number_of_time_steps = jnp.array(100),
    time_step = jnp.array(1.0),
    wind_speed_temporal_std = jnp.array(1.0),
    wind_direction_temporal_std = jnp.array(30.0),
    wind_temporal_correlation = jnp.array(0.1),
    wind_speed_seed = 2,
    wind_direction_seed = jnp.array(4),
)

# Define an atmospheric state settings
atmospheric_state = gp.AtmosphericState(
    vertical_angle = jnp.deg2rad(jnp.array(25.0)),
    horizontal_angle = jnp.deg2rad(jnp.array(20.0)),
    emission_rate = jnp.array(1.6e-3),      
    half_width = jnp.array(2.0),
    max_abl = jnp.array(1000.0),
    background_mean = jnp.array(1.9),       
    background_std = jnp.array(1e-4),       
    background_seed = jnp.array(56),
    background_filter = "power-law",        
    Gaussian_filter_kernel = 1,              
    downwind_power_H = jnp.array(1.0),         
    downwind_power_V = jnp.array(0.75),         
)

# Define sensor settings
sensors_settings =  gp.SensorsSettings(
    sensor_number = jnp.array(12),
    measurement_error_var = jnp.array(1e-1),
    measurement_elevation = jnp.array(0.0),
    sensor_seed = jnp.array(5),
    measurement_error_seed = jnp.array(420),
    sensor_locations =  [[8, 8], [16,16], [20,20], [40,40], [16,8], [8,16],  [8,32], [32,8], [40,16], [16,40], [32,32], [28,28]], 
)


save = False



## Gaussian plume Class : ---------------------------------------------
gaussianplume = gp.GaussianPlume(grid, source_location, wind_field, atmospheric_state, sensors_settings)
gaussianplume.initial_gaussian_plume_plot(save, format='png')
gaussianplume.log_initial_gaussian_plume_plot(save, format='png')


# Background concentration Class : ---------------------------------------------
background = gp.BackgroundGas(grid, source_location, atmospheric_state)
background.background_plot(save, format='jpeg')


# Changing wind speed and direction : ---------------------------------------------
gaussianplume.wind_direction_plot(save, format='png')
gaussianplume.wind_speed_plot(save, format='png')


# Sensors Class : ---------------------------------------------
sensors = gp.Sensors(gaussianplume, background)
sensors.atmospheric_methane_and_sensors(save, format='png')
sensors.log_atmospheric_methane_and_sensors(save, format='png')


# Data
truth = sensors.temporal_sensors_measurements()
data = truth[0]











# MCMC Class : ---------------------------------------------
Gibbsparams = {
    'background': jnp.full(sensors_settings.sensor_number, 1.9),
    'sigma_squared':   0.05}
gibbs_flat, gibbs_unflat_func = ravel_pytree(Gibbsparams)

MHparams = {
    'log_b_H': jnp.log(0.95),    
    'log_b_V': jnp.log(0.7), 
    'log_s': jnp.log(truth[2] + 1e-9), 
    'log_tan_gamma_H':jnp.log(0.35),  
    'log_tan_gamma_V': jnp.log(0.5)}
mh_flat, mh_unflat_func = ravel_pytree(MHparams)

priors = mcmc.Priors(
    # slab allocation prior
    theta = 0.1,
    # Original scale SS
    spike_var = 1e-9,
    spike_mean = 0.0,
    slab_var = 1e-6,
    slab_mean = 1.6e-3,
    # Log scale SS
    log_spike_mean = -30.0,
    log_spike_var = 1.0,
    log_slab_mean = -6.5,
    log_slab_var = 5e-1,
    # Sigma squared
    sigma_squared_con = 1.5,
    sigma_squared_rate = 8.0,
    # Background
    mean_log_background_prior = jnp.log(1.9),
    variance_log_background_prior = 5e-2,
    # tan_gamma_H and tan_gamma_V
    tan_gamma_con = jnp.log(0.6),
    tan_gamma_rate = 0.1,
    # b_H and b_V
    b_mean = jnp.log(0.5),
    b_var = 0.1,
)

def log_posterior(params, sigma_squared, betas, ss_var, ss_mean, data, priors):
    """
    Returns the positive log posterior of the point sensors measurements model. 

    """
    emissions = params["log_s"].reshape(-1,1)

    A = gaussianplume.temporal_coupling_matrix(fixed, jnp.exp(params["log_tan_gamma_H"]), jnp.exp(params["log_tan_gamma_V"]), jnp.exp(params["log_b_H"]), jnp.exp(params["log_b_V"]))

    log_likelihood = tfd.Normal(loc = (jnp.matmul(A,jnp.exp(emissions))+ betas), \
                                scale= jnp.sqrt(sigma_squared)).log_prob(data)

    log_prior_tan_gamma_H = tfd.Normal(loc = priors.tan_gamma_con, scale = jnp.sqrt(priors.tan_gamma_rate)).log_prob(params["log_tan_gamma_H"])
    log_prior_tan_gamma_V = tfd.Normal(loc = priors.tan_gamma_con, scale = jnp.sqrt(priors.tan_gamma_rate)).log_prob(params["log_tan_gamma_V"])
    log_prior_b_H = tfd.Normal(loc = priors.b_mean, scale = jnp.sqrt(priors.b_var)).log_prob(params["log_b_H"])
    log_prior_b_V = tfd.Normal(loc = priors.b_mean, scale = jnp.sqrt(priors.b_var)).log_prob(params["log_b_V"])

    log_posterior_emission_rate = tfd.MultivariateNormalDiag(loc = ss_mean, scale_diag = jnp.sqrt(ss_var)).log_prob(emissions)

    log_posterior = jnp.sum(log_likelihood) + jnp.sum(log_prior_tan_gamma_H) + jnp.sum(log_prior_tan_gamma_V) + jnp.sum(log_prior_b_H) + jnp.sum(log_prior_b_V) \
                    + jnp.sum(log_posterior_emission_rate)

    return log_posterior

fixed = gaussianplume.fixed_objects_of_coupling_matrix()
iterations = 3_500
r_eps = 1e-5


mala_chains = mcmc.MALA_Within_Gibbs(gaussianplume, data, log_posterior, priors, MHparams, Gibbsparams, fixed).mala_chains(Gibbsparams, mh_flat, iterations, r_eps)


plots = mcmc.Plots(gaussianplume, truth)
plots.true_source_location_emission_rate_chains(mala_chains)
plots.true_source_location_emission_rate_density(mala_chains, burn_in=0)
plots.true_source_location_emission_rate_chains(mala_chains)
plots.true_source_location_emission_rate_density(mala_chains, burn_in=0)
plots.zero_emission_rates_chains(mala_chains)
plots.measurement_error_var_chains(mala_chains)
plots.measurement_error_var_density(mala_chains, burn_in=0)
plots.background_chains(mala_chains)
plots.background_density(mala_chains, burn_in=0)
plots.tan_gamma_H_chains(mala_chains)
plots.tan_gamma_H_density(mala_chains, burn_in=0)  
plots.tan_gamma_V_chains(mala_chains)
plots.tan_gamma_V_density(mala_chains, burn_in=0)
plots.b_H_chains(mala_chains)
plots.b_H_density(mala_chains, burn_in=0)
plots.b_V_chains(mala_chains)
plots.b_V_density(mala_chains, burn_in=0)

plots.layerwise_dog_step_size_chains(mala_chains)
plots.samples_acceptance_rate(mala_chains)
plots.spike_slab_allocation(mala_chains)







iterations = 2
r_eps = 1e-5

t3 = time.time()
manifold_mala_chains = mcmc.Manifold_MALA_Within_Gibbs(gaussianplume, data, log_posterior, priors, MHparams, Gibbsparams, fixed).manifold_mala_chains(Gibbsparams, mh_flat, iterations, r_eps)
t4 = time.time()

running_time_posi_MALA = t4-t3
print("Running time MALA within Gibbs: " + str(round(running_time_posi_MALA // 60)) + " minutes " + str(round(running_time_posi_MALA % 60)) + " seconds")

MALA_within_Gibbs_traces = {
    "b_H": jnp.exp(mh_gibbs_out[0][:,0]),
    "b_V": jnp.exp(mh_gibbs_out[0][:,1]),
    "background" : mh_gibbs_out[9],
    "s": jnp.exp(mh_gibbs_out[0][:,2:2+len(truth[2])]),
    "sigma_squared": mh_gibbs_out[8],
    "tan_gamma_H": jnp.exp(mh_gibbs_out[0][:,-2]),
    "tan_gamma_V": jnp.exp(mh_gibbs_out[0][:,-1]),
    "ll": mh_gibbs_out[1],
    "dt": mh_gibbs_out[2],
    "acceptance_rate": mh_gibbs_out[3]/np.arange(1,iterations+1),
    "z_count": mh_gibbs_out[4],
    "a": mh_gibbs_out[5],
    "new_grad_squared_sum": mh_gibbs_out[6],
    "max_dist": mh_gibbs_out[7],
}

