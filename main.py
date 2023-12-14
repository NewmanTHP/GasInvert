import gasinvert.atmospheric_measurements as gp
import gasinvert.mcmc as mcmc
import jax.numpy as jnp
import jax
from jax import config
import matplotlib.pylab as pylab
from jax.flatten_util import ravel_pytree
import tensorflow_probability.substrates.jax as tfp
tfd = tfp.distributions
import time
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


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


#############################################
# 1. Defining Atmospheric Conditions and Sensors 
#############################################

# Make a grid
grid = gp.Grid(
    x_range = (jnp.array(0.0), jnp.array(40.0)), 
    y_range = (jnp.array(0.0), jnp.array(40.0)),
    z_range= (jnp.array(0.0), jnp.array(0.0)), # change it to cube
    dx = jnp.array(4),
    dy = jnp.array(4),
    dz = jnp.array(1),
)

# Make a source location
source_location = gp.SourceLocation(
    source_location_x = jnp.array(1),
    source_location_y = jnp.array(16),
    source_location_z = jnp.array(1.0),
)

# Define the wind field
wind_field = gp.WindField(
    initial_wind_speed = jnp.array(10.0),
    initial_wind_direction = jnp.array(0.0),
    number_of_time_steps = jnp.array(1_000),
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
    emission_rate = jnp.array(1.6e-3),      #  realistic value 1.6e-3 kg/s (from Stockie 2011)
    half_width = jnp.array(2.0),
    max_abl = jnp.array(1000.0),
    background_mean = jnp.array(1.9),       #1.9        # equivalent to 1.9 ppm
    background_std = jnp.array(1e-4),       
    background_seed = jnp.array(56),
    background_filter = "power-law",        # filter to apply to the background concentration (e.g. Gaussian)
    Gaussian_filter_kernel = 1,              # controls Gaussian filter smoothness
    downwind_power_H = jnp.array(1.0),         # power of the downwind in wind sigma
    downwind_power_V = jnp.array(0.75),         # power of the downwind in wind sigma
)

# Define sensor settings
sensors_settings =  gp.SensorsSettings(
    sensor_number = jnp.array(4),
    measurement_error_var = jnp.array(1e-1),
    measurement_elevation = jnp.array(0.0),
    sensor_seed = jnp.array(5),
    measurement_error_seed = jnp.array(420),
    sensor_locations =  [[8, 8], [20,20], [8,32], [32,20]],
)


save = False



## Gaussian plume Class : ---------------------------------------------
gaussianplume = gp.GaussianPlume(grid, source_location, wind_field, atmospheric_state, sensors_settings)

gaussianplume.initial_gaussian_plume_plot(save, format='png')
gaussianplume.log_initial_gaussian_plume_plot(save, format='png')

# Background concentration Class : ---------------------------------------------
background = gp.BackgroundGas(grid, source_location, atmospheric_state)

background.background_plot(save, format='png')

# Changing wind speed and direction : ---------------------------------------------

gaussianplume.wind_direction_plot(save, format='png')
gaussianplume.wind_speed_plot(save, format='png')

# Sensors Class : ---------------------------------------------
sensors = gp.Sensors(gaussianplume, background)

sensors.atmospheric_methane_and_sensors(save, format='png')
sensors.log_atmospheric_methane_and_sensors(save, format='png')

# DATA
truth = sensors.temporal_sensors_measurements()
data = truth[0]











# MCMC Class : ---------------------------------------------

Gibbsparams = {
    'background': jnp.full(sensors_settings.sensor_number, 1.9),
    'sigma_squared':   0.05,   
}

gibbs_flat, gibbs_unflat_func = ravel_pytree(Gibbsparams)

MHparams = {
    'log_b_H': jnp.log(0.95),    
    'log_b_V': jnp.log(0.7), 
    'log_s': jnp.log(truth[2] + 1e-9), 
    'log_tan_gamma_H':jnp.log(0.5),  
    'log_tan_gamma_V': jnp.log(0.5), 
}

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


fixed = gaussianplume.fixed_objects_of_coupling_matrix()
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


iterations = 200
r_eps = 1e-5

t1 = time.time()
mh_gibbs_out = mcmc.MALA_Within_Gibbs(gaussianplume, data, log_posterior, priors, MHparams, Gibbsparams, fixed).mh_gibbs_scan(Gibbsparams, mh_flat, iterations, r_eps)
t2 = time.time()

running_time_posi_MALA = t2-t1
print("Running time MALA within Gibbs: " + str(round(running_time_posi_MALA // 60)) + " minutes " + str(round(running_time_posi_MALA % 60)) + " seconds")

MALA_within_Gibbs_traces = {
    "b_H": jnp.exp(mh_gibbs_out[0][:,0]),
    "b_V": jnp.exp(mh_gibbs_out[0][:,1]),
    "background" : mh_gibbs_out[9],
    "s": jnp.exp(mh_gibbs_out[0][:,2:2+36]),
    "sigma_squared": mh_gibbs_out[8],
    "tan_gamma_H": jnp.exp(mh_gibbs_out[0][:,2]),
    "tan_gamma_V": jnp.exp(mh_gibbs_out[0][:,3]),
    "ll": mh_gibbs_out[1],
    "dt": mh_gibbs_out[2],
    "acceptance_rate": mh_gibbs_out[3]/np.arange(1,iterations+1),
    "z_count": mh_gibbs_out[4],
    "a": mh_gibbs_out[5],
    "new_grad_squared_sum": mh_gibbs_out[6],
    "max_dist": mh_gibbs_out[7],
}


# 2.3.1 Source emission rate #

plt.plot(MALA_within_Gibbs_traces["s"][:,2])
plt.title("Source emission rate")
plt.axhline(1.6e-3, color='red', linestyle='--')
# plt.savefig("tuned 0.001 m-MALA Source emission traceplots after burn-in.pdf", dpi=300, bbox_inches="tight")
plt.show()

sns.kdeplot(MALA_within_Gibbs_traces["s"][2_000:,3].squeeze(), color='seagreen', label='MALA', fill=True, alpha=0.1)
plt.axvline(state.emission_rate, color='red', linestyle='--')
plt.xlabel("Emission rate")
plt.ylabel("Density")
plt.title("Source emission rate")
plt.legend()
# plt.savefig("sigma squared mala compare density scena 2.pdf", dpi=300, bbox_inches="tight")
plt.show()



# 2.3.2 Total Emissions Rates Over Grid #

plt.plot(jnp.sum(MALA_within_Gibbs_traces["s"][:,:], axis=1))
plt.title("Total Emissions Rates Over Grid")
plt.axhline(1.6e-3, color='red', linestyle='--')
if save==True:
    plt.savefig("tuned 0.001 m-MALA total Emission traceplots.pdf", dpi=300, bbox_inches="tight")
plt.show()

sns.kdeplot(jnp.sum(MALA_within_Gibbs_traces["s"][:,:], axis=1).squeeze(), color='seagreen', label='MALA', fill=True, alpha=0.1)
plt.axvline(state.emission_rate, color='red', linestyle='--')
plt.xlabel("Total Emission rate")
plt.ylabel("Density")
plt.title("Total Emission rate")
plt.legend()
# plt.savefig("sigma squared mala compare density scena 2.pdf", dpi=300, bbox_inches="tight")
plt.show()



# 2.3.3 Zero emission cells #

plt.plot(np.delete(MALA_within_Gibbs_traces["s"][:], 21, axis=1))
plt.title("Emissions source")
# plt.savefig("tuned 0.001 m-MALA Source emission traceplots after burn-in.pdf", dpi=300, bbox_inches="tight")
plt.show()



# 2.3.4 Sigma squared #

plt.plot(MALA_within_Gibbs_traces["sigma_squared"][:])
plt.title("sigma squared")
plt.axhline(model["true_sigma_squared"], color='red', linestyle='--')
# plt.savefig("tuned 0.001 m-MALA sigma squared traceplots.pdf", dpi=300, bbox_inches="tight")
plt.show()

sns.kdeplot(posi_MALA_traces["sigma_squared"][1000:].squeeze(), color='seagreen', label='MALA', fill=True, alpha=0.1)
plt.axvline(model["true_sigma_squared"], color='red', linestyle='--')
plt.xlabel("sigma squared")
plt.ylabel("Density")
plt.title("sigma squared")
plt.legend()
# plt.savefig("sigma squared mala compare density scena 2.pdf", dpi=300, bbox_inches="tight")
plt.show()



# 2.3.5 Background #

plt.plot(MALA_within_Gibbs_traces["background"][:,:])
plt.title("background")
plt.axhline(jnp.unique(model["background_true"])[0], color='red', linestyle='--')
# plt.savefig("tuned 0.001 m-MALA background traceplots.pdf", dpi=300, bbox_inches="tight")
plt.show()

sns.kdeplot(posi_MALA_traces["background"][1000:].squeeze(), color='seagreen', label='MALA', fill=True, alpha=0.1)
plt.axvline(jnp.unique(model["background_true"])[0], color='red', linestyle='--')
plt.xlabel("background")
plt.ylabel("Density")
plt.title("background")
plt.legend()
# plt.savefig("sigma squared mala compare density scena 2.pdf", dpi=300, bbox_inches="tight")
plt.show()



# 2.3.6 tan_gamma_H #

plt.plot(MALA_within_Gibbs_traces["tan_gamma_H"][:])
plt.title("tan_gamma_H")
plt.axhline(model["true_tan_gamma_H"], color='red', linestyle='--')
# plt.savefig("tuned 0.001 m-MALA tan_gamma traceplots.pdf", dpi=300, bbox_inches="tight")
plt.show()

sns.kdeplot(posi_MALA_traces["tan_gamma_H"][1000:].squeeze(), color='cyan', label='MALA', fill=True, alpha=0.1)
plt.axvline(model["true_tan_gamma_H"], color='red', linestyle='--')
plt.xlabel("tan_gamma_H")
plt.ylabel("Density")
plt.title("tan_gamma_H")
plt.legend()
# plt.savefig("sigma squared mala compare density scena 2.pdf", dpi=300, bbox_inches="tight")
plt.show()



# 2.3.7 tan_gamma_V #

plt.plot(MALA_within_Gibbs_traces["tan_gamma_V"][:])
plt.title("tan_gamma_V")
plt.axhline(model["true_tan_gamma_V"], color='red', linestyle='--')
# plt.savefig("tuned 0.001 m-MALA tan_gamma traceplots.pdf", dpi=300, bbox_inches="tight")
plt.show()

sns.kdeplot(posi_MALA_traces["tan_gamma_V"][1000:].squeeze(), color='cyan', label='MALA', fill=True, alpha=0.1)
plt.axvline(model["true_tan_gamma_V"], color='red', linestyle='--')
plt.xlabel("tan_gamma_V")
plt.ylabel("Density")
plt.title("tan_gamma_V")
plt.legend()
# plt.savefig("sigma squared mala compare density scena 2.pdf", dpi=300, bbox_inches="tight")
plt.show()



# 2.3.8 b_H #

plt.plot(MALA_within_Gibbs_traces["b_H"][:])
plt.title("b_H")
plt.axhline(model["true_b_H"], color='red', linestyle='--')
# plt.savefig("tuned 0.001 m-MALA b traceplots.pdf", dpi=300, bbox_inches="tight")
plt.show()

sns.kdeplot(posi_MALA_traces["b_H"][1000:].squeeze(), color='cyan', label='MALA', fill=True, alpha=0.1)
plt.axvline(model["true_b_H"], color='red', linestyle='--')
plt.xlabel("b_H")
plt.ylabel("Density")
plt.title("b_H")
plt.legend()
# plt.savefig("sigma squared mala compare density scena 2.pdf", dpi=300, bbox_inches="tight")
plt.show()



# 2.3.8 b_V #

plt.plot(MALA_within_Gibbs_traces["b_V"][:])
plt.title("b_V")
plt.axhline(model["true_b_V"], color='red', linestyle='--')
# plt.savefig("tuned 0.001 m-MALA b traceplots.pdf", dpi=300, bbox_inches="tight")
plt.show()

sns.kdeplot(posi_MALA_traces["b_V"][1000:].squeeze(), color='cyan', label='MALA', fill=True, alpha=0.1)
plt.axvline(model["true_b_V"], color='red', linestyle='--')
plt.xlabel("b_V")
plt.ylabel("Density")
plt.title("b_V")
plt.legend()
# plt.savefig("sigma squared mala compare density scena 2.pdf", dpi=300, bbox_inches="tight")
plt.show()




# 2.3.7 Step size, acceptance rate, z count, log posterior, acceptance probability #

plt.plot(MALA_within_Gibbs_traces["dt"][:,:])
plt.title("step size")
if save==True:
    plt.savefig("tuned 0.001 m-MALA step size traceplots.pdf", dpi=300, bbox_inches="tight")
plt.show()

plt.plot(MALA_within_Gibbs_traces["acceptance_rate"])
plt.title("acceptance rate")
if save==True:
    plt.savefig("tuned 0.001 m-MALA acceptance rate traceplots.pdf", dpi=300, bbox_inches="tight")
plt.show()

plt.plot(MALA_within_Gibbs_traces["z_count"])
plt.title("z count")
if save==True:
    plt.savefig("tuned 0.001 m-MALA z count traceplots.pdf", dpi=300, bbox_inches="tight")
plt.show()

plt.plot(posi_MALA_traces["ll"][:])
plt.axhline(MALA().log_posterior({"s": model["true_s"], "sigma_squared": model["true_sigma_squared"], "background": jnp.unique(model["background_true"]) }, \
            model["true_ss_var"].reshape(-1,1), model["true_ss_mean"].reshape(-1,1)), color='red', linestyle='--')
plt.title("log posterior")
if save==True:
    plt.savefig("tuned 0.001 m-MALA log likelihood traceplots.pdf", dpi=300, bbox_inches="tight")
plt.show()

plt.plot(jnp.exp(MALA_within_Gibbs_traces["a"][:]))
plt.title("a")
if save==True:
    plt.savefig("tuned 0.001 m-MALA a traceplots.pdf", dpi=300, bbox_inches="tight")
plt.show()

plt.plot(MALA_within_Gibbs_traces["new_grad_squared_sum"][:])
plt.title("new_grad_squared_sum")
if save==True:
    plt.savefig("tuned 0.001 m-MALA new_grad_squared_sum traceplots.pdf", dpi=300, bbox_inches="tight")
plt.show()

plt.plot(jnp.log(MALA_within_Gibbs_traces["max_dist"][:,10:46]))
plt.title("max_dist")
if save==True:
    plt.savefig("tuned 0.001 m-MALA max_dist traceplots.pdf", dpi=300, bbox_inches="tight")
plt.show()




