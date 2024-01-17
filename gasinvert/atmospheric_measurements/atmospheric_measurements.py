from dataclasses import dataclass
from jaxtyping import Float, Array
from typing import Tuple
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.ndimage import gaussian_filter
import matplotlib.pylab as pylab
from mpl_toolkits.axes_grid1 import make_axes_locatable

params = {
        'legend.fontsize': '15',
        'axes.labelsize': '20',
        'axes.titlesize': '30',
        'xtick.labelsize':'x-large',
        'ytick.labelsize':'x-large',
        'figure.figsize': '10, 7',
        }
pylab.rcParams.update(params)




__all__ = ["Grid", "SourceLocation", "WindField", "AtmosphericState", "SensorsSettings",
        "GaussianPlume", "BackgroundGas", "Sensors"]





# Make a grid


def _compute_range(range: Tuple[Float[Array, ""], Float[Array, ""]], dx: Float[Array, ""]):
    """ Compute the range of a grid.

    Args:
        range (Tuple[float, float]): The range of the grid.
        dx (float): The grid spacing.

    Returns:
        Tuple[float, float]: The range of the grid.
    """
    return jnp.arange(range[0], range[1] + dx, dx)


@dataclass
class Grid:
    """ A grid of points in 3D space. """
    x_range: (Array, Array)
    y_range: (Array, Array)
    z_range: (Array, Array)
    dx: Array 
    dy: Array
    dz: Array


    @property
    def x(self):
        return _compute_range(self.x_range, self.dx)
    
    @property
    def y(self):
        return _compute_range(self.y_range, self.dy)
    
    @property
    def z(self):
        return _compute_range(self.z_range, self.dz)



@dataclass
class SourceLocation:
    """ A source location in 3D space. """

    source_location_x: Array
    source_location_y: Array
    source_location_z: Array



@dataclass
class WindField:
    """ A wind field. We assume the direction is in radians. """

    initial_wind_speed: Array
    initial_wind_direction: Array
    threesixty_degrees: bool
    number_of_time_steps: Array
    time_step: Array
    wind_speed_temporal_std: Array
    wind_direction_temporal_std: Array
    wind_temporal_correlation: Array
    wind_speed_seed: Array
    wind_direction_seed: Array


    

@dataclass
class AtmosphericState:
    """ An atmospheric  We assume the direction is in radians. """
    
    vertical_angle: Array
    horizontal_angle: Array
    emission_rate: Array
    half_width: Array
    max_abl: Array
    background_mean: Array                  # mean background concentration in ppb
    background_std: Array                   # std background concentration in ppb
    background_seed: Array                  # seed for the background concentration
    background_filter: str                  # filter to apply to the background concentration (e.g. Gaussian)
    Gaussian_filter_kernel: Array           # controls Gaussian filter smoothness
    downwind_power_H: Array                 # power of the downwind Gaussian filter
    downwind_power_V: Array                 # power of the downwind Gaussian filter



@dataclass
class SensorsSettings:
    """ Point sensors settings on the grid.

    Parameters
    ----------
    sensor_locations : list of tuples initial sensor cell number locations."""

    sensor_number: Array
    measurement_error_var: Array
    measurement_elevation: Array
    sensor_seed: Array
    measurement_error_seed: Array
    sensor_locations: Array





class GaussianPlume(Grid, SourceLocation, WindField, AtmosphericState, SensorsSettings):
    """ Returns Gaussian plumes for changing wind speed and direction. """

    def __init__(self, grid, source_location, wind_field, atmospheric_state, sensors_settings):
        
        self.grid = grid
        self.source_location = source_location
        self.wind_field = wind_field
        self.atmospheric_state = atmospheric_state
        self.sensors_settings = sensors_settings



    def generate_ornstein_u_process(self, parameters, key):
        """Generate a time varying Ornstein-Uhlenbeck process of input parameter
        """
        [time_varying_array, sigma, mean] = parameters
        drift = self.wind_field.wind_temporal_correlation * (mean - time_varying_array) * self.wind_field.time_step 
        diffusion = sigma * jnp.sqrt(self.wind_field.time_step) * jax.random.normal(key)
        time_varying_array = time_varying_array + drift + diffusion

        parameters = [time_varying_array, sigma, mean]
        
        return parameters, parameters



    def wind_speed(self):
        """
        Returns a time varying wind speed. 

        """
        key = jax.random.PRNGKey(self.wind_field.wind_speed_seed)
        initial_value, mean = self.wind_field.initial_wind_speed, self.wind_field.initial_wind_speed
        std = self.wind_field.wind_speed_temporal_std
        _, wind_speed = jax.lax.scan(self.generate_ornstein_u_process, [initial_value, std, mean], jax.random.split(key, self.wind_field.number_of_time_steps))
        wind_speed = jnp.where(wind_speed[0] < 1.0, 1.0, wind_speed[0])
        
        return wind_speed



    def wind_direction(self):
        """
        Returns a time varying wind direction. 

        """
        if self.wind_field.threesixty_degrees == True:
            wind_direction = jnp.arange(0, 360*3, (360*3)/self.wind_field.number_of_time_steps)
        else:
            key = jax.random.PRNGKey(self.wind_field.wind_direction_seed)
            initial_value, mean = self.wind_field.initial_wind_direction, self.wind_field.initial_wind_direction
            std = self.wind_field.wind_direction_temporal_std
            _, wind_d = jax.lax.scan(self.generate_ornstein_u_process, [initial_value, std, mean], jax.random.split(key, self.wind_field.number_of_time_steps))
            wind_direction = wind_d[0]
            
        return wind_direction
    


    def structure_to_vectorise(self, sensor_x, sensor_y, sensor_z):
        """Produces the vectors needed to avoid for loop in temporal coupling matrix computation."""
        meshedgrid = jnp.meshgrid(jnp.arange(self.grid.x_range[0], self.grid.x_range[1] + self.grid.dx, self.grid.dx) \
                            , jnp.arange(self.grid.y_range[0], self.grid.y_range[1] + self.grid.dy, self.grid.dy))
        sx, sy = meshedgrid[0], meshedgrid[1]
        source_coord_grid = jnp.array([sx.flatten(), sy.flatten(), np.full(sx.size, self.source_location.source_location_z)]).T
        temporal_sensor_x = jnp.repeat(sensor_x, self.wind_field.number_of_time_steps)
        temporal_sensor_y = jnp.repeat(sensor_y, self.wind_field.number_of_time_steps)
        temporal_sensor_z = jnp.repeat(sensor_z, self.wind_field.number_of_time_steps)
        
        return source_coord_grid, temporal_sensor_x, temporal_sensor_y, temporal_sensor_z



    def downwind_distance(self, s, coupling_x, coupling_y, wd):
        """Compute the downwind distance.

        Args:
            x (Float[Array, ""]): x coordinate of the point.
            y (Float[Array, ""]): y coordinate of the point.
        
        Returns:
            Float[Array, ""]: Downwind distance.
        """
        wd = jnp.deg2rad(wd)

        return jnp.cos(wd) * (coupling_x - s[0,:].reshape(-1,1).T) + jnp.sin(wd) * (coupling_y - s[1,:].reshape(-1,1).T)



    def horizontal_stddev(self, downwind, tan_gamma_H, b_H):
        """Compute the horizontal standard deviation.

        Args:
            x (Float[Array, ""]): x coordinate of the point.
            y (Float[Array, ""]): y coordinate of the point.
            z (Float[Array, ""]): z coordinate of the point.

        Returns:
            Float[Array, ""]: Vertical standard deviation.
        """
        posi_dw = jnp.where(downwind < 0.0, 1.0, downwind)
        horizontal_std = jnp.power(posi_dw, b_H) * tan_gamma_H
        return horizontal_std



    def vertical_stddev(self, downwind, tan_gamma_V, b_V):
        """Compute the vertical standard deviation.

        Args:
            x (Float[Array, ""]): x coordinate of the point.
            y (Float[Array, ""]): y coordinate of the point.
            z (Float[Array, ""]): z coordinate of the point.

        Returns:
            Float[Array, ""]: Vertical standard deviation.
        """
        posi_dw = jnp.where(downwind < 0.0, 1.0, downwind)
        vertical_std = jnp.power(posi_dw, b_V) * tan_gamma_V
        return vertical_std


    def horizontal_offset(self, s, x, y, wd):
        """Compute the horizontal offset.

        Args:
            x (Float[Array, ""]): x coordinate of the point.
            y (Float[Array, ""]): y coordinate of the point.

        Returns:
            Float[Array, ""]: Horizontal offset.
        """
        wd = jnp.deg2rad(wd)

        return  -jnp.sin(wd) * (x - s[0,:].reshape(-1,1).T) + jnp.cos(wd) * (y - s[1,:].reshape(-1,1).T)



    def vertical_offset(self, z, s):
        """Compute the vertical offset.

        Args:
            x (Float[Array, ""]): x coordinate of the point.
            y (Float[Array, ""]): y coordinate of the point.
            z (Float[Array, ""]): z coordinate of the point.

        Returns:
            Float[Array, ""]: Vertical offset.
        """
        
        return z - s[2,:].reshape(-1,1).T



    def methane_kg_m3_to_ppm(self, kg):
        """ Converts kg/m3 to ppm.
            kg: concentration in kg/m3.
            ppm: concentration in ppm.
            0.657: density of methane in kg/m3 at 25°C and 1 atm."""
        
        return kg * 1e6/0.657 



    def initial_gaussian_plume(self):
        """Compute the gas concentration at a single point.

        Args:
            x (Float[Array, ""]): x coordinate of the point.
            y (Float[Array, ""]): y coordinate of the point.
            z (Float[Array, ""]): z coordinate of the point.
        
        Returns:
            Float[Array, ""]: Gas concentration.
        """
        source = jnp.array([[self.source_location.source_location_x], [self.source_location.source_location_y]])
        meshedgrid = jnp.meshgrid(jnp.arange(self.grid.x_range[0], self.grid.x_range[1] + self.grid.dx, self.grid.dx) \
                    , jnp.arange(self.grid.y_range[0], self.grid.y_range[1] + self.grid.dy, self.grid.dy))
        all_x, all_y = meshedgrid[0].flatten(), meshedgrid[1].flatten()
        delta_R = self.downwind_distance(source, all_x, all_y, self.wind_field.initial_wind_direction)
        delta_H = self.horizontal_offset(source, all_x, all_y, self.wind_field.initial_wind_direction)
        delta_V = self.grid.z - self.source_location.source_location_z
        sigma_V = self.vertical_stddev(delta_R, jnp.tan(self.atmospheric_state.vertical_angle), self.atmospheric_state.downwind_power_V)
        sigma_H = self.horizontal_stddev(delta_R, jnp.tan(self.atmospheric_state.horizontal_angle), self.atmospheric_state.downwind_power_H)

        ws = self.wind_field.initial_wind_speed
        er = self.atmospheric_state.emission_rate
        max_abl = self.atmospheric_state.max_abl
        height = self.source_location.source_location_z
        twosigmavsqrd = 2 * (sigma_V ** 2)

        concentration = (1/(2 * jnp.pi * ws * sigma_H * sigma_V)) * \
                jnp.exp(-(delta_H **2)/(2*(sigma_H **2))) * \
                (jnp.exp(-((delta_V )**2)/(twosigmavsqrd)) + \
                jnp.exp(-((delta_V  - 2*(max_abl - height))**2)/(twosigmavsqrd)) + \
                jnp.exp(-((2 * height + delta_V  )**2)/(twosigmavsqrd)) + \
                jnp.exp(-((2 * max_abl + delta_V )**2)/(twosigmavsqrd))) * er
        
        concentration = concentration.flatten().squeeze()

        concentration = jnp.where(jnp.isnan(concentration), 0.0, concentration)
        concentration = jnp.where(delta_R < 0.0, 0.0, concentration)
        
        return self.methane_kg_m3_to_ppm(concentration)



    def initial_gaussian_plume_plot(self, save = False, format = "pdf"):
        df = pd.DataFrame( (self.initial_gaussian_plume() + 1e-8).reshape(len(self.grid.x), len(self.grid.y)))
        plt.figure(figsize=(15, 10))
        if (len(self.grid.x) > 10) and (len(self.grid.y) > 10):
            ax = sns.heatmap(df, cmap="jet", xticklabels=round(len(self.grid.x)/10), yticklabels=round(len(self.grid.y)/10))
            ax.set_xticks(jnp.arange(0, len(self.grid.x) + 1, round(len(self.grid.x)/10)))
            ax.set_yticks(jnp.arange(0, len(self.grid.y) + 1, round(len(self.grid.y)/10)))
            ax.set_yticklabels([round(i,1) for i in jnp.arange(self.grid.y_range[0], self.grid.y_range[1] + self.grid.dy + 1, self.grid.y_range[1]/10)], rotation=0)
            ax.set_xticklabels([round(i,1) for i in jnp.arange(self.grid.x_range[0], self.grid.x_range[1] + self.grid.dx + 1, self.grid.x_range[1]/10)], rotation=0)
        else:
            ax = sns.heatmap(df, cmap="jet")
            ax.set_yticklabels(self.grid.y)
            ax.set_xticklabels(self.grid.x)
        ax.invert_yaxis()
        ax.scatter(float(self.source_location.source_location_x/self.grid.dx), float(self.source_location.source_location_y/self.grid.dy), marker='.', s=100, color='orange')
        colorbar = ax.collections[0].colorbar
        colorbar.set_label('Parts per million (PPM)')
        plt.title("Initial Gaussian Plume")
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        if save == True:
            plt.savefig("Initial Gaussian Plume." + format, dpi=300, bbox_inches="tight")

        return plt.show()
    


    def log_initial_gaussian_plume_plot(self, save = False, format = "pdf"):
        df = pd.DataFrame(jnp.log(self.initial_gaussian_plume() + 1e-8).reshape(len(self.grid.x), len(self.grid.y)))
        plt.figure(figsize=(15, 10))
        if (len(self.grid.x) > 10) and (len(self.grid.y) > 10):
            ax = sns.heatmap(df, cmap="jet", xticklabels=round(len(self.grid.x)/10), yticklabels=round(len(self.grid.y)/10))
            ax.set_xticks(jnp.arange(0, len(self.grid.x) + 1, round(len(self.grid.x)/10)))
            ax.set_yticks(jnp.arange(0, len(self.grid.y) + 1, round(len(self.grid.y)/10)))
            ax.set_yticklabels([round(i,1) for i in jnp.arange(self.grid.y_range[0], self.grid.y_range[1] + self.grid.dy + 1, self.grid.y_range[1]/10)], rotation=0)
            ax.set_xticklabels([round(i,1) for i in jnp.arange(self.grid.x_range[0], self.grid.x_range[1] + self.grid.dx + 1, self.grid.x_range[1]/10)], rotation=0)
        else:
            ax = sns.heatmap(df, cmap="jet")
            ax.set_yticklabels(self.grid.y)
            ax.set_xticklabels(self.grid.x)
        ax.invert_yaxis()
        ax.scatter(float(self.source_location.source_location_x/self.grid.dx), float(self.source_location.source_location_y/self.grid.dy), marker='.', s=100, color='orange')
        colorbar = ax.collections[0].colorbar
        colorbar.set_label('Log parts per million (PPM)')
        plt.title("Log Initial Gaussian Plume")
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        if save == True:
            plt.savefig("Log initial Gaussian Plume." + format, dpi=300, bbox_inches="tight")

        return plt.show()



    def fixed_objects_of_grided_coupling_matrix(self):
        """ Returns the fixed objects for the coupling matrix.
            Avoids having to recompute them. These are not dependent on parameters being estimated. """
        sensor_x = jnp.array([i[0] for i in self.sensors_settings.sensor_locations])
        sensor_y = jnp.array([i[1] for i in self.sensors_settings.sensor_locations])
        sensor_z = jnp.array([i[2] for i in self.sensors_settings.sensor_locations])

        windspeeds = jnp.tile(self.wind_speed(), self.sensors_settings.sensor_number).reshape(-1,1)
        winddirection = jnp.tile(self.wind_direction(), self.sensors_settings.sensor_number).reshape(-1,1)
        s = self.structure_to_vectorise(sensor_x, sensor_y, sensor_z)[0].T # sensor_z not given because we assume cells are at correct source height
        s = s + self.grid.dx/2 # cell centered coupling matrix source locations
        xx = self.structure_to_vectorise(sensor_x, sensor_y, sensor_z)[1].reshape(-1,1)
        yy = self.structure_to_vectorise(sensor_x, sensor_y, sensor_z)[2].reshape(-1,1)
        delta_R = self.downwind_distance(s, xx, yy, winddirection)
        delta_H = self.horizontal_offset(s, xx, yy, winddirection)
        z = self.structure_to_vectorise(sensor_x, sensor_y, sensor_z)[3].reshape(-1,1)
        delta_V = self.vertical_offset(z, s)
        max_abl = self.atmospheric_state.max_abl
        height = self.source_location.source_location_z

        return windspeeds, winddirection, delta_R, delta_H, delta_V, max_abl, height



    def temporal_grided_coupling_matrix(self, fixed, tan_gamma_H = None, tan_gamma_V = None, b_H = None, b_V = None):
        """Computes the coupling matrix for a time varying wind speed and direction."""
        if tan_gamma_H is None:
            tan_gamma_H = jnp.tan(self.atmospheric_state.horizontal_angle)
        if tan_gamma_V is None:
            tan_gamma_V = jnp.tan(self.atmospheric_state.vertical_angle)
        if b_H is None:
            b_H = self.atmospheric_state.downwind_power_H
        if b_V is None:
            b_V = self.atmospheric_state.downwind_power_V
        windspeeds = fixed[0]
        delta_R, delta_H, delta_V = fixed[2], fixed[3], fixed[4]
        max_abl, height = fixed[5], fixed[6]

        sigma_V = self.vertical_stddev(delta_R, tan_gamma_V, b_V)
        sigma_H = self.horizontal_stddev(delta_R, tan_gamma_H, b_H)
        twosigmavsqrd = 2.0 * (sigma_V ** 2)

        coupling = ( 1.0/ (2.0 * jnp.pi * windspeeds * sigma_H * sigma_V ) ) * \
            jnp.exp( -(delta_H **2 )/(2.0*(sigma_H **2)) ) * \
            (   jnp.exp( -((delta_V )**2)/(twosigmavsqrd) ) + \
                jnp.exp( -((delta_V  - 2.0*(max_abl - height))**2)/(twosigmavsqrd)) + \
                jnp.exp( -((2 * height + delta_V  )**2)/(twosigmavsqrd)) + \
                jnp.exp( -((2 * max_abl + delta_V )**2)/(twosigmavsqrd)) )
        
        coupling = jnp.where(delta_R <= 0.0, 0.0, coupling)
        A = self.methane_kg_m3_to_ppm(coupling)

        return A



    def fixed_objects_of_gridfree_coupling_matrix(self):
        """ Returns the fixed objects for the coupling matrix.
            Avoids having to recompute them. These are not dependent on parameters being estimated. """
        sensor_x = jnp.array([i[0] for i in self.sensors_settings.sensor_locations])
        sensor_y = jnp.array([i[1] for i in self.sensors_settings.sensor_locations])

        windspeeds = jnp.tile(self.wind_speed(), self.sensors_settings.sensor_number).reshape(-1,1)
        winddirection = jnp.tile(self.wind_direction(), self.sensors_settings.sensor_number).reshape(-1,1)
        temporal_sensor_x = self.structure_to_vectorise(sensor_x, sensor_y)[1].reshape(-1,1)
        temporal_sensor_y = self.structure_to_vectorise(sensor_x, sensor_y)[2].reshape(-1,1)
        max_abl = self.atmospheric_state.max_abl
        height = self.source_location.source_location_z
        z = jnp.full(self.wind_field.number_of_time_steps * self.sensors_settings.sensor_number, self.sensors_settings.measurement_elevation).reshape(-1,1)

        return windspeeds, winddirection, temporal_sensor_x, temporal_sensor_y, max_abl, height, z



    def temporal_gridfree_coupling_matrix(self, fixed, x_coord = None, y_coord = None, tan_gamma_H = None, tan_gamma_V = None, b_H = None, b_V = None):
        if x_coord is None:
            x_coord = self.source_location.source_location_x
        if y_coord is None:
            y_coord = self.source_location.source_location_y
        if tan_gamma_H is None:
            tan_gamma_H = jnp.tan(self.atmospheric_state.horizontal_angle)
        if tan_gamma_V is None:
            tan_gamma_V = jnp.tan(self.atmospheric_state.vertical_angle)
        if b_H is None:
            b_H = self.atmospheric_state.downwind_power_H
        if b_V is None:
            b_V = self.atmospheric_state.downwind_power_V
        
        windspeeds, winddirection = fixed[0], fixed[1]
        temporal_sensor_x, temporal_sensor_y = fixed[2], fixed[3]
        max_abl, height = fixed[4], fixed[5]
        z = fixed[6]
        
        s = jnp.array([x_coord, y_coord, jnp.full(len(x_coord), self.source_location.source_location_z)]).reshape(3,len(x_coord))

        delta_R = self.downwind_distance(s, temporal_sensor_x, temporal_sensor_y, winddirection)
        delta_H = self.horizontal_offset(s, temporal_sensor_x, temporal_sensor_y, winddirection)
        delta_V = self.vertical_offset(z, s)

        sigma_V = self.vertical_stddev(delta_R, tan_gamma_V, b_V)
        sigma_H = self.horizontal_stddev(delta_R, tan_gamma_H, b_H)
        twosigmavsqrd = 2.0 * (sigma_V ** 2)

        coupling = ( 1.0/ (2.0 * jnp.pi * windspeeds * sigma_H * sigma_V ) ) * \
            jnp.exp( -(delta_H **2 )/(2.0*(sigma_H **2)) ) * \
            (   jnp.exp( -((delta_V )**2)/(twosigmavsqrd) ) + \
                jnp.exp( -((delta_V  - 2.0*(max_abl - height))**2)/(twosigmavsqrd)) + \
                jnp.exp( -((2 * height + delta_V  )**2)/(twosigmavsqrd)) + \
                jnp.exp( -((2 * max_abl + delta_V )**2)/(twosigmavsqrd)) )
        
        coupling = jnp.where(delta_R <= 0.0, 0.0, coupling)
        A = self.methane_kg_m3_to_ppm(coupling)

        return A







    def wind_speed_plot(self, save = False, format = "pdf"):
        """ Plot the wind speed over time. """
        speeds = self.wind_speed()[:self.wind_field.number_of_time_steps]
        plt.plot(speeds)
        plt.xlabel("Time (s)")
        plt.ylabel("Wind speed (m/s)")
        plt.title("Wind speed over time")
        if save == True:
            plt.savefig("Wind speed over time." + format, dpi=300, bbox_inches="tight")

        return plt.show()



    def wind_direction_plot(self, save = False, format = "pdf"):
        """ Plot the wind speed over time. """
        direction = self.wind_direction()[:self.wind_field.number_of_time_steps]
        plt.plot(direction)
        plt.xlabel("Time (s)")
        plt.ylabel("Wind direction (degrees)")
        plt.title("Wind direction over time")
        if save == True:
            plt.savefig("Wind direction over time." + format, dpi=300, bbox_inches="tight")

        return plt.show()













class BackgroundGas(Grid, SourceLocation, AtmosphericState):
    """ Background gas concentration using Gaussian random fields."""

    def __init__(self, grid, source_location, atmospheric_state):
        
        self.grid = grid
        self.source_location = source_location
        self.atmospheric_state = atmospheric_state


    def mean_and_std_correction(self, field: np.ndarray) -> np.ndarray:
        """ Corrects the mean and standard deviation of the field to match
            the background mean and standard deviation.
            
        Args:
            field (np.ndarray): 2D array.

        Returns:
            field (np.ndarray): 2D array with corrected mean and standard deviation.
        """
        field = field - np.mean(field)
        field = field/np.std(field)
        field = field*self.atmospheric_state.background_std + self.atmospheric_state.background_mean
        
        return field


    def Gaussian_filtered_Gaussian_random_field(self, kernel_std: Float[Array, ""] ) -> np.ndarray:
        """ Returns a Gaussian random field with Gaussian filter to the field"""
        np.random.seed(self.atmospheric_state.background_seed)
        gaussian_random_field = np.random.normal(loc=self.atmospheric_state.background_mean, scale=self.atmospheric_state.background_std, size=(len(self.grid.x), len(self.grid.y)))
        smoothed_GRF = gaussian_filter(gaussian_random_field, sigma=kernel_std)

        return self.mean_and_std_correction(gaussian_random_field), self.mean_and_std_correction(smoothed_GRF)


    def power_law_filtered_Gaussian_random_field(self) -> np.ndarray:
        """ Returns a Gaussian random field with power-law filter to the field
            using the inverse Fourier transform.
            
            We generate the x and y coordinates of the grid and create the mesh grid using NumPy's "meshgrid" function.
            Next, we generate the Gaussian random field using NumPy's "random.normal" function. To apply the power-law 
            filter to the field, we compute the power spectrum of the field using the Fourier transform. We first compute
            the 2D frequencies using NumPy's "fftfreq" function, and then compute the magnitude of the frequencies raised
            to the power of the exponent of the power-law filter. We add a small value epsilon (in this example, 1e-6) to
            the spectrum before raising it to the power of the exponent. This avoids division by zero errors or the
            creation of infinite values. We also modify the exponent to be a negative value, since we want the power-law
            filter to reduce the power of high-frequency components. We then set the DC component to zero to avoid division
            by zero errors. We then multiply the Fourier transform of the field by the power spectrum and compute the
            inverse Fourier transform to obtain the filtered field. Finally, we plot both the original and filtered fields
            
            """
        
        # Generate the Gaussian random field
        np.random.seed(self.atmospheric_state.background_seed)
        field = np.random.normal(loc=self.atmospheric_state.background_mean, scale=self.atmospheric_state.background_std, size=(len(self.grid.y), len(self.grid.x)))

        # Compute the power spectrum of the field using the Fourier transform
        field_fft = np.fft.fft2(field)
        freqs_x = np.fft.fftfreq(len(self.grid.x), self.grid.dx)
        freqs_y = np.fft.fftfreq(len(self.grid.y.tolist()), self.grid.dy)
        freqs_2d = np.meshgrid(freqs_x, freqs_y)
        freqs = np.sqrt(freqs_2d[0]**2 + freqs_2d[1]**2)
        exponent = -1.5
        epsilon = 1e-6
        freqs = (freqs + epsilon) ** exponent
        freqs[len(self.grid.y.tolist())//2,len(self.grid.x.tolist())//2] = 0 # Remove the DC component

        # Apply the power-law filter to the field using the inverse Fourier transform
        filtered_field_fft = field_fft * freqs
        filtered_field = np.fft.ifft2(filtered_field_fft).real

        return self.mean_and_std_correction(field), self.mean_and_std_correction(filtered_field)


    def colorbar(self, mappable):
        ax = mappable.axes
        fig = ax.figure
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="10%", pad=10)
        return fig.colorbar(mappable, cax=cax)


    def background_plot(self, save: bool = False, format: str = "pdf") :
        """ Plots the background gas concentration. 
        
        Args:
            background Float[Array, ""]): 2D array.

        Returns:
            plt.imshow(): Heatmap of background gas concentration.
        """
        field, filtered_field = self.power_law_filtered_Gaussian_random_field()
        df1 = pd.DataFrame(field).T
        df2 = pd.DataFrame(filtered_field).T
        fig, axs = plt.subplots(1, 2, figsize=(35, 15))
        
        if len(self.grid.x) > 10:
            axs[0] = sns.heatmap(df1, ax= axs[0], cmap="viridis", xticklabels=round(len(self.grid.x)/10), yticklabels=round(len(self.grid.y)/10))
            axs[0].set_xticks(jnp.arange(0, len(self.grid.x) + 1, round(len(self.grid.x)/10)))
            axs[0].set_yticks(jnp.arange(0, len(self.grid.y) + 1, round(len(self.grid.y)/10)))
            axs[0].set_yticklabels([round(i,1) for i in jnp.arange(self.grid.y_range[0], self.grid.y_range[1] + self.grid.dy + 1, self.grid.y_range[1]/10)], rotation=0)
            axs[0].set_xticklabels([round(i,1) for i in jnp.arange(self.grid.x_range[0], self.grid.x_range[1] + self.grid.dx + 1, self.grid.x_range[1]/10)], rotation=0)
        else:
            axs[0] = sns.heatmap(df1, ax= axs[0], cmap="viridis")
            axs[0].set_yticklabels(self.grid.y)
            axs[0].set_xticklabels(self.grid.x)
        axs[0].invert_yaxis()
        axs[0].scatter(float(self.source_location.source_location_x/self.grid.dx) - (self.grid.x_range[0]/self.grid.dx), float(self.source_location.source_location_y/self.grid.dy)-(self.grid.y_range[0]/self.grid.dy), marker='.', s=100, color='orange')
        colorbar = axs[0].collections[0].colorbar
        colorbar.set_label('Parts per million (PPM)')
        
        if len(self.grid.x) > 10:
            axs[1] = sns.heatmap(df2, ax= axs[1], cmap="viridis", xticklabels=round(len(self.grid.x)/10), yticklabels=round(len(self.grid.y)/10))
            axs[1].set_xticks(jnp.arange(0, len(self.grid.x) + 1, round(len(self.grid.x)/10)))
            axs[1].set_yticks(jnp.arange(0, len(self.grid.y) + 1, round(len(self.grid.y)/10)))
            axs[1].set_yticklabels([round(i,1) for i in jnp.arange(self.grid.y_range[0], self.grid.y_range[1] + self.grid.dy + 1, self.grid.y_range[1]/10)], rotation=0)
            axs[1].set_xticklabels([round(i,1) for i in jnp.arange(self.grid.x_range[0], self.grid.x_range[1] + self.grid.dx + 1, self.grid.x_range[1]/10)], rotation=0)
        else:
            axs[1] = sns.heatmap(df2, ax= axs[1], cmap="viridis")
            axs[1].set_yticklabels(self.grid.y)
            axs[1].set_xticklabels(self.grid.x)
        axs[1].invert_yaxis()
        axs[1].scatter(float(self.source_location.source_location_x/self.grid.dx), float(self.source_location.source_location_y/self.grid.dy), marker='.', s=100, color='orange')
        colorbar = axs[1].collections[0].colorbar
        colorbar.set_label('Parts per million (PPM)')
        
        plt.tight_layout(pad=3.0)
        if save == True:
            plt.savefig("backgrounds plot." + format, dpi=300, bbox_inches="tight")

        return plt.show()

"""
In this paper, we implement this by generating the x and y coordinates of the grid
and creating the mesh grid using NumPy's `meshgrid' function. Next, we generate the
Gaussian random field using NumPy's `random.normal' function. To apply the power-law
filter to the field, we compute the power spectrum of the field using the Fourier
transform. We first compute the 2D frequencies using NumPy's `fftfreq' function and
then compute the magnitude of the frequencies raised to the power of the exponent of
the power-law filter. We add a small value epsilon (in this example, 1e-6) to the
spectrum before raising it to the power of the exponent. This avoids division by zero
errors or the creation of infinite values. We also modify the exponent to be a negative
value, since we want the power-law filter to reduce the power of high-frequency components. We then set the Direct Current component to zero to avoid division by zero errors as it represents the low-frequency content of the signal that might be affected by noise or unwanted artifacts. We then multiply the Fourier transform of the field by the power spectrum and compute the inverse Fourier transform to obtain the filtered field.

"""












class Sensors(GaussianPlume, BackgroundGas):
    """ Point sensors on the grid."""

    def __init__(self, gaussianplume, backgroundgas):
        
        self.gaussianplume = gaussianplume
        self.backgroundgas = backgroundgas
    
    
    def meters_to_cell_nbr(self, coordx, coordy):
        """ Converts meters to cell number."""
        column = int(coordx/self.gaussianplume.grid.dx) 
        row = int(coordy/self.gaussianplume.grid.dy)
        index = (column)*len(self.gaussianplume.grid.y) + row

        return index    


    def source_rate(self):
        """ Creates a source rate vector."""
        grid_source_rate = np.zeros(int(len(self.gaussianplume.grid.x)) * int(len(self.gaussianplume.grid.y))).reshape(-1,1)
        grid_source_rate[self.meters_to_cell_nbr(self.gaussianplume.source_location.source_location_x, self.gaussianplume.source_location.source_location_y)] = self.gaussianplume.atmospheric_state.emission_rate
        
        return grid_source_rate


    def measurement_errors(self):
        """ Creates a measurement error vector."""
        np.random.seed(self.gaussianplume.sensors_settings.measurement_error_seed)
        measurement_errors = np.random.normal(0, np.sqrt(self.gaussianplume.sensors_settings.measurement_error_var), self.gaussianplume.sensors_settings.sensor_number * self.gaussianplume.wind_field.number_of_time_steps)

        return measurement_errors
    

    def background_vector(self) -> np.ndarray:
        """ Turns a background concentration 2D array to a vector."""
        filter = self.gaussianplume.atmospheric_state.background_filter
        if filter.lower() == "gaussian":
            background_concentration = self.backgroundgas.Gaussian_filtered_Gaussian_random_field(self.state, kernel_std = self.gaussianplume.atmospheric_state.Gaussian_filter_kernel)[1]
        elif filter.lower() == "power-law":
            background_concentration = self.backgroundgas.power_law_filtered_Gaussian_random_field()[1]
        else:
            print("Please choose a valid filter. Try \"Gaussian\" or \"power-law\".")

        return jnp.array([background_concentration.flatten()[j] for j in [self.meters_to_cell_nbr(i[0], i[1]) for i in self.gaussianplume.sensors_settings.sensor_locations]]), background_concentration
    


    def temporal_sensors_measurements(self):
        """ Creates an observation vector."""
        fixed = self.gaussianplume.fixed_objects_of_grided_coupling_matrix()
        A = self.gaussianplume.temporal_grided_coupling_matrix(fixed)
        source_rate = self.source_rate()
        measurement_errors = self.measurement_errors().reshape(-1,1)
        background_concentration = np.repeat(self.background_vector()[0], self.gaussianplume.wind_field.number_of_time_steps).reshape(-1,1)        
        sensors_measurements =  np.matmul(A, source_rate) + background_concentration + measurement_errors  
        
        return sensors_measurements.reshape(-1,1), A, source_rate, measurement_errors, background_concentration



    # def ols_estimate_sources(self, initial_background):
    #     """ Estimate the initial values for the source rates using OLS.

    #         Args:
    #             y: the observations
    #             X: the design matrix

    #         Returns:
    #             the OLS estimate of the source rates
    #     """
    #     fixed = self.gaussianplume.fixed_objects_of_grided_coupling_matrix()
    #     A = jgp.GaussianPlume(self.temporal, self.state, self.grid, self.sensors).coupling_matrix(fixed)
    #     y = self.temporal_sensors_measurements()[0]
    #     beta = np.full(self.sensors.sensor_number*self.temporal.number_of_time_steps, initial_background).reshape(-1,1)
        
    #     return np.linalg.lstsq(A, y - beta, rcond=None)[0]



    # def ols_estimate_sources_plot(self, initial_background, save: bool = False):
    #     plt.figure(figsize=(15,10))
    #     ax = sns.heatmap(pd.DataFrame(self.ols_estimate_sources(initial_background).reshape(len(self.grid.x),len(self.grid.y))), annot=False, cmap='viridis')
    #     if len(self.grid.x) > 10:
    #         ax.set_xticks(np.arange(0, len(self.grid.x) + 1, round(len(self.grid.x)/10)))
    #         ax.set_yticks(np.arange(0, len(self.grid.y) + 1, round(len(self.grid.y)/10)))
    #         ax.set_yticklabels([round(i,1) for i in jnp.arange(self.grid.y_range[0], self.grid.y_range[1] + self.grid.dy + 1, self.grid.y_range[1]/10)], rotation=0, fontsize=25)
    #         ax.set_xticklabels([round(i,1) for i in jnp.arange(self.grid.x_range[0], self.grid.x_range[1] + self.grid.dx + 1, self.grid.x_range[1]/10)], rotation=0, fontsize=25)
    #     else:
    #         ax.set_xticks(np.arange(0, len(self.grid.x) + 1, 1))
    #         ax.set_yticks(np.arange(0, len(self.grid.y) + 1, 1))
    #         ax.set_yticklabels([round(i,1) for i in jnp.arange(self.grid.y_range[0], self.grid.y_range[1] + self.grid.dy + 1, self.grid.dy)], rotation=0, fontsize=25)
    #         ax.set_xticklabels([round(i,1) for i in jnp.arange(self.grid.x_range[0], self.grid.x_range[1] + self.grid.dx + 1, self.grid.dx)], rotation=0, fontsize=25)
    #     ax.set_xlabel("X in meters", fontsize=30)
    #     ax.set_ylabel("Y in meters", fontsize=30)
    #     ax.invert_yaxis()
    #     plt.title("OLS estimate of emission rates")
    #     colorbar = ax.collections[0].colorbar
    #     colorbar.set_label('Parts per million (PPM)')
    #     colorbar.ax.tick_params(labelsize=20)

    #     if save == True:
    #         plt.savefig("ols_estimate_sources.pdf", dpi=300, bbox_inches="tight")

    #     return plt.show()


    def atmospheric_methane_and_sensors(self, save = False, format = "pdf"):
        sensors = self.gaussianplume.sensors_settings.sensor_locations
        background_concentration = self.background_vector()[1]
        concentration = self.gaussianplume.initial_gaussian_plume().reshape(len(self.gaussianplume.grid.x), len(self.gaussianplume.grid.y))
        df = pd.DataFrame( (concentration + background_concentration))
        plt.figure(figsize=(15, 10))
        if (len(self.gaussianplume.grid.x) > 10) and (len(self.gaussianplume.grid.y) > 10):
            ax = sns.heatmap(df, cmap="jet", xticklabels=round(len(self.gaussianplume.grid.x)/10), yticklabels=round(len(self.gaussianplume.grid.y)/10))
            ax.set_xticks(jnp.arange(0, len(self.gaussianplume.grid.x) + 1, round(len(self.gaussianplume.grid.x)/10)))
            ax.set_yticks(jnp.arange(0, len(self.gaussianplume.grid.y) + 1, round(len(self.gaussianplume.grid.y)/10)))
            ax.set_yticklabels([round(i,1) for i in jnp.arange(self.gaussianplume.grid.y_range[0], self.gaussianplume.grid.y_range[1] + self.gaussianplume.grid.dy + 1, self.gaussianplume.grid.y_range[1]/10)], rotation=0)
            ax.set_xticklabels([round(i,1) for i in jnp.arange(self.gaussianplume.grid.x_range[0], self.gaussianplume.grid.x_range[1] + self.gaussianplume.grid.dx + 1, self.gaussianplume.grid.x_range[1]/10)], rotation=0)
        else:
            ax = sns.heatmap(df, cmap="jet")
            ax.set_yticklabels(self.gaussianplume.grid.y)
            ax.set_xticklabels(self.gaussianplume.grid.x)
        ax.invert_yaxis()
        ax.scatter(float(self.gaussianplume.source_location.source_location_x/self.gaussianplume.grid.dx), float(self.gaussianplume.source_location.source_location_y/self.gaussianplume.grid.dy), marker='.', s=100, color='orange')
        for i in range(len(sensors)):
            ax.scatter(float(sensors[i][0]/self.gaussianplume.grid.dx), float(sensors[i][1]/self.gaussianplume.grid.dy), marker='*', s=50, color='white')
        colorbar = ax.collections[0].colorbar
        colorbar.set_label('Parts per million (PPM)')
        plt.title("Gaussian plume, background and sensors")
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        if save == True:
            plt.savefig("Gaussian plume, background and sensors." + format, dpi=300, bbox_inches="tight")

        return plt.show()
    

    def log_atmospheric_methane_and_sensors(self, save = False, format = "pdf"):
        sensors = self.gaussianplume.sensors_settings.sensor_locations
        background_concentration = self.background_vector()[1]
        concentration = self.gaussianplume.initial_gaussian_plume().reshape(len(self.gaussianplume.grid.x), len(self.gaussianplume.grid.y))
        df = pd.DataFrame( np.log(concentration + background_concentration))
        plt.figure(figsize=(15, 10))
        if (len(self.gaussianplume.grid.x) > 10) and (len(self.gaussianplume.grid.y) > 10):
            ax = sns.heatmap(df, cmap="jet", xticklabels=round(len(self.gaussianplume.grid.x)/10), yticklabels=round(len(self.gaussianplume.grid.y)/10))
            ax.set_xticks(jnp.arange(0, len(self.gaussianplume.grid.x) + 1, round(len(self.gaussianplume.grid.x)/10)))
            ax.set_yticks(jnp.arange(0, len(self.gaussianplume.grid.y) + 1, round(len(self.gaussianplume.grid.y)/10)))
            ax.set_yticklabels([round(i,1) for i in jnp.arange(self.gaussianplume.grid.y_range[0], self.gaussianplume.grid.y_range[1] + self.gaussianplume.grid.dy + 1, self.gaussianplume.grid.y_range[1]/10)], rotation=0)
            ax.set_xticklabels([round(i,1) for i in jnp.arange(self.gaussianplume.grid.x_range[0], self.gaussianplume.grid.x_range[1] + self.gaussianplume.grid.dx + 1, self.gaussianplume.grid.x_range[1]/10)], rotation=0)
        else:
            ax = sns.heatmap(df, cmap="jet")
            ax.set_yticklabels(self.gaussianplume.grid.y)
            ax.set_xticklabels(self.gaussianplume.grid.x)
        ax.invert_yaxis()
        ax.scatter(float(self.gaussianplume.source_location.source_location_x/self.gaussianplume.grid.dx), float(self.gaussianplume.source_location.source_location_y/self.gaussianplume.grid.dy),
                    marker='.', s=100, color='orange')
        for i in range(len(sensors)):
            ax.scatter(float(sensors[i][0]/self.gaussianplume.grid.dx), float(sensors[i][1]/self.gaussianplume.grid.dy), marker='*', s=50, color='white')
        colorbar = ax.collections[0].colorbar
        colorbar.set_label('Log parts per million (PPM)')
        plt.title("Log Gaussian plume, background and sensors")
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        if save == True:
            plt.savefig("Log Gaussian plume, background and sensors." + format, dpi=300, bbox_inches="tight")

        return plt.show()


