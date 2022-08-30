"""Parametric propeller IC engine."""
# -*- coding: utf-8 -*-
#  This file is part of FAST-OAD_CS23 : A framework for rapid Overall Aircraft Design
#  Copyright (C) 2022  ONERA & ISAE-SUPAERO
#  FAST is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.

import logging

import numpy
import pandas as pd
from typing import Union, Sequence, Tuple, Optional
from scipy.interpolate import interp2d
import os.path as pth
import numpy as np

import fastoad.api as oad
from fastoad.constants import EngineSetting
from fastoad.exceptions import FastUnknownEngineSettingError
from stdatm import Atmosphere

from aenum import Enum, IntEnum, extend_enum

from fastga.models.propulsion.fuel_propulsion.base import AbstractFuelPropulsion
from fastga.models.propulsion.dict import DynamicAttributeDict, AddKeyAttributes

from .exceptions import FastBasicICEngineInconsistentInputParametersError
from . import resources

# Logger for this module
_LOGGER = logging.getLogger(__name__)

# Set of dictionary keys that are mapped to instance attributes.
ENGINE_LABELS = {
    "power_SL": dict(doc="Power at sea level in watts."),
    "displacement": dict(doc="displacement of engine in cm3"),
    "mass": dict(doc="Mass in kilograms."),
    "length": dict(doc="Length in meters."),
    "height": dict(doc="Height in meters."),
    "width": dict(doc="Width in meters."),
}
# Set of dictionary keys that are mapped to instance attributes.
NACELLE_LABELS = {
    "wet_area": dict(doc="Wet area in meters²."),
    "length": dict(doc="Length in meters."),
    "height": dict(doc="Height in meters."),
    "width": dict(doc="Width in meters."),
}


class FlightPhase(Enum):
    """
    Enumeration of flight phases.
    """

    TAXI_OUT = "taxi_out"
    TAKEOFF = "takeoff"
    INITIAL_CLIMB = "initial_climb"
    CLIMB = "climb"
    CRUISE = "cruise"
    DESCENT = "descent"
    LANDING = "landing"
    TAXI_IN = "taxi_in"


class PropellerSetting(IntEnum):
    """
    Enumeration of propeller settings.
    """

    def __eq__(self, other):
        if isinstance(other, str):
            return self.name.lower() == other.lower()

        return super().__eq__(other)

    def __hash__(self):
        return self.value

    @classmethod
    def convert(cls, name: str) -> "PropellerSetting":
        """
        :param name:
        :return: the EngineSetting instance that matches the provided name (case-insensitive)
        """
        for instance in cls:
            if instance.name.lower() == name.lower():
                return instance

        return None


# Using the extensibility of EngineSetting is not needed here, but it allows to
# test it.
extend_enum(PropellerSetting, "TAKEOFF")
extend_enum(PropellerSetting, "CLIMB")
extend_enum(PropellerSetting, "CRUISE")
extend_enum(PropellerSetting, "IDLE")


class BasicICEngineD(AbstractFuelPropulsion):
    def __init__(
            self,
            max_power: float,
            cruise_altitude_propeller: float,
            fuel_type: float,
            strokes_nb: float,
            prop_layout: float,
            k_factor_sfc: float,
            speed_SL,
            thrust_SL,
            thrust_limit_SL,
            efficiency_SL,
            speed_CL,
            thrust_CL,
            thrust_limit_CL,
            efficiency_CL,
            effective_J,
            effective_efficiency_ls,
            effective_efficiency_cruise,
    ):
        """
        Parametric Internal Combustion engine.

        It computes engine characteristics using fuel type, motor architecture
        and constant propeller efficiency using analytical model from following sources:

        :param max_power: maximum delivered mechanical power of engine (units=W)
        :param cruise_altitude_propeller: design altitude for cruise (units=m)
        :param cruise_speed: design altitude for cruise (units=m/s)
        :param fuel_type: 1.0 for gasoline and 2.0 for diesel engine and 3.0 for Jet Fuel
        :param strokes_nb: can be either 2-strokes (=2.0) or 4-strokes (=4.0)
        :param prop_layout: propulsion position in nose (=3.0) or wing (=1.0)
        """
        if fuel_type == 1.0:
            self.ref = {
                "max_power": 132480,
                "length": 0.83,
                "height": 0.57,
                "width": 0.85,
                "mass": 136,
            }  # Lycoming IO-360-B1A
            self.map_file_path = pth.join(resources.__path__[0], "FourCylindersAtmospheric.csv")
        if fuel_type == 2.0:
            self.ref = {
                "max_power": 99000,
                "length": 0.816,
                "height": 0.636,
                "width": 0.778,
                "mass": 134,
            }  # TAE 125-02-99
            self.map_file_path = pth.join(resources.__path__[0], "FourCylindersAtmospheric.csv")
        else:
            self.ref = {
                "max_power": 160000,
                "length": 0.859,
                "height": 0.659,
                "width": 0.650,
                "mass": 205,
            }  # TDA CR 1.9 16V
            # FIXME: change the map file for those engines
            self.map_file_path = pth.join(resources.__path__[0], "FourCylindersAtmospheric.csv")
        self.prop_layout = prop_layout
        self.max_power = max_power
        self.cruise_altitude_propeller = cruise_altitude_propeller
        self.fuel_type = fuel_type
        self.strokes_nb = strokes_nb
        self.idle_thrust_rate = 0.01
        self.k_factor_sfc = k_factor_sfc
        self.speed_SL = speed_SL
        self.thrust_SL = thrust_SL
        self.thrust_limit_SL = thrust_limit_SL
        self.efficiency_SL = efficiency_SL
        self.speed_CL = speed_CL
        self.thrust_CL = thrust_CL
        self.thrust_limit_CL = thrust_limit_CL
        self.efficiency_CL = efficiency_CL
        self.effective_J = float(effective_J)
        self.effective_efficiency_ls = float(effective_efficiency_ls)
        self.effective_efficiency_cruise = float(effective_efficiency_cruise)
        self.specific_shape = None
        self.displacement = 1991    # [cm3]
        self.power_rated = self.max_power


        # Declare sub-components attribute
        self.engine = Engine(power_SL=max_power)
        self.engine.mass = None
        self.engine.length = None
        self.engine.width = None
        self.engine.height = None

        self.nacelle = Nacelle()
        self.nacelle.wet_area = None

        self.propeller = None

        # This dictionary is expected to have a Mixture coefficient for all EngineSetting values
        self.mixture_values = {
            EngineSetting.TAKEOFF: 8.8,
            EngineSetting.CLIMB: 8.8,
            EngineSetting.CRUISE: 10,
            EngineSetting.IDLE: 11,
        }
        self.rpm_engine = {
            EngineSetting.TAKEOFF: 3900.0,
            EngineSetting.CLIMB: 3900.0,
            EngineSetting.CRUISE: 3400.0,
            EngineSetting.IDLE: 1500.0,
        }
        self.rpm_prop = {
            PropellerSetting.TAKEOFF: 2300.0,
            PropellerSetting.CLIMB: 2300.0,
            PropellerSetting.CRUISE: 1600.0,
            PropellerSetting.IDLE: 890.0,
        }

        # ... so check that all EngineSetting values are in dict
        unknown_keys = [key for key in EngineSetting if key not in self.mixture_values.keys()]
        if unknown_keys:
            raise FastUnknownEngineSettingError("Unknown flight phases: %s", str(unknown_keys))


    def power_graph(self, altitude, rpm):
        """
        This function is for computing maximum power at an altitude and RPM for the reference engine.
        Refer Operations & Maintenance manual for the power graph.
        """

        at_25 = 42
        at_50 = 72
        at_65 = 91
        at_75 = 100
        if altitude <= 9000:
            at_90 = 121
        else:
            at_90 = 139 - altitude / 450

        if altitude <= 6000:
            at_100 = 132
        else:
            at_100 = 157 - altitude * 19 / 6000

        throttle = rpm * 100 / 2300
        array_throttle = [25, 50, 65, 75, 90, 100]
        array_power = [at_25, at_50, at_65, at_75, at_90, at_100]
        power_graph = numpy.interp(throttle, array_throttle, array_power) * 745.7  # conversion from hp to watt included
        return power_graph

    def read_map(self, altitude, rpm):
        """ Estimates MEP value of the engine at a flight point (power and RPM) """

        rpm_range = np.array([800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200,
                              2300])
        ns = 2
        displacement = 1991/10**6  # of reference engine; [cm3] to [m3]
        mep_altitude = np.zeros(np.size(rpm_range))
        for idx, rpm_values in enumerate(rpm_range):
            power_value = self.power_graph(altitude, rpm_values)
            rps = rpm_values * 1.689 * 2 * np.pi / 60
            mep = power_value * ns * np.pi * 2 / displacement / rps  # [pascal]
            mep_altitude[idx] = mep / 10 ** 5  # [bar]

        if 120000 > self.power_rated >= 100000:
            mep_altitude = mep_altitude * 1.24  # [bar]
        elif 200000 > self.power_rated >= 120000:
            mep_altitude = mep_altitude * 1.23

        mep = numpy.interp(rpm, rpm_range, mep_altitude)  # [bar]

        return mep

    def engine_displacement(self, rpm_rated):
        """ Estimates engine displacement based on rated power, rated RPM and max. MEP """
        ns = 2
        if self.power_rated <= 100000:
            mep_max = 15.5
        elif 100000 < self.power_rated < 120000:
            mep_max = 17.70
        elif 120000 <= self.power_rated < 200000:
            mep_max = 19.0
        elif self.power_rated >= 200000:
            print("power too high")

        rps = rpm_rated * 1.689 * 2 * np.pi / 60
        displacement = (2 * np.pi * ns * self.power_rated / rps / mep_max) * 10 ** 6 / 10 ** 5  # [cm3]
        return displacement


    def compute_flight_points(self, flight_points: oad.FlightPoint):
        # pylint: disable=too-many-arguments
        # they define the trajectory
        self.specific_shape = np.shape(flight_points.mach)
        if isinstance(flight_points.mach, float):
            sfc, thrust_rate, thrust = self._compute_flight_points(
                flight_points.mach,
                flight_points.altitude,
                flight_points.engine_setting,
                flight_points.thrust_is_regulated,
                flight_points.thrust_rate,
                flight_points.thrust,
            )
            flight_points.sfc = sfc
            flight_points.thrust_rate = thrust_rate
            flight_points.thrust = thrust
        else:
            mach = np.asarray(flight_points.mach)
            altitude = np.asarray(flight_points.altitude).flatten()
            engine_setting = np.asarray(flight_points.engine_setting).flatten()
            if flight_points.thrust_is_regulated is None:
                thrust_is_regulated = None
            else:
                thrust_is_regulated = np.asarray(flight_points.thrust_is_regulated).flatten()
            if flight_points.thrust_rate is None:
                thrust_rate = None
            else:
                thrust_rate = np.asarray(flight_points.thrust_rate).flatten()
            if flight_points.thrust is None:
                thrust = None
            else:
                thrust = np.asarray(flight_points.thrust).flatten()
            self.specific_shape = np.shape(mach)
            sfc, thrust_rate, thrust = self._compute_flight_points(
                mach.flatten(),
                altitude,
                engine_setting,
                thrust_is_regulated,
                thrust_rate,
                thrust,
            )
            if len(self.specific_shape) != 1:  # reshape data that is not array form
                # noinspection PyUnresolvedReferences
                flight_points.sfc = sfc.reshape(self.specific_shape)
                # noinspection PyUnresolvedReferences
                flight_points.thrust_rate = thrust_rate.reshape(self.specific_shape)
                # noinspection PyUnresolvedReferences
                flight_points.thrust = thrust.reshape(self.specific_shape)
            else:
                flight_points.sfc = sfc
                flight_points.thrust_rate = thrust_rate
                flight_points.thrust = thrust


    def _compute_flight_points(
            self,
            mach: Union[float, Sequence],
            altitude: Union[float, Sequence],
            engine_setting: Union[EngineSetting, Sequence],
            thrust_is_regulated: Optional[Union[bool, Sequence]] = None,
            thrust_rate: Optional[Union[float, Sequence]] = None,
            thrust: Optional[Union[float, Sequence]] = None,
    ) -> Tuple[Union[float, Sequence], Union[float, Sequence], Union[float, Sequence]]:
        """
        Same as :meth:`compute_flight_points`.

        :param mach: Mach number
        :param altitude: (unit=m) altitude w.r.t. to sea level
        :param engine_setting: define engine settings
        :param thrust_is_regulated: tells if thrust_rate or thrust should be used (works element-
        wise)
        :param thrust_rate: thrust rate (unit=none)
        :param thrust: required thrust (unit=N)
        :return: SFC (in kg/s/N), thrust rate, thrust (in N)
        """
        # Treat inputs (with check on thrust rate <=1.0)
        if thrust_is_regulated is not None:
            thrust_is_regulated = np.asarray(np.round(thrust_is_regulated, 0), dtype=bool)
        thrust_is_regulated, thrust_rate, thrust = self._check_thrust_inputs(
            thrust_is_regulated, thrust_rate, thrust
        )
        thrust_is_regulated = np.asarray(np.round(thrust_is_regulated, 0), dtype=bool)
        thrust_rate = np.asarray(thrust_rate)
        thrust = np.asarray(thrust)

        # Get maximum thrust @ given altitude & mach
        atmosphere = Atmosphere(np.asarray(altitude), altitude_in_feet=False)
        mach = np.asarray(mach) + (np.asarray(mach) == 0) * 1e-12
        atmosphere.mach = mach
        max_thrust = self.max_thrust(np.asarray(engine_setting), atmosphere)

        # We compute thrust values from thrust rates when needed
        idx = np.logical_not(thrust_is_regulated)
        if np.size(max_thrust) == 1:
            maximum_thrust = max_thrust
            out_thrust_rate = thrust_rate
            out_thrust = thrust
        else:
            out_thrust_rate = (
                np.full(np.shape(max_thrust), thrust_rate.item())
                if np.size(thrust_rate) == 1
                else thrust_rate
            )
            out_thrust = (
                np.full(np.shape(max_thrust), thrust.item()) if np.size(thrust) == 1 else thrust
            )
            maximum_thrust = max_thrust[idx]
        if np.any(idx):
            out_thrust[idx] = out_thrust_rate[idx] * maximum_thrust
        if np.any(thrust_is_regulated):
            out_thrust[thrust_is_regulated] = np.minimum(
                out_thrust[thrust_is_regulated], max_thrust[thrust_is_regulated]
            )

        # thrust_rate is obtained from entire thrust vector (could be optimized if needed,
        # as some thrust rates that are computed may have been provided as input)
        out_thrust_rate = out_thrust / max_thrust

        # Now SFC (g/kwh) can be computed and converted to sfc_thrust (kg/N) to match computation
        # from turbo shaft
        sfc, mech_power = self.sfc(out_thrust, engine_setting, atmosphere)
        sfc_time = (mech_power * 1e-3) * sfc / 3.6e6  # sfc in kg/s
        sfc_thrust = sfc_time / np.maximum(out_thrust, 1e-6)  # avoid 0 division

        return sfc_thrust, out_thrust_rate, out_thrust

    @staticmethod
    def _check_thrust_inputs(
            thrust_is_regulated: Optional[Union[float, Sequence]],
            thrust_rate: Optional[Union[float, Sequence]],
            thrust: Optional[Union[float, Sequence]],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Checks that inputs are consistent and return them in proper shape.
        Some inputs can be None, but outputs will be proper numpy arrays.
        :param thrust_is_regulated:
        :param thrust_rate:
        :param thrust:
        :return: the inputs, but transformed in numpy arrays.
        """
        # Ensure they are numpy array
        if thrust_is_regulated is not None:
            # As OpenMDAO may provide floats that could be slightly different
            # from 0. or 1., a rounding operation is needed before converting
            # to booleans
            thrust_is_regulated = np.asarray(np.round(thrust_is_regulated, 0), dtype=bool)
        if thrust_rate is not None:
            thrust_rate = np.asarray(thrust_rate)
        if thrust is not None:
            thrust = np.asarray(thrust)

        # Check inputs: if use_thrust_rate is None, we will use the provided input between
        # thrust_rate and thrust
        if thrust_is_regulated is None:
            if thrust_rate is not None:
                thrust_is_regulated = False
                thrust = np.empty_like(thrust_rate)
            elif thrust is not None:
                thrust_is_regulated = True
                thrust_rate = np.empty_like(thrust)
            else:
                raise FastBasicICEngineInconsistentInputParametersError(
                    "When use_thrust_rate is None, either thrust_rate or thrust should be provided."
                )

        elif np.size(thrust_is_regulated) == 1:
            # Check inputs: if use_thrust_rate is a scalar, the matching input(thrust_rate or
            # thrust) must be provided.
            if thrust_is_regulated:
                if thrust is None:
                    raise FastBasicICEngineInconsistentInputParametersError(
                        "When thrust_is_regulated is True, thrust should be provided."
                    )
                thrust_rate = np.empty_like(thrust)
            else:
                if thrust_rate is None:
                    raise FastBasicICEngineInconsistentInputParametersError(
                        "When thrust_is_regulated is False, thrust_rate should be provided."
                    )
                thrust = np.empty_like(thrust_rate)

        else:
            # Check inputs: if use_thrust_rate is not a scalar, both thrust_rate and thrust must be
            # provided and have the same shape as use_thrust_rate
            if thrust_rate is None or thrust is None:
                raise FastBasicICEngineInconsistentInputParametersError(
                    "When thrust_is_regulated is a sequence, both thrust_rate and thrust should be "
                    "provided."
                )
            if np.shape(thrust_rate) != np.shape(thrust_is_regulated) or np.shape(
                    thrust
            ) != np.shape(thrust_is_regulated):
                raise FastBasicICEngineInconsistentInputParametersError(
                    "When use_thrust_rate is a sequence, both thrust_rate and thrust should have "
                    "same shape as use_thrust_rate"
                )

        return thrust_is_regulated, thrust_rate, thrust

    def propeller_efficiency(
            self, thrust: Union[float, Sequence[float]], atmosphere: Atmosphere
    ) -> Union[float, Sequence]:
        """
        Compute the propeller efficiency.

        :param thrust: Thrust (in N)
        :param atmosphere: Atmosphere instance at intended altitude
        :return: efficiency
        """
        # Include advance ratio loss in here, we will assume that since we work at constant RPM
        # the change in advance ration is equal to a change in velocity
        installed_airspeed = atmosphere.true_airspeed * self.effective_J

        propeller_efficiency_SL = interp2d(
            self.thrust_SL,
            self.speed_SL,
            self.efficiency_SL * self.effective_efficiency_ls,  # Include the efficiency loss
            # in here
            kind="cubic",
        )
        propeller_efficiency_CL = interp2d(
            self.thrust_CL,
            self.speed_CL,
            self.efficiency_CL * self.effective_efficiency_cruise,  # Include the efficiency loss
            # in here
            kind="cubic",
        )
        if isinstance(atmosphere.true_airspeed, float):
            thrust_interp_SL = np.minimum(
                np.maximum(np.min(self.thrust_SL), thrust),
                np.interp(installed_airspeed, self.speed_SL, self.thrust_limit_SL),
            )
            thrust_interp_CL = np.minimum(
                np.maximum(np.min(self.thrust_CL), thrust),
                np.interp(installed_airspeed, self.speed_CL, self.thrust_limit_CL),
            )
        else:
            thrust_interp_SL = np.minimum(
                np.maximum(np.min(self.thrust_SL), thrust),
                np.interp(list(installed_airspeed), self.speed_SL, self.thrust_limit_SL),
            )
            thrust_interp_CL = np.minimum(
                np.maximum(np.min(self.thrust_CL), thrust),
                np.interp(list(installed_airspeed), self.speed_CL, self.thrust_limit_CL),
            )
        if np.size(thrust) == 1:  # calculate for float
            lower_bound = float(propeller_efficiency_SL(thrust_interp_SL, installed_airspeed))
            upper_bound = float(propeller_efficiency_CL(thrust_interp_CL, installed_airspeed))
            altitude = atmosphere.get_altitude(altitude_in_feet=False)
            propeller_efficiency = np.interp(
                altitude, [0, self.cruise_altitude_propeller], [lower_bound, upper_bound]
            )
        else:  # calculate for array
            propeller_efficiency = np.zeros(np.size(thrust))
            for idx in range(np.size(thrust)):
                lower_bound = propeller_efficiency_SL(
                    thrust_interp_SL[idx], installed_airspeed[idx]
                )
                upper_bound = propeller_efficiency_CL(
                    thrust_interp_CL[idx], installed_airspeed[idx]
                )
                altitude = atmosphere.get_altitude(altitude_in_feet=False)[idx]
                propeller_efficiency[idx] = (
                        lower_bound
                        + (upper_bound - lower_bound)
                        * np.minimum(altitude, self.cruise_altitude_propeller)
                        / self.cruise_altitude_propeller
                )
        # print("prop eff are", propeller_efficiency)
        return propeller_efficiency

    def compute_max_power(self, flight_points: oad.FlightPoint) -> Union[float, Sequence]:
        """
        Compute the ICE maximum power @ given flight-point.

        :param flight_points: current flight point(s)
        :return: maximum power in kW
        """
        atmosphere = Atmosphere(np.asarray(flight_points.altitude), altitude_in_feet=False)
        sigma = atmosphere.density / Atmosphere(0.0).density
        max_power = (self.max_power / 1e3) * (sigma - (1 - sigma) / 7.55)  # max power in kW

        return max_power

    def sfc(
            self,
            thrust: Union[float, Sequence[float]],
            engine_setting: Union[float, Sequence[float]],
            atmosphere: Atmosphere,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computation of the SFC.

        :param thrust: Thrust (in N)
        :param engine_setting: Engine settings (climb, cruise,... )
        :param atmosphere: Atmosphere instance at intended altitude
        :return: SFC (in g/kw) and Power (in W)
        """
        """
        Hoag, Kevin, and Brian Dondlinger. Vehicular engine design. Springer, 2015.
        """
        rpm_rated = 2300

        # Define RPM & mixture using engine settings
        if np.size(engine_setting) == 1:
            rpm_engine = self.rpm_engine[int(engine_setting)]
            mixture_values = self.mixture_values[int(engine_setting)]
        else:
            rpm_engine = np.array(
                [self.rpm_engine[engine_setting[idx]] for idx in range(np.size(engine_setting))]
            )
            mixture_values = np.array(
                [self.mixture_values[engine_setting[idx]] for idx in range(np.size(engine_setting))]
            )

        # Compute sfc @ 2500RPM
        real_power = np.zeros(np.size(thrust))
        torque = np.zeros(np.size(thrust))
        throttle = np.zeros(np.size(thrust))
        eff_vol = np.zeros(np.size(thrust))
        sfc = np.zeros(np.size(thrust))
        air_flowrate = np.zeros(np.size(thrust))
        fuel_flowrate = np.zeros(np.size(thrust))
        fuel_flowrate_check = np.zeros(np.size(thrust))
        if np.size(thrust) == 1:
            real_power = (
                    thrust * atmosphere.true_airspeed / self.propeller_efficiency(thrust, atmosphere)
            )
            torque = real_power / (rpm_engine * np.pi / 30.0)
            throttle = rpm_engine*100/3900
            eff_vol = -3 * 10**-5 * throttle**2 + 0.0041 * throttle + 0.7049
            displacement = self.engine_displacement(rpm_rated)
            air_flowrate = (displacement / 1000) * 1.225 * rpm_engine * eff_vol / 2000  # [kg/min]
            fuel_flowrate = air_flowrate / mixture_values  # [kg/min]
            fuel_flowrate_check = fuel_flowrate * 60 / 0.84  # [kg/min] to [l/h]
            sfc = (fuel_flowrate / real_power) * 1000 * 1000 * 60  # TSFC [gram/kW - min]
        else:
            for idx in range(np.size(thrust)):
                local_atmosphere = Atmosphere(
                    atmosphere.get_altitude()[idx], altitude_in_feet=False
                )
                local_atmosphere.mach = atmosphere.mach[idx]
                real_power[idx] = (
                        thrust[idx]
                        * atmosphere.true_airspeed[idx]
                        / self.propeller_efficiency(thrust[idx], local_atmosphere)
                )
                torque[idx] = real_power[idx] / (rpm_engine[idx] * np.pi / 30.0)
                throttle[idx] = rpm_engine[idx] * 100 / 3900
                eff_vol[idx] = -3 * 10 ** -5 * throttle[idx] ** 2 + 0.0041 * throttle[idx] + 0.7049
                local_atmosphere.rho = atmosphere.density[idx]
                displacement = self.engine_displacement(rpm_rated)
                air_flowrate[idx] = (displacement / 1000) * 1.225 * rpm_engine[idx] * eff_vol[idx] / 2000  # [kg/min]
                fuel_flowrate[idx] = air_flowrate[idx] / mixture_values[idx]  # [kg/min]
                fuel_flowrate_check[idx] = fuel_flowrate[idx] * 60/0.84  # [kg/min] to [l/h]
                sfc[idx] = fuel_flowrate[idx] * 1000 * 1000 * 60 / real_power[idx]   # TSFC [gram/kW hr]
        # print("real power is", real_power)
        # print("fuel flow [kg/min] is", fuel_flowrate)
        # print("displacement is", displacement)
        # print("fuel flow [lb/hr] is", fuel_flowrate_check)
        # print("TSFC is [g/kWh]", sfc)
        # print("\n")
        return sfc, real_power


    def max_thrust(
            self,
            engine_setting: Union[float, Sequence[float]],
            atmosphere: Atmosphere) -> np.ndarray:
        rpm_rated = 2300
        """
        Computation of maximum thrust either due to propeller thrust limit or ICE max power.
        :param engine_setting: Engine settings (climb, cruise,... )
        :param atmosphere: Atmosphere instance at intended altitude (should be <=20km)
        :return: maximum thrust (in N)
        """
        # Calculate maximum propeller thrust @ given altitude and speed
        if isinstance(atmosphere.true_airspeed, float):
            lower_bound = np.interp(atmosphere.true_airspeed, self.speed_SL, self.thrust_limit_SL)
            upper_bound = np.interp(atmosphere.true_airspeed, self.speed_CL, self.thrust_limit_CL)
        else:
            lower_bound = np.interp(
                list(atmosphere.true_airspeed), self.speed_SL, self.thrust_limit_SL
            )
            upper_bound = np.interp(
                list(atmosphere.true_airspeed), self.speed_CL, self.thrust_limit_CL
            )
        altitude = atmosphere.get_altitude(altitude_in_feet=False)
        thrust_max_propeller = (
                lower_bound
                + (upper_bound - lower_bound)
                * np.minimum(altitude, self.cruise_altitude_propeller)
                / self.cruise_altitude_propeller
        )
        # Calculate engine max power @ given RPM & altitude

        thrust_interp = np.linspace(
            np.min(self.thrust_SL) * np.ones(np.size(thrust_max_propeller)),
            thrust_max_propeller,
            10,
        ).transpose()

        if np.size(altitude) == 1:  # Calculate for float

            local_atmosphere = Atmosphere(
                altitude * np.ones(np.size(thrust_interp)), altitude_in_feet=False
            )

            local_atmosphere.mach = atmosphere.mach * np.ones(np.size(thrust_interp))
            propeller_efficiency = self.propeller_efficiency(thrust_interp[0], local_atmosphere)
            # power computation
            rpm_prop = np.array(self.rpm_prop[int(engine_setting)])
            ns = 2
            mep = self.read_map(altitude, rpm_prop)
            engine_displacement = self.engine_displacement(rpm_rated)
            max_power = mep * engine_displacement * (rpm_prop * 1.689 * 2 * np.pi / 60) * 10**5 / 10**6 / (2 * np.pi * ns)

            mechanical_power = thrust_interp[0] * atmosphere.true_airspeed / propeller_efficiency

            thrust_max_global = np.interp(max_power, mechanical_power, thrust_interp[0])

        else:  # Calculate for array
            thrust_max_global = np.zeros(np.size(altitude))
            max_power = np.zeros(np.size(altitude))
            for idx in range(np.size(altitude)):

                local_atmosphere = Atmosphere(
                    altitude[idx] * np.ones(np.size(thrust_interp[idx])), altitude_in_feet=False
                )
                local_atmosphere.mach = atmosphere.mach[idx] * np.ones(np.size(thrust_interp[idx]))
                propeller_efficiency = self.propeller_efficiency(
                    thrust_interp[idx], local_atmosphere
                )
                # power computation based on RPM
                rpm_prop = np.array(self.rpm_prop[int(engine_setting[idx])])

                ns = 2
                mep = self.read_map(altitude[idx], rpm_prop)
                engine_displacement = self.engine_displacement(rpm_rated)
                max_power[idx] = mep * engine_displacement * (rpm_prop * 1.689 * 2 * np.pi / 60) * 10**5 / 10**6 / (2 * np.pi * ns)

                mechanical_power = (
                        thrust_interp[idx] * atmosphere.true_airspeed[idx] / propeller_efficiency
                )
                thrust_max_global[idx] = np.interp(
                    max_power[idx], mechanical_power, thrust_interp[idx]
                )
                # propeller_efficiency = self.propeller_efficiency(
                #     thrust_max_global[idx], local_atmosphere
                # )
        return thrust_max_global

    def compute_weight(self) -> float:

        """
        Computes dry weight of the engine based on rated power.
        Refer excel sheet.
        """
        """
        :param max_power: rated power of engine [watts]
        :return: uninstalled_weight (in lbs)
        """
        power_sl = self.max_power
        uninstalled_weight = (0.876 * (power_sl/1000) + 50.749) * 2.205  # power in Watts (conversion to kW included), weight in lbs from kg
        self.engine.mass = uninstalled_weight

        return uninstalled_weight

    def compute_dimensions(self) -> (float, float, float, float):
        """
        Computes propulsion dimensions (engine/nacelle) from maximum power.
        Model from :...

        """

        # Compute engine dimensions
        self.engine.length = self.ref["length"] * (self.max_power / self.ref["max_power"]) ** (
                1 / 3
        )
        self.engine.height = self.ref["height"] * (self.max_power / self.ref["max_power"]) ** (
                1 / 3
        )
        self.engine.width = self.ref["width"] * (self.max_power / self.ref["max_power"]) ** (1 / 3)

        if self.prop_layout == 3.0:
            nacelle_length = 1.15 * self.engine.length
            # Based on the length between nose and firewall for TB20 and SR22
        else:
            nacelle_length = 2.0 * self.engine.length

        # Compute nacelle dimensions
        self.nacelle = Nacelle(
            height=self.engine.height * 1.1,
            width=self.engine.width * 1.1,
            length=nacelle_length,
        )
        self.nacelle.wet_area = 2 * (self.nacelle.height + self.nacelle.width) * self.nacelle.length

        return (
            self.nacelle["height"],
            self.nacelle["width"],
            self.nacelle["length"],
            self.nacelle["wet_area"],
        )

    def compute_drag(self, mach, unit_reynolds, wing_mac):
        """
        Compute nacelle drag coefficient cd0.

        """

        # Compute dimensions
        _, _, _, _ = self.compute_dimensions()
        # Local Reynolds:
        reynolds = unit_reynolds * self.nacelle.length
        # Roskam method for wing-nacelle interaction factor (vol 6 page 3.62)
        cf_nac = 0.455 / (
                (1 + 0.144 * mach ** 2) ** 0.65 * (np.log10(reynolds)) ** 2.58
        )  # 100% turbulent
        fineness_ratio = self.nacelle.length / np.sqrt(
            4 * self.nacelle.height * self.nacelle.width / np.pi
        )
        ff_nac = 1 + 0.35 / fineness_ratio  # Raymer (seen in Gudmunsson)
        if_nac = 1.2  # Jenkinson (seen in Gudmundsson)
        drag_force = cf_nac * ff_nac * self.nacelle.wet_area * if_nac

        # Roskam part 6 chapter 4.5.2.1 with no incidence
        interference_drag = 0.036 * wing_mac * self.nacelle.width * 0.2 ** 2.0

        # The interference drag is for the nacelle/wing interference, since for fuselage mounted
        # engine the nacelle drag is not taken into account we can do like so
        return drag_force + interference_drag


@AddKeyAttributes(ENGINE_LABELS)
class Engine(DynamicAttributeDict):
    """
    Class for storing data for engine.

    An instance is a simple dict, but for convenience, each item can be accessed
    as an attribute (inspired by pandas DataFrames). Hence, one can write::

        >>> engine = Engine(power_SL=10000.)
        >>> engine["power_SL"]
        10000.0
        >>> engine["mass"] = 70000.
        >>> engine.mass
        70000.0
        >>> engine.mass = 50000.
        >>> engine["mass"]
        50000.0

    Note: constructor will forbid usage of unknown keys as keyword argument, but
    other methods will allow them, while not making the matching between dict
    keys and attributes, hence::

        >>> engine["foo"] = 42  # Ok
        >>> bar = engine.foo  # raises exception !!!!
        >>> engine.foo = 50  # allowed by Python
        >>> # But inner dict is not affected:
        >>> engine.foo
        50
        >>> engine["foo"]
        42

    This class is especially useful for generating pandas DataFrame: a pandas
    DataFrame can be generated from a list of dict... or a list of oad.FlightPoint
    instances.

    The set of dictionary keys that are mapped to instance attributes is given by
    the :meth:`get_attribute_keys`.
    """


@AddKeyAttributes(NACELLE_LABELS)
class Nacelle(DynamicAttributeDict):
    """
    Class for storing data for nacelle.

    Similar to :class:`Engine`.
    """
