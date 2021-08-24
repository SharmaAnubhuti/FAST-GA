"""
    Test module for geometry functions of cg components
"""
#  This file is part of FAST : A framework for rapid Overall Aircraft Design
#  Copyright (C) 2020  ONERA & ISAE-SUPAERO
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

from typing import Union

import pandas as pd
from openmdao.core.component import Component
import numpy as np

from fastoad.module_management.service_registry import RegisterPropulsion
from fastoad.model_base.propulsion import IOMPropulsionWrapper
from fastoad.model_base import FlightPoint

from fastga.models.propulsion.fuel_propulsion.base import AbstractFuelPropulsion
from fastga.models.propulsion.propulsion import IPropulsion

ENGINE_WRAPPER_BE76 = "test.wrapper.cg.beechcraft.dummy_engine"
ENGINE_WRAPPER_SR22 = "test.wrapper.cg.cirrus.dummy_engine"


########################################################################################################################
########################### Beechcraft BE76 dummy engine ###############################################################
########################################################################################################################


class DummyEngineBE76(AbstractFuelPropulsion):
    def __init__(
        self,
        max_power: float,
        design_altitude: float,
        design_speed: float,
        fuel_type: float,
        strokes_nb: float,
        prop_layout: float,
    ):
        """
        Dummy engine model returning thrust in particular conditions defined for htp/vtp areas.

        """
        super().__init__()
        self.prop_layout = prop_layout
        self.max_power = max_power
        self.design_altitude = design_altitude
        self.design_speed = design_speed
        self.fuel_type = fuel_type
        self.strokes_nb = strokes_nb

    def compute_flight_points(self, flight_points: Union[FlightPoint, pd.DataFrame]):
        flight_points.thrust = 3500.0
        flight_points.sfc = 0.0

    def compute_weight(self) -> float:
        return 562.83 / 2.0

    def compute_dimensions(self) -> (float, float, float, float, float, float):
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    def compute_drag(
        self, mach: Union[float, np.array], unit_reynolds: Union[float, np.array], wing_mac: float,
    ) -> Union[float, np.array]:
        return 0.0

    def get_consumed_mass(self, flight_point: FlightPoint, time_step: float) -> float:
        return 0.0

    def compute_max_power(self, flight_points: Union[FlightPoint, pd.DataFrame]) -> float:
        return 0.0


@RegisterPropulsion(ENGINE_WRAPPER_BE76)
class DummyEngineWrapperBE76(IOMPropulsionWrapper):
    def setup(self, component: Component):
        component.add_input("data:propulsion:IC_engine:max_power", np.nan, units="W")
        component.add_input("data:propulsion:IC_engine:fuel_type", np.nan)
        component.add_input("data:propulsion:IC_engine:strokes_nb", np.nan)
        component.add_input("data:TLAR:v_cruise", np.nan, units="m/s")
        component.add_input("data:mission:sizing:main_route:cruise:altitude", np.nan, units="m")
        component.add_input("data:geometry:propulsion:layout", np.nan)

    @staticmethod
    def get_model(inputs) -> IPropulsion:
        engine_params = {
            "max_power": inputs["data:propulsion:IC_engine:max_power"],
            "design_altitude": inputs["data:mission:sizing:main_route:cruise:altitude"],
            "design_speed": inputs["data:TLAR:v_cruise"],
            "fuel_type": inputs["data:propulsion:IC_engine:fuel_type"],
            "strokes_nb": inputs["data:propulsion:IC_engine:strokes_nb"],
            "prop_layout": inputs["data:geometry:propulsion:layout"],
        }

        return DummyEngineBE76(**engine_params)


########################################################################################################################
########################### Cirrus SR22 dummy engine ###################################################################
########################################################################################################################


class DummyEngineSR22(AbstractFuelPropulsion):
    def __init__(
        self,
        max_power: float,
        design_altitude: float,
        design_speed: float,
        fuel_type: float,
        strokes_nb: float,
        prop_layout: float,
    ):
        """
        Dummy engine model returning thrust in particular conditions defined for htp/vtp areas.

        """
        super().__init__()
        self.prop_layout = prop_layout
        self.max_power = max_power
        self.design_altitude = design_altitude
        self.design_speed = design_speed
        self.fuel_type = fuel_type
        self.strokes_nb = strokes_nb

    def compute_flight_points(self, flight_points: Union[FlightPoint, pd.DataFrame]):
        flight_points.thrust = 5417.0
        flight_points.sfc = 0.0

    def compute_weight(self) -> float:
        return 331.88

    def compute_dimensions(self) -> (float, float, float, float, float, float):
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    def compute_drag(
        self, mach: Union[float, np.array], unit_reynolds: Union[float, np.array], wing_mac: float,
    ) -> Union[float, np.array]:
        return 0.0

    def get_consumed_mass(self, flight_point: FlightPoint, time_step: float) -> float:
        return 0.0

    def compute_max_power(self, flight_points: Union[FlightPoint, pd.DataFrame]) -> float:
        return 0.0


@RegisterPropulsion(ENGINE_WRAPPER_SR22)
class DummyEngineWrapperSR22(IOMPropulsionWrapper):
    def setup(self, component: Component):
        component.add_input("data:propulsion:IC_engine:max_power", np.nan, units="W")
        component.add_input("data:propulsion:IC_engine:fuel_type", np.nan)
        component.add_input("data:propulsion:IC_engine:strokes_nb", np.nan)
        component.add_input("data:TLAR:v_cruise", np.nan, units="m/s")
        component.add_input("data:mission:sizing:main_route:cruise:altitude", np.nan, units="m")
        component.add_input("data:geometry:propulsion:layout", np.nan)

    @staticmethod
    def get_model(inputs) -> IPropulsion:
        engine_params = {
            "max_power": inputs["data:propulsion:IC_engine:max_power"],
            "design_altitude": inputs["data:mission:sizing:main_route:cruise:altitude"],
            "design_speed": inputs["data:TLAR:v_cruise"],
            "fuel_type": inputs["data:propulsion:IC_engine:fuel_type"],
            "strokes_nb": inputs["data:propulsion:IC_engine:strokes_nb"],
            "prop_layout": inputs["data:geometry:propulsion:layout"],
        }

        return DummyEngineSR22(**engine_params)
