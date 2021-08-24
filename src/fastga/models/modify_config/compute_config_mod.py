"""
    FAST - Copyright (c) 2016 ONERA ISAE
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


import openmdao.api as om
from .compute_span import ComputeSpan
from .compute_fuselage_mod import ComputeFuselageMod


class ComputeConfigMod(om.Group):
    def initialize(self):
        self.options.declare("span_mod", types=list, default=[1.0, True, True])
        self.options.declare("fuselage_mod", types=list, default=[0, 0, 0, 0, 0])

    def setup(self):
        if self.options["span_mod"][0] != 1.0:
            self.add_subsystem(
                "compute_span", ComputeSpan(span_mod=self.options["span_mod"]), promotes=["*"]
            )
        if self.options["fuselage_mod"][1] != 0 or self.options["fuselage_mod"][2] != 0:
            self.add_subsystem(
                "compute_fuselage_mod",
                ComputeFuselageMod(fuselage_mod=self.options["fuselage_mod"]),
                promotes=["*"],
            )