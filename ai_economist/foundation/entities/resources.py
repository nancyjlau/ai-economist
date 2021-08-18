# Copyright (c) 2020, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

import numpy as np

from ai_economist.foundation.base.registrar import Registry


class Resource:
    """Base class for Resource entity classes.

    Resource classes describe entities that can be a part of an agent's inventory.

    Resources can also be a part of the world as collectible entities: for each
    Resource class with Resource.collectible=True, a complementary
    ResourceSourceBlock Landmark class will be created in landmarks.py. For each
    collectible resource in the environment, the world map will include a resource
    source block channel (representing landmarks where collectible resources are
    generated) and a resource channel (representing locations where collectible
    resources have generated).
    """

    name = None
    collectible = None  # Is this something that exists in the world?
    # (versus something that can only be owned)

    def __init__(self):
        assert self.name is not None
        assert self.collectible is not None


resource_registry = Registry(Resource)


@resource_registry.add
class Time(Resource):
    """Time resource. Non collectible."""

    name = "Time"
    collectible = False


@resource_registry.add
class Project(Resource):
    """Project resource"""
    name = "Project"
    hardness = 0
    collectible = True
    def __init__(self,project_time,hardness,payment):
        self.project_time= project_time
        self.hardness = hardness
        self.payment = payment
    


@resource_registry.add
class Coin(Resource):
    """Coin resource. Included in all environments by default. Not collectible."""

    name = "Coin"
    collectible = False
