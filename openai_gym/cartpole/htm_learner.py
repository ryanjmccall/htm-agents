# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2015, Numenta, Inc.  Unless you have an agreement
# with Numenta, Inc., for a separate license for this software code, the
# following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Affero Public License for more details.
#
# You should have received a copy of the GNU Affero Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------

from __future__ import division

from collections import defaultdict
import random
import json

import numpy

from nupic.engine import Network
from nupic.encoders.scalar import ScalarEncoder
from nupic.encoders.multi import MultiEncoder
from htmresearch.support.register_regions import registerAllResearchRegions


_VERBOSITY = 0

SP_PARAMS = {
  "spVerbosity": _VERBOSITY,
  "spatialImp": "cpp",
  "columnCount": 2048,
  "inputWidth": 0,
  "seed": 1936,
  "potentialPct": 0.8,
  "potentialRadius": 2048,
  "globalInhibition": 1,
  "numActiveColumnsPerInhArea": 40,
  "stimulusThreshold": 1,
  "synPermInactiveDec": 0.01,
  "synPermActiveInc": 0.02,
  "synPermConnected": 0.5,
  "minPctOverlapDutyCycle": 0.0,
  "minPctActiveDutyCycle": 0.0,
  "dutyCyclePeriod": 1000,
  "maxBoost": 1.0,
}

TM_PARAMS = {
  "basalInputWidth": 2048,
  "columnCount": 2048,
  "cellsPerColumn": 8,
  # "formInternalConnections": 1,
  "formInternalBasalConnections": True,  # inconsistency between CPP and PY
  "learningMode": True,
  "inferenceMode": True,
  "learnOnOneCell": False,
  "initialPermanence": 0.51,
  "connectedPermanence": 0.6,
  "permanenceIncrement": 0.1,
  "permanenceDecrement": 0.001,
  "minThreshold": 20,
  "predictedSegmentDecrement": 0.002,
  "activationThreshold": 20,
  "maxNewSynapseCount": 30,
  # "monitor": 0,
  "implementation": "etm_cpp",
}



class HtmLearner(object):
  """Only supports Discrete action spaces and Box observation spaces."""

  def __init__(self, environment, alpha=0.3, epsilon=0.75, epsilonDecay=0.99,
               discount=0.95, k=0.01):
    self.environment = environment
    self.actions = range(environment.action_space.n)

    self.alpha = alpha
    self.epsilon = epsilon
    self.epsilonDecay = epsilonDecay
    self.discount = discount
    self.k = k

    self.columnCount = TM_PARAMS["columnCount"]
    self.cellsPerColumn = TM_PARAMS["cellsPerColumn"]
    self.n = self.columnCount * self.cellsPerColumn
    self.weights = defaultdict(lambda: numpy.zeros(self.n))

    self.iterations = 0
    self.sequenceId = 0
    self.network = self._createNetwork()


  def _createNetwork(self):
    registerAllResearchRegions()
    network = Network()

    network.addRegion("observationSensor", "py.RawSensor",
                      json.dumps({"verbosity": _VERBOSITY}))
    self.observationSensor = network.regions["observationSensor"].getSelf()
    self.observationEncoder = self._createObservationEncoder()

    network.addRegion("actionSensor", "py.RawSensor",
                      json.dumps({"verbosity": _VERBOSITY}))
    self.actionSensor = network.regions["actionSensor"].getSelf()

    # print "action sensor %s" % self.actionSensor.spec
    self.actionEncoder = dict(
      [(a, self._generatePattern()) for a in self.actions]
    )

    SP_PARAMS["inputWidth"] = self.observationEncoder.getWidth()
    network.addRegion("spatialPoolerRegion", "py.SPRegion",
                      json.dumps(SP_PARAMS))
    network.addRegion("temporalMemoryRegion", "py.ExtendedTMRegion",
                      json.dumps(TM_PARAMS))

    network.link("observationSensor", "spatialPoolerRegion", "UniformLink", "",
                 srcOutput="dataOut", destInput="bottomUpIn")
    network.link("observationSensor", "spatialPoolerRegion", "UniformLink", "",
                 srcOutput="resetOut", destInput="resetIn")
    network.link("spatialPoolerRegion", "temporalMemoryRegion", "UniformLink",
                 "")
    network.link("actionSensor", "temporalMemoryRegion", "UniformLink", "",
                 srcOutput="dataOut", destInput="externalBasalInput")

    self.spatialPoolerRegion = network.regions[
      "spatialPoolerRegion"
    ].getSelf()
    self.temporalMemoryRegion = network.regions[
      "temporalMemoryRegion"
    ].getSelf()

    # print "spatialPoolerRegion %s" % self.spatialPoolerRegion.spec
    # print "temporalMemoryRegion %s" % self.temporalMemoryRegion.spec

    self.spatialPoolerRegion.setParameter("learningMode", 1, 1)
    self.temporalMemoryRegion.setParameter("learningMode", 1, 1)

    return network


  def _createObservationEncoder(self, n=512, w=41):
    """
    Only works with Box observation spaces.

    Hacky modifications for Cartpole.
    0) Cart Position (-4.8, 4.8) TODO when does it fail?
    1) Cart Velocity (-inf, inf) -> (-2.2, 2.2)  TODO use actual range
    2) Pole Angle (-0.418, 0.418) -> (-0.209, 0.209) TODO +-15 is stopping point
    3) Pole Velocity at Tip (-inf, inf) -> (-2.2, 2.2) TODO use actual range
    """
    multiEncoder = MultiEncoder()
    for i in xrange(self.environment.observation_space.shape[0]):
      high = self.environment.observation_space.high[i]
      low = self.environment.observation_space.low[i]
      # hacky
      if i == 1 or i == 3:
        # high and low are essentially inf; so threshold
        high = 2.2
        low = -2.2
      elif i == 2:
        high /= 2.  # Why?
        low /= 2.

      encoder = ScalarEncoder(w, low, high, n=n, name=str(i), clipInput=True)
      multiEncoder.addEncoder(str(i), encoder)

    return multiEncoder


  def bestAction(self, state):
    bestActions = []
    maxQValue = float("-inf")

    if random.random() < self.epsilon:  # TODO research why this initial randomness, due to need to learn weights?
      return random.choice(self.actions)

    for action in self.actions:
      qValue = self._qValue(state, action)

      if qValue > maxQValue:
        bestActions = [action]
        maxQValue = qValue
      elif qValue == maxQValue:
        bestActions.append(action)

    return random.choice(bestActions) if len(bestActions) else None


  def update(self, state, action, nextState, reward):
    self.iterations += 1
    targetValue = reward + (self.discount * self._value(nextState))
    qValue = self._qValue(state, action)
    if sum(state) > 0:
      correction = (targetValue - qValue) / sum(state)
    else:
      correction = 0.

    for i in state.nonzero()[0]:
      self.weights[action][i] += self.alpha * correction


  def compute(self, observation, action):
    obs = dict([(str(i), observation[i]) for i in xrange(len(observation))])
    encodedObservation = self.observationEncoder.encode(obs)
    encodedAction = self.actionEncoder[action]
    self.observationSensor.addDataToQueue(
      list(encodedObservation.nonzero()[0]), 0, self.sequenceId
    )
    self.actionSensor.addDataToQueue(list(encodedAction), 0, self.sequenceId)
    self.network.run(1)
    return self.getState()


  def getState(self):
    outputSize = self.n
    activeCells = numpy.zeros(shape=(outputSize,))
    activeCellIndices = numpy.array(
      self.temporalMemoryRegion._tm.getActiveCells())
    if len(activeCellIndices):
      activeCells[activeCellIndices] = 1

    predictiveCells = numpy.zeros(shape=(outputSize,))
    predCellIndices = numpy.array(
      self.temporalMemoryRegion._tm.getPredictiveCells())
    if len(predCellIndices):
      predictiveCells[predCellIndices] = 1

    return activeCells * predictiveCells
    # previous code
    # return self.temporalMemoryRegion.activeState * \
    #        self.temporalMemoryRegion.previouslyPredictedCells

  def _qValue(self, state, action):
    qValue = 0
    for i in state.nonzero()[0]:
      qValue += self.weights[action][i]
    return qValue


  def _value(self, state):  # returns max q-value of all actions from state
    qValues = [self._qValue(state, action) for action in self.actions]
    return max(qValues) if len(qValues) else 0.0

  def _generatePattern(self):
    cellsIndices = range(2048)
    random.shuffle(cellsIndices)
    return set(cellsIndices[:40])

  def updateWhenDone(self, cumulative_reward, ave_cumulative_reward):
    if ave_cumulative_reward is None:
      ave_cumulative_reward = cumulative_reward
    else:
      ave_cumulative_reward = (self.k * cumulative_reward +
                               (1 - self.k) * ave_cumulative_reward)
    if cumulative_reward > ave_cumulative_reward:
      # Only decay epsilon when reward improves, epsilon controls chance of random action
      self.epsilon *= self.epsilonDecay

  def reset(self):
    self.iterations = 0
    self.observationSensor.addResetToQueue(self.sequenceId)
    self.actionSensor.addResetToQueue(self.sequenceId)
    self.network.run(1)
    self.temporalMemoryRegion.reset()
    self.sequenceId += 1
