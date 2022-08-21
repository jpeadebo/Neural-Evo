import math
import random
from dataclasses import dataclass, field

bias = 1  # 0 for off 1 for on
mutatePower = .05

# TODO: add comments to everything


class NeuralEvolution:

    # creates genSize network of size framework
    def __init__(self, genSize, framework):
        self.genSize = genSize
        self.numParents = int(genSize * .2)
        self.framework = framework
        self.agents = [Network(self.framework) for _ in range(self.genSize)]

    # acquires the output for each agent
    def getAgentOutputs(self):
        return [agent.getOutputs() for agent in self.agents]

    # if enough inputs are given set each agent to its corresponding output
    def setInputs(self, inputs):
        if len(inputs) == self.genSize:
            for counter, input in enumerate(inputs):
                self.agents[counter].setInputs(input[:-1])  # update input[:-1] to input once testing is over and
                # seperate inputs and outputs
        else:
            raise Exception("bad inputs, actual, wanted", len(inputs), self.genSize)

    # run this as many times as needed to calculate the nn fitness
    def getAgentDecisions(self, inputs):
        self.resetAgents()
        self.setInputs(inputs)
        decisions = []
        for agent in self.agents:
            agent.feedForward()
            decisions.append(agent.getOutputs())
            # we can reset node values now that we have gotten the nn's decision
        return decisions

    def pickParents(self, fitness):
        parents = [0] * self.genSize

        # always include the max fitness agent
        maxFittnesSpot = fitness.index(max(fitness))
        parents[maxFittnesSpot] = 1
        fitness[maxFittnesSpot] = 0

        secondMaxFitnessSpot = fitness.index(max(fitness))
        parents[secondMaxFitnessSpot] = 1
        fitness[secondMaxFitnessSpot] = 0
        # pick the remaining parents, current implementation pick a random num 0-sum all fitness, find that agent
        for i in range(self.numParents - 2):
            totalFitness = sum(fitness)
            choice = random.uniform(0, totalFitness)
            sumFit = 0
            for counter, agent in enumerate(self.agents):
                sumFit += fitness[counter]
                if choice < sumFit:
                    parents[counter] = 1
                    fitness[counter] = 0
                    break

        # num parents = sum(parents)
        return parents, [maxFittnesSpot, secondMaxFitnessSpot]

    def getParents(self, parents):
        parentList = []
        for counter, agent in enumerate(self.agents):
            if parents[counter] == 1:
                parentList.append(agent)

        return parentList

    # requires same framework
    # TODO: rewrite all of this probably
    def createChild(self, dad, mom):
        child = Network(self.framework)
        child2 = Network(self.framework)
        dConnections = dad.connections
        mConnections = mom.connections
        c1Connections = dConnections
        c2Connections = mConnections
        # create 2 children, 1 being one mix of mom dad, 2 being the opposite mix of mom dad
        for counter, dConnection in enumerate(dConnections):
            if dConnection["in"] == mConnections[counter]["in"] and dConnection["out"] == mConnections[counter]["out"]:
                c1Connections[counter]["weight"] = (
                    dConnection["weight"] if (counter % 2 == 0) else mConnections[counter]["weight"])
                c2Connections[counter]["weight"] = (
                    dConnection["weight"] if (counter % 2 == 1) else mConnections[counter]["weight"])
            else:
                raise Exception("incompatible connection: d in, d out, m in, m out", dConnection["in"],
                                mConnections[counter]["in"], dConnection["out"], mConnections[counter]["out"])
        child.setConnections(c1Connections)
        child2.setConnections(c2Connections)

        return child, child2

    def mutateAgents(self):
        for agent in self.agents:
            agent.mutateConnectionWeights(mutatePower)

    def resetAgents(self):
        for agent in self.agents:
            agent.clearNodes()

    # creates new generation using fitness score
    def createNextGeneration(self, fitness):
        # makes a list of the agents that will be parents
        parents, bestAgentsSpot = self.pickParents(fitness)
        parentList = self.getParents(parents)

        # checks if the correct number of parents were made
        if len(parentList) == self.numParents:
            numChildren = self.genSize

            # creates new agents of the best two agents and sets their connections
            bestAgentOne = Network(self.framework)
            bestAgentTwo = Network(self.framework)
            bestAgentOne.setConnections(self.agents[bestAgentsSpot[0]].connections)
            bestAgentTwo.setConnections(self.agents[bestAgentsSpot[1]].connections)

            children = [bestAgentOne, bestAgentTwo]
            for i in range(int((numChildren - 2) / 2)):
                # forces consistent amount of uses for each parent, and then a rand other parent
                dad = parentList[i % len(parentList)]
                mom = parentList[random.randint(0, len(parentList) - 1)]
                twins = self.createChild(dad, mom)
                children.append(twins[0])
                children.append(twins[1])

            if len(children) == self.genSize:
                self.agents = children
            else:
                raise Exception("incompatible new gen size, children size:, gen size:", len(children), self.genSize)

            self.mutateAgents()
        else:
            raise Exception("incorrect number of parents", len(parentList), self.numParents, "parents", parentList)


# applies sigmoid to the node, however since the network cant handle 0's well we max out the node strength
def sigmoidNode(node):
    maxSigInput = 100
    if -math.inf < node < -maxSigInput:
        node = -maxSigInput
    elif maxSigInput < node < math.inf:
        node = maxSigInput
    S = 1 / (1 + math.pow(math.e, -node))
    return S


# takes in an input of one layer of the network, applies sigmoid to each node, then returns the adjusted layer
def sigmoidLayer(layer):
    vector = []
    for node in layer:
        vector.append(sigmoidNode(node))
    return vector


@dataclass
class Node:
    value: float
    nodeType: str
    networkPosition: int


# confirm network works with a bias node
class Network:

    def __init__(self, framework):
        # creates framework, adds bias nodes if requested, the output node doesn't need a bias so exclude that
        self.framework = [layerSize + bias for layerSize in framework[:-1]]
        self.framework.append(framework[-1])
        # create a 2d array of nodes where [layer][node] with the last node of each layer being a bias node if requested

        self.nodes = []
        self.setBaseNodes()

        self.connections = []
        self.setBaseConnections()

    # create the initial nodes based on the input framework
    def setBaseNodes(self):
        nodeCounter = 0
        # creates nodes based on the framework
        for counter, layer in enumerate(self.framework):
            nodeLayer = []
            output = counter == len(self.framework) - 1
            # if bias we skip the last node for later
            for node in range(layer - (bias if not output else 0)):
                nodeLayer.append(Node(0, ("input" if counter == 0 else
                                          ("output" if output else
                                           "hidden")), nodeCounter))
                nodeCounter += 1
            # if bias and we aren't on the last layer add a bias node
            if bias and not output:
                nodeLayer.append(Node(1, "bias", nodeCounter))
                nodeCounter += 1

            self.nodes.append(nodeLayer)

    def getNodeOffPosition(self, position):
        for layerIndex, layer in enumerate(self.nodes):
            for nodeIndex, node in enumerate(layer):
                if node.networkPosition == position:
                    return self.nodes[layerIndex][nodeIndex]
        return Exception("position doesn't exist", position)

    def setNodeOffPosition(self, position, value):
        for layerIndex, layer in enumerate(self.nodes):
            for nodeIndex, node in enumerate(layer):
                if node.networkPosition == position:
                    self.nodes[layerIndex][nodeIndex].value = value
        return Exception("position doesn't exist", position)

    # clear all nodes except bias node which it sets to one
    def clearNodes(self):
        for layerCounter, layer in enumerate(self.nodes):
            for nodeCounter, node in enumerate(layer[:-bias]):
                self.nodes[layerCounter][nodeCounter].value = 0
            if bias:
                self.nodes[layerCounter][-1].value = 1

    # creates a new connection and assigns all needed information for that connection, skipping if output node is bias
    def createNewConnection(self, input, output, weight):
        outputType = self.getNodeOffPosition(output).nodeType
        if outputType != "bias":
            inputType = self.getNodeOffPosition(input).nodeType
            connection = {"in": input, "out": output, "weight": weight,
                          "connected layer": (inputType + " to " + outputType)}

            return connection
        else:
            return 0

    def setBaseConnections(self):
        for layers in range(len(self.framework[:-1])):
            layerConnection = []
            pos = len(self.framework) - layers
            start = sum(self.framework[:-pos])
            for inputs in range(start, start + self.framework[layers]):
                for output in range(start + self.framework[layers],
                                    start + self.framework[layers] + self.framework[layers + 1]):
                    connection = self.createNewConnection(inputs, output, random.uniform(-1, 1))
                    if connection != 0:
                        layerConnection.append(connection)
            self.connections.append(layerConnection)

    # TODO: add bounds detection
    def setConnections(self, connections):
        self.connections = connections

    def setInputs(self, inputs):
        if len(inputs) == len(self.nodes[0]) - bias:
            for inputPos, input in enumerate(inputs):
                self.nodes[0][inputPos].value = input
        else:
            raise Exception("incompatable input size", len(inputs), len(self.nodes[0]) - bias)

    def getOutputs(self):
        return self.getNodeValueLayer(len(self.nodes)-1)

    def getNodeValueLayer(self, layer):
        vector = []
        for node in self.nodes[layer]:
            vector.append(node.value)
        return vector

    # TODO: add bounds detection
    def setNodeValueLayer(self, layerPos, layer):
        for counter, node in enumerate(self.nodes[layerPos]):
            self.nodes[layerPos][counter].value = layer[counter]

    def feedForward(self):
        for counter, layer in enumerate(self.connections):
            for connection in layer:
                self.setNodeOffPosition(connection["out"], connection["in"] * connection["weight"])
            sigmoidedLayer = sigmoidLayer(self.getNodeValueLayer(counter+1))
            self.setNodeValueLayer(counter+1, sigmoidedLayer)

    def mutateConnectionWeights(self, mutatePower):
        maxConnection = 0
        for connections in self.connections:
            maxConnection = connections["weight"] if maxConnection < connections["weight"] else maxConnection

        for connections in self.connections:
            connections["weight"] += random.uniform(-maxConnection, maxConnection) * mutatePower

    # first clears the nodes of the network, then adds in inputs and calculates the output for that input
    def executeNN(self, inputs):
        self.clearNodes()
        self.setInputs(inputs)
        self.feedForward()
        return self.getOutputs()


def testXor():
    # inputs = [[0, 0, 0, 0], [0, 0, 1, 1], [0, 1, 0, 1], [0, 1, 1, 0], [1, 0, 0, 1], [1, 0, 1, 0], [1, 1, 0, 0],[1, 1, 1, 1]]
    inputs = [[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]]
    hiddenLayer1Length = 2
    hiddenLayer2Length = 2
    numOutputs = 1

    size = 100
    network = NeuralEvolution(size, [len(inputs[0]) - 1, hiddenLayer1Length, hiddenLayer2Length, numOutputs])

    runs = 100
    for i in range(runs):
        print("-----------------------", i, "-----------------------", )
        # creating inputs, this is a simple problem the agent only needs to check 4 times
        for i in inputs:
            input = [i] * size
            agentDecisions = network.getAgentDecisions(input)

        # need to calculate the error of the agents decisions, we don't care about direction just how close to correct
        agentFitness = [abs(agent[0] - input[0][len(input[0]) - 1]) for agent in agentDecisions]

        # creates the next generation and updates the agents to that new list of agents
        network.createNextGeneration(agentFitness)
        print(agentDecisions)


testXor()
