import math
import random
from dataclasses import dataclass, field

bias = 1  # 0 for off 1 for on
mutatePower = .1


# TODO: add comments to everything


# using two selected parents combine their layers one by one until a new child is made
def combineConnections(parents):
    child = []
    counter = 0
    for counter in range(len(parents[0].connections)):
        if len(parents[0].connections[counter]) == len(parents[1].connections[counter]):
            child.append(parents[0].connections[counter] if counter % 2 == 0 else parents[1].connections[counter])
            counter += 1
        else:
            raise Exception("bad combine layers, parent 0, parent 1",
                            parents[0].connections[counter], parents[1].connections[counter])
    return child


# select two parents that aren't the same based off fitness score
def getParentsFitness(fitness):
    parents = []
    fitnessSum = 0
    for agents in fitness:
        fitnessSum += agents[0]

    while len(parents) < 2:
        randomIndex = random.uniform(0, fitnessSum)
        for pos, agent in fitness:
            randomIndex -= pos
            if randomIndex <= 0 and len(parents) == 0:
                parents.append(agent)
            elif randomIndex <= 0 and agent != parents[0]:
                parents.append(agent)
    return parents


def first(n):
    return n[0]


class NeuralEvolution:

    # creates genSize network of size framework
    def __init__(self, genSize, framework):
        self.genSize = genSize

        # number of each type of child generation
        self.elitismNum = int(self.genSize * .1)
        self.crossoverNum = int(self.genSize * .5)
        self.mutationNum = self.genSize - self.elitismNum - self.crossoverNum

        self.framework = framework
        self.agents = [Network(self.framework) for _ in range(self.genSize)]

    # acquires the output for each agent
    def getAgentOutputs(self):
        return [agent.getOutputs() for agent in self.agents]

    # if enough inputs are given set each agent to its corresponding output
    def setInputs(self, inputs):
        if len(inputs) == self.genSize:
            for counter, input in enumerate(inputs):
                self.agents[counter].setInputs(input)
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

    def elitismChildren(self, fitness):
        fitness, agent = zip(*fitness)
        children = [children for children in agent[:self.elitismNum]]
        return children

    # select two parents and combine their layers to generate a new child
    def crossoverChildren(self, fitness):
        crossoverChildren = []

        for runs in range(self.crossoverNum):
            child = Network(self.framework)
            parents = getParentsFitness(fitness)

            child.connections = combineConnections(parents)

            crossoverChildren.append(child)
        return crossoverChildren

    # randomly select one parent based on their fitness
    def getMutateParent(self, fitness):
        parent = Network(self.framework)
        fitnessSum = 0
        for agents in fitness:
            fitnessSum += agents[0]

        randomIndex = random.uniform(0, fitnessSum)
        for pos, agent in fitness:
            randomIndex -= pos
            if randomIndex <= 0:
                parent = agent
        parent.mutateConnectionWeights(mutatePower)
        return parent

    # select a parent and randomly adjust its weights by a certain power then add it to the children
    def mutateParents(self, fitness):
        children = []
        for runs in range(self.mutationNum):
            parent = self.getMutateParent(fitness)
            children.append(parent)

        return children

    # reset all agents to 0 nodes and bias nodes
    def resetAgents(self):
        for agent in self.agents:
            agent.clearNodes()

    # creates new generation using fitness score
    def createNextGeneration(self, fitness):
        # takes the N best agents and directly copies them to the next generation

        # create sorted array of agents and their fitness score
        fitnessAgent = sorted(zip(fitness, self.agents), key=first)
        fitnessAgent.reverse()

        # use different ways of generating the next generation to increase diversity of agents
        eliteism = self.elitismChildren(fitnessAgent)
        crossover = self.crossoverChildren(fitnessAgent)
        mutation = self.mutateParents(fitnessAgent)

        # add the new children from the different forms of generation
        children = eliteism + crossover + mutation

        # make sure there is the correct genSize
        if len(children) != self.genSize:
            raise Exception("incorrect number of children, actual children num, wanted children num",
                                                                len(children), self.genSize)
        else:
            self.agents = children


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

    # returns a specified node based on its node networkPosition
    def getNodeOffPosition(self, position):
        for layerIndex, layer in enumerate(self.nodes):
            for nodeIndex, node in enumerate(layer):
                if node.networkPosition == position:
                    return self.nodes[layerIndex][nodeIndex]
        return Exception("position doesn't exist", position)

    # updates a specific node based off of its node networkPosition
    def setNodeOffPosition(self, position, value):
        self.getNodeOffPosition(position).value = value

    def sumNodeOffPosition(self, position, value):
        self.getNodeOffPosition(position).value += value

    # clear all nodes except bias node which it sets to one
    def clearNodes(self):
        for layerCounter, layer in enumerate(self.nodes):
            output = layerCounter == len(self.framework) - 1
            for nodeCounter, node in enumerate(layer[:-bias] if not output else layer):
                self.nodes[layerCounter][nodeCounter].value = 0
            if bias and not output:
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

    # creates the intial set of connections in the form of a 2d array, the first dim is layers,
    # the second is each connection in that layer
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

    # checks if input size is correct, if it is it sets the node values of input layer - bias to the input vector
    def setInputs(self, inputs):
        if len(inputs) == len(self.nodes[0]) - bias:
            for inputPos, input in enumerate(inputs):
                self.nodes[0][inputPos].value = input
        else:
            raise Exception("incompatable input size", len(inputs), len(self.nodes[0]) - bias)

    def getInputs(self):
        return self.getNodeValueLayer(0)

    # uses getNodeValueLayer to return the output layer values
    def getOutputs(self):
        return self.getNodeValueLayer(len(self.nodes) - 1)

    # returns the values of a specified layer of nodes
    def getNodeValueLayer(self, layer):
        vector = []
        for node in self.nodes[layer]:
            vector.append(node.value)
        return vector

    def getNodeValueLayerBiasless(self, layer):
        vector = []
        for node in self.nodes[layer]:
            if not node.nodeType == "bias":
                vector.append(node.value)
        return vector

    # TODO: add bounds detection
    def setNodeValueLayer(self, layerPos, layer):
        for counter, node in enumerate(self.nodes[layerPos]):
            if not node.nodeType == "bias":
                self.nodes[layerPos][counter].value = layer[counter]

    # simply feeding forward an input to an output while sigmoiding each layer once its calculated
    def feedForward(self):
        for counter, layer in enumerate(self.connections):
            for connection in layer:
                self.sumNodeOffPosition(connection["out"], self.getNodeOffPosition(connection["in"]).value * connection["weight"])
            sigmoidedLayer = sigmoidLayer(self.getNodeValueLayerBiasless(counter + 1))
            self.setNodeValueLayer(counter + 1, sigmoidedLayer)

    # call this method after creating the new agents, this will slightly adjust the values of each weight to
    # allow for changes within the agents
    def mutateConnectionWeights(self, mutatePower):
        maxConnection = 0
        for layer in self.connections:
            for connection in layer:
                maxConnection = connection["weight"] if maxConnection < connection["weight"] else maxConnection

        for layer in self.connections:
            for connection in layer:
                connection["weight"] += random.uniform(-maxConnection, maxConnection) * mutatePower

    # first clears the nodes of the network, then adds in inputs and calculates the output for that input
    def executeNN(self, inputs):
        self.clearNodes()
        self.setInputs(inputs)
        self.feedForward()
        self.printNodes()
        return self.getOutputs()

    def printNodes(self):
        for counter, layer in enumerate(self.nodes):
            print(self.getNodeValueLayer(counter))


# TODO: once above code is functional, fix this to be the minimum # of calls possible while maintaing all functionality
class TestXor:
    def __init__(self):
        # self.inputs = [[0, 0, 0, 0], [0, 0, 1, 1], [0, 1, 0, 1], [0, 1, 1, 0], [1, 0, 0, 1], [1, 0, 1, 0], [1, 1, 0, 0],[1, 1, 1, 1]]
        self.inputs = [[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]]
        self.hiddenLayer1Length = 15
        self.hiddenLayer2Length = 15
        self.numOutputs = 1

        self.size = 30
        self.network = NeuralEvolution(self.size,
                                       [len(self.inputs[0]) - 1, self.hiddenLayer1Length, self.hiddenLayer2Length,
                                        self.numOutputs])

    def calcFitness(self, output, agentDecisions):
        agentFitness = []
        for counter in range(len(agentDecisions[0])):
            agentSum = 0
            for run in range(len(agentDecisions)):
                agentSum += abs(agentDecisions[run][counter][0] - output[run][-1])

            agentScore = agentSum / 4.0
            if agentScore == 1:
                print("______________________________________________SUCCESS ON ", counter, "_________________________________________")
                break
            agentFitness.append(agentSum / 4.0)

        return agentFitness

    def runNetwork(self):

        runs = 100000
        for i in range(runs):
            print("-----------------------", i, "-----------------------", )
            # creating inputs, this is a simple problem the agent only needs to check 4 times

            setDecisions = []
            for i in self.inputs:
                input = [i[:-1]] * self.size
                setDecisions.append(self.network.getAgentDecisions(input))

            # need to calculate the error of the agents decisions, we don't care about direction just how close to
            # correct
            agentFitness = self.calcFitness(self.inputs, setDecisions)
            print(agentFitness)
            # creates the next generation and updates the agents to that new list of agents
            self.network.createNextGeneration(agentFitness)


xorTest = TestXor()

xorTest.runNetwork()
