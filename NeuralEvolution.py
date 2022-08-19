import random
import enum

bias = 0  # 0 for off 1 for on


class NeuralEvolution:

    # creates genSize network of size framework
    def __init__(self, genSize, framework):
        self.genSize = genSize
        self.numParents = int(genSize*.2)
        self.framework = framework
        self.agents = [Network(self.framework) for _ in range(self.genSize)]

    # acquires the output for each agent
    def getAgentOutputs(self):
        return [agent.getOutputs() for agent in self.agents]

    # if enough inputs are given set each agent to its corresponding output
    def setInputs(self, inputs):
        if len(inputs) == self.genSize:
            for counter, input in enumerate(inputs):
                self.agents[counter].setInputs(input[:-1]) # update input[:-1] to input once testing is over and
                # seperate inputs and outputs
        else:
            raise Exception("bad inputs, actual, wanted", len(inputs), self.genSize)

    # run this as many times as needed to calculate the nn fitness
    def getAgentDecisions(self, inputs):
        self.setInputs(inputs)
        decisions = []
        for agent in self.agents:
            agent.feedForward()
            decisions.append(agent.getOutputs())
            agent.clear()
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
                c1Connections[counter]["weight"] = (dConnection["weight"] if (counter % 2 == 0) else mConnections[counter]["weight"])
                c2Connections[counter]["weight"] = (dConnection["weight"] if (counter % 2 == 1) else mConnections[counter]["weight"])
            else:
                raise Exception("incompatible connection: d in, d out, m in, m out", dConnection["in"],
                                mConnections[counter]["in"], dConnection["out"], mConnections[counter]["out"])
        child.setConnections(c1Connections)
        child2.setConnections(c2Connections)

        return child, child2

    def resetAgents(self):
        for agent in self.agents:
            agent.clear()

    # creates new generation using fitness score
    def createNextGeneration(self, fitness):
        # makes a list of the agents that will be parents
        parents, bestAgentsSpot = self.pickParents(fitness)
        parentList = self.getParents(parents)

        # checks if the correct number of parents were made
        if len(parentList) == self.numParents:
            numChildren = self.genSize
            children = [self.agents[bestAgentsSpot[0]], self.agents[bestAgentsSpot[1]]]
            for i in range(int((numChildren-2) / 2)):
                # forces consistent amount of uses for each parent, and then a rand other parent
                dad = parentList[i % len(parentList)]
                mom = parentList[random.randint(0, len(parentList)-1)]
                twins = self.createChild(dad, mom)
                children.append(twins[0])
                children.append(twins[1])

            print(children)
            if len(children) == self.genSize:
                self.agents = children
            else:
                raise Exception("incompatible new gen size, children size:, gen size:", len(children), self.genSize)

            self.resetAgents()
        else:
            raise Exception("incorrect number of parents", len(parentList), self.numParents)


# confirm network works with a bias node
class Network:

    def __init__(self, framework):
        self.framework = framework
        # adds a bias to the network if needed, for testing this will be turned off for now
        self.framework[0] += bias
        self.inputs = self.framework[0]
        self.outputs = self.framework[len(self.framework) - 1]
        # initiates a network with random weights between -1 and 1
        self.nodes = [0 for i in range(sum(self.framework))]
        self.connections = []
        self.setBaseConnections()

    def printNetwork(self):
        print("network connections and weights", self.connections)
        print("network node values", self.nodes)

    def getNumberConnections(self):
        return len(self.connections)

    def createNewConnection(self, input, output, weight, inputLayer):
        # creates a new connection and assigns all needed information for that connection
        connection = {"in": input, "out": output, "weight": weight,
                      "connected layer": ("input to hidden" if inputLayer == 0 else
                                          ("hidden to output" if inputLayer == len(self.framework) - 2
                                           else "hidden to hidden"))}

        self.connections.append(connection)

    # allows for the repeated use of agents by overriding their connections with a new set of connections
    def setConnections(self, connections):
        if len(self.connections) == len(connections):
            self.connections = connections
            self.clear()

    # creates connections between layers, assigning the input node to an output node for the connection, a random weight
    # for that connection, and the type of connection it is(input to hidden, hidden to hidden, hidden to output
    def setBaseConnections(self):
        for layers in range(len(self.framework[:-1])):
            pos = len(self.framework) - layers
            start = sum(self.framework[:-pos])
            for inputs in range(start, start + self.framework[layers]):
                for output in range(start + self.framework[layers],
                                    start + self.framework[layers] + self.framework[layers + 1]):
                    self.createNewConnection(inputs, output, random.uniform(-1, 1), layers)

    def clear(self):
        for i in self.nodes:
            i = 0

    # confirms that the input fits the input size of the network, throws error if it isn't
    def setInputs(self, inputs):
        if self.inputs - bias == len(inputs):
            for i in range(len(inputs)):
                self.nodes[i] = inputs[i]
        else:
            raise Exception("incorrect input size, wanted:", self.inputs, " actual:", len(inputs))

    # runs through each connection updating the output node of each connection with the multiplication between its input
    # and weight
    def feedForward(self):
        for connection in self.connections:
            self.nodes[connection["out"]] += self.nodes[connection["in"]] * connection["weight"]

    # grabs output nodes as an array and returns them
    def getOutputs(self):
        outputStart = len(self.nodes) - self.outputs
        outputValues = []
        for outputs in self.nodes[outputStart:]:
            outputValues.append(outputs)
        return outputValues

    # first clears the nodes of the network, then adds in inputs and calculates the output for that input
    def executeNN(self, inputs):
        self.clear()
        self.setInputs(inputs)
        self.feedForward()
        return self.getOutputs()


def testXor():
    # inputs = [[0, 0, 0, 0], [0, 0, 1, 1], [0, 1, 0, 1], [0, 1, 1, 0], [1, 0, 0, 1], [1, 0, 1, 0], [1, 1, 0, 0],[1, 1, 1, 1]]
    inputs = [[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]]
    hiddenLayer1Length = 3
    hiddenLayer2Length = 3
    numOutputs = 1

    size = 100
    network = NeuralEvolution(size, [len(inputs[0])-1, hiddenLayer1Length, hiddenLayer2Length, numOutputs])

    runs = 100
    for i in range(runs):
        # creating inputs, since this is a simple problem the agent only needs to check once so inputs are always the same
        input = [inputs[i % 4]] * size
        agentDecisions = network.getAgentDecisions(input)

        # need to calculate the error of the agents decisions, we don't care about direction just how close to correct
        agentFitness = [abs(agent[0] - input[0][len(input[0])-1]) for agent in agentDecisions]

        # creates the next generation and updates the agents to that new list of agents
        network.createNextGeneration(agentFitness)
    print(agentDecisions)

testXor()
