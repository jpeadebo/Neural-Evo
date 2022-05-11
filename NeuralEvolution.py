import random

bias = 1 # 0 for off 1 for on


class NeuralEvolution:

    def __init__(self):
        self.genSize = 100
        self.framework = [3,4,2]
        self.agents = [Network(self.framework)] * self.genSize

    def getAgentOutput(self):
        return [agent.getOutputs() for agent in self.agents]

    def setInputs(self, inputs):
        if len(inputs) == self.genSize:
            for counter, input in enumerate(inputs):
                self.agents[counter].setInputs(input)

    # run this as many times as needed to calculate the nn fitness
    def getAgentDecisions(self, inputs):
        self.setInputs(inputs)
        decisions = []
        for agent in self.agents:
            if agent.agentLife:
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

        # pick the remaining parents, current implementation pick a random num 0-sum all fitness, find that agent
        totalFitness = sum(fitness)
        for i in range(int(self.genSize*.2)-1):
            choice = random.randint(0,totalFitness)
            sumFit = 0
            for counter, agent in enumerate(self.agents):
                sumFit += fitness[counter]
                if choice < sumFit:
                    parents[counter] = 1
                    fitness[counter] = 0
                    break

        # num parents = sum(parents)
        return parents

    def getParents(self, parents):
        parentList = []
        for counter, agent in enumerate(self.agents):
            if parents[counter] == 1:
                parentList.append(agent)

        return parentList

    # requires same framework
    def createChild(self, dad, mom):
        child = Network(self.framework)
        dConnections = dad.connections
        mConnections = mom.connections
        for dConnection, mConnection in dConnections, mConnections:
            if dConnection["in"] == mConnection["in"] and dConnection["out"] == mConnection["out"]:
                dConnection["weight"] = dConnection["weight"] / mConnection["weight"]
            else:
                raise Exception("incompatible connection: d in, d out, m in, m out", dConnection["in"], mConnection["in"], dConnection["out"], mConnection["out"])
        child.setConnections(dConnections)

        return child

    def resetAgents(self):
        for agent in self.agents:
            agent.clear()
            agent.revive()

    def createNextGeneration(self, fitness):
        parents = self.pickParents(fitness)
        parentList = self.getParents(parents)

        numChildren = self.genSize - sum(parents)
        children = []

        for i in range(numChildren):
            # forces consistent amount of uses for each parent, and then a rand other parent
            dad = parentList[i%len(parentList)]
            mom = parentList[random.randint(0,len(parentList))]
            children.append(self.createChild(dad, mom))

        if len(parentList) + len(children) == self.genSize:
            self.agents = parentList + children
        else:
            raise Exception("incompatible new gen size, parent size:, child size:", len(parentList), len(children))

        self.resetAgents()

class Network:

    def __init__(self, framework):
        self.agentLife = True
        self.framework = framework
        self.framework[0] += bias
        self.inputs = self.framework[0]
        self.outputs = self.framework[len(self.framework) - 1]
        self.nodes = [0] * sum(self.framework)
        self.connections = []
        self.setBaseConnections()

    def killAgent(self):
        self.agentLife = False

    def revive(self):
        self.agentLife = True

    def getNumberConnections(self):
        return len(self.connections)

    def createNewConnection(self, input, output, weight):
        connection = {"in" : input, "out" : output, "weight" : weight, "state" : True}
        self.connections.append(connection)

    def setConnections(self, connections):
        if len(self.connections) == len(connections):
            self.connections = connections

    def setBaseConnections(self):
        for layers in range(len(self.framework[:-1])):
            pos = len(self.framework)-layers
            start = sum(self.framework[:-pos])
            for inputs in range(start, start + self.framework[layers]):
                for output in range(start + self.framework[layers], start + self.framework[layers] + self.framework[layers + 1]):
                    self.createNewConnection(inputs, output, random.uniform(-1,1))

    def clear(self):
        self.agentLife = true
        for node in self.nodes:
            node = 0

    def setInputs(self, inputs):
        if self.inputs == len(inputs):
            for i in range(len(inputs)):
                self.nodes[i] = inputs[i]
        else:
            raise Exception("incorrect input size, wanted:", self.inputs, " actual:", inputs)

    def feedForward(self):
        for connection in self.connections:
            if connection["state"]:
                self.nodes[connection["out"]] += self.nodes[connection["in"]] * connection["weight"]

    def getOutputs(self):
        outputStart = len(self.nodes) - self.outputs
        return self.nodes[outputStart:]

    def executeNN(self, inputs):
        self.clear()
        self.setInputs(inputs)
        self.feedForward()
        return self.getOutputs()


def testXor():
    #inputs = [[0, 0, 0, 0], [0, 0, 1, 1], [0, 1, 0, 1], [0, 1, 1, 0], [1, 0, 0, 1], [1, 0, 1, 0], [1, 1, 0, 0],[1, 1, 1, 1]]
    inputs = [[0,0,0],[0,1,1], [1,0,1],[1,1,0]]
    hiddenLayer1Length = 3
    hiddenLayer2Length = 3
    numOutputs = 1

    network = NeuralEvolution()
    input = [inputs[1]]*100
    print(network.testInstance(input))

testXor()
