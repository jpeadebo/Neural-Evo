import random

bias = 1 # 0 for off 1 for on

class Network:

    def __init__(self, framework):
        self.framework = framework
        self.framework[0] += bias
        self.inputs = self.framework[0]
        self.outputs = self.framework[len(self.framework) - 1]
        self.nodes = [0] * sum(self.framework)
        self.connections = []
        self.setBaseConnections()
        print(self.connections)

    def createNewConnection(self, input, output, weight):
        connection = {"in" : input, "out" : output, "weight" : weight, "state" : True}
        self.connections.append(connection)

    def setBaseConnections(self):
        for layers in range(len(self.framework[:-1])):
            pos = len(self.framework)-layers
            start = sum(self.framework[:-pos])
            for inputs in range(start, start + self.framework[layers]):
                for output in range(start + self.framework[layers], start + self.framework[layers] + self.framework[layers + 1]):
                    self.createNewConnection(inputs, output, random.uniform(-1,1))

    def clear(self):
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

    framework = [len(inputs[0]) - 1, hiddenLayer1Length, hiddenLayer2Length, numOutputs]
    network = Network(framework)

    print(network.executeNN(inputs[1]))

testXor()
