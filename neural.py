# This neural class creates the fundamentals of a
# neural network. All of the classes are to achieve
# standard, plain bread and butter goals of a
# standard backpropogation supervised learning
# network.
#
# It can work out of the box for basic stuff, but to extend it
# beyond just the plain standard, you can extend the classes
# in here by importing this module and extending neural.A_class.
#
# Best practice: put those extensions in package neural.ext.

# Array of neuron/node-value/gradient pairs
# [
#   [
#     0: Neuron/Node object of concern
#     1: Value/Gradient
#   ]
# ]
#
# Sorry, reason for this is because dictionaries
# do not support objects as keys.

import random

# Standard digital-output perceptron
# IN list of NeuronInputs
# OUT list of Nodes
class Neuron:
	def __init__(self, inputs = list(), outputs = list()):
		self.inputs = inputs
		self.outputs = outputs

	def output(self):
		sum = 0
		for input in self.inputs:
			sum += input.weight * input.input.value

		return self.activation(sum)

	def updateOutput(self):
		outputValue = self.output()

		for output in self.outputs:
			output.value = outputValue

	def inputGradient(self, input):
		neuroninput_gradient = self.activationGradient() # times 1

		return {"next": neuroninput_gradient * input.weight,
			"weight": neuroninput_gradient * input.input.value}

	def inputGradients(self):
		temp = list()
		for input in self.inputs:
			temp.append([input, self.inputGradient(input)])

		return temp

	# Input gradients
	# "next": Gradient of external node that inputs to the neuron (for backpropogation)
	# "weight": Gradient of weight of the neuron input (for changing the weight)
	def updateInputGradients(self):
		for input in self.inputGradients():
			input[0].input.gradient = input[1]["next"]
			input[0].weight_gradient = input[1]["weight"]

	def activation(self, sum):
		return 1 if sum >= 1 else 0

	def activationGradient(self):
		output = self.output

		sum = 0
		for output in self.outputs:
			sum += output.gradient

		return sum

#		if output is 1 and sum > 0:
#			return 0
#		elif output is not 1 and sum > 0:
#			return sum
#		elif output is 0 and sum < 0:
#			return 0
#		elif output is not 0 and sum < 0:
#			return sum
#		else:
#			return 0 # sum == 0

# Standard weighted perceptron input
# IN Node (External of owner neuron)
# Must only belong to one neuron
class NeuronInput:
	def __init__(self, input = None, weight = 0, weight_gradient = 0):
		self.input = input
		self.weight = weight
		self.weight_gradient = weight_gradient

# Standard "wire" with standard backpropogation learning gradient
# NEXT Neuron
# PREV Neuron
class Node:
	def __init__(self, value = 0, gradient = 0, nextInput = None, prevOutput = None):
		self.value = value
		self.gradient = gradient
		self.nextInput = nextInput
		self.prevOutput = prevOutput

# Standard layered network with inputs, outputs and hidden layers
class Network:
	def __init__(self, inputs = list(), outputs = list(), hiddens = list()):
		self.inputs = inputs
		self.outputs = outputs
		self.hiddens = hiddens

	def output(self):
		temp = list()
		for input in self.inputs:
			temp.append(self.forward(input)) # To be optimized

		return temp

	# Returns output value of output neuron
	def forward(self, neuron):
		neuron.updateOutput()

		for output in neuron.outputs:
			if output.nextInput is not None: # Put no next input in node to stop forward propogation.
				forward(output.nextInput)
			else:
				return [neuron, neuron.output()]

# Standard backpropogation teacher to evoke one specific output per output
# Do note that even if it can support multiple output nodes per output, if it
# is different, they will still be equal to one another.
class Teacher:
	def __init__(self, network = None, step = 0.01):
		self.network = network
		self.step = step

	def backward(self, neuron):
		neuron.updateInputGradients()
		
		for input in neuron.inputs:
			input.weight += self.step * input.weight_gradient
			
			if input.input.prevOutput is not None:
				input.input.value += self.step * input.input.gradient
				backward(input.input.prevOutput)
	
	def teachStep(self, nodeGradients):
		for nodeGradient in nodeGradients:
			nodeGradient[0].gradient = nodeGradient[1]
			
			if nodeGradient[0].prevOutput is not None:
				self.backward(nodeGradient[0].prevOutput) # To be optimized

	# Teaching lesson format
	#
	# [
	#   {
	#     "inputs"
	#       [
	#         [
	#           0: Input node
	#           1: Input value
	#     "outputs"
	#       [
	#         [
	#           0: Output node
	#           1: Desired output value

	def teach(self, values, steps):
		for i in range(1, steps):
			value = random.choice(values)

			for inputNode in value["inputs"]:
				inputNode[0].value = inputNode[1]

			self.network.output()
			
			temp = list()
			for outputNode in value["outputs"]:
				temp.append([outputNode[0], 1 if outputNode[1] > outputNode[0].value else (-1 if outputNode[1] < outputNode[0].value else 0)])

			self.teachStep(temp)
