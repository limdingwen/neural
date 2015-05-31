import neural

# Construct network

neuron = neural.Neuron()

inputN1 = neural.Node(nextInput=neuron)
inputN2 = neural.Node(nextInput=neuron)
biasN = neural.Node(value=1, nextInput=neuron)

neuron.inputs = [neural.NeuronInput(input=inputN1), neural.NeuronInput(input=inputN2), neural.NeuronInput(input=biasN)]

outputN = neural.Node(prevOutput=neuron)
neuron.outputs = [outputN]

network = neural.Network(inputs=[neuron], outputs=[neuron])

# Teach network

a = 13

teacher = neural.Teacher(network=network, step=0.005)
teacher.teach(values=[
		{"inputs": [
				[inputN1, 0],
				[inputN2, 0],
			],
		"outputs": [
				[outputN, 1]
			]
		},
		{"inputs": [
				[inputN1, 0],
				[inputN2, 1 * a],
			],
		"outputs": [
				[outputN, 0]
			]
		},
		{"inputs": [
				[inputN1, 1 * a],
				[inputN2, 0],
			],
		"outputs": [
				[outputN, 0]
			]
		},
		{"inputs": [
				[inputN1, 1 * a],
				[inputN2, 1 * a],
			],
		"outputs": [
				[outputN, 0]
			]
		}
	]
	, steps=10000)

# Print results

print ""
print "Weight 1:", neuron.inputs[0].weight
print "Weight 2:", neuron.inputs[1].weight
print "Weight bias:", neuron.inputs[2].weight
print ""

inputN1.value = 0
inputN2.value = 0

print "00:", network.output()[0][1]

inputN1.value = 0
inputN2.value = 1 * a

print "01:", network.output()[0][1]

inputN1.value = 1 * a
inputN2.value = 0

print "10:", network.output()[0][1]

inputN1.value = 1 * a
inputN2.value = 1 * a

print "11:", network.output()[0][1]
