using System;
using DotNetCraft.NeuralNetwork.Core;
using DotNetCraft.NeuralNetwork.Domain.Neurons;
using DotNetCraft.NeuralNetwork.Domain.Synapses;
using DotNetCraft.NeuralNetwork.Domain.WeightBuilders;

namespace DotNetCraft.NeuralNetwork.Domain.Layers
{
    public class SimpleNeuronLayer: BaseLayer
    {
        private readonly IWeightBuilder _weightBuilder;
        private readonly SimpleNeuron[] _neurons;

        public int NeuronsCount => _neurons.Length;

        public SimpleNeuronLayer(int neurons, IActivationFunction activationFunction, IWeightBuilder weightBuilder)
        {
            if (weightBuilder == null) 
                throw new ArgumentNullException(nameof(weightBuilder));
            _weightBuilder = weightBuilder;
            _neurons = new SimpleNeuron[neurons];
            for (int i = 0; i < neurons; i++)
            {
                _neurons[i] = new SimpleNeuron(activationFunction);
            }
        }

        public double[] SetStartInputs(double[] inputs)
        {
            double[] result = new double[_neurons.Length];
            for (var index = 0; index < _neurons.Length; index++)
            {
                var neuron = _neurons[index];
                var input = inputs[index];
                result[index] = neuron.SetInput(input);
            }

            return result;
        }

        public double[] SetInputs(double[] inputs)
        {
            double[] result = new double[_neurons.Length];
            for (var index = 0; index < _neurons.Length; index++)
            {
                var neuron = _neurons[index];
                result[index] = neuron.SetInputs(inputs);
            }

            return result;
        }

        public void BackPropagation(double[] idealOutputs)
        {
            for (var index = 0; index < _neurons.Length; index++)
            {
                var neuron = _neurons[index];
                var idealOutput = idealOutputs[index];
                neuron.BackPropagation(idealOutput);
            }
        }

        public void BackPropagation()
        {
            for (var index = 0; index < _neurons.Length; index++)
            {
                var neuron = _neurons[index];
                neuron.BackPropagation();
            }
        }

        public void Connect(SimpleNeuronLayer layer, int window = -1)
        {
            //Current => layer
            if (window == -1)
            {
                //Full connections
                foreach (var sourceNeuron in _neurons)
                {
                    foreach (var destinationNeuron in layer._neurons)
                    {
                        var w = _weightBuilder.BuildWeight();
                        var synapse = new Synapse(sourceNeuron, destinationNeuron, w);
                        sourceNeuron.AddForwardSynapse(synapse);
                        destinationNeuron.AddBackwardSynapse(synapse);
                    }
                }
                return;
            }

            //Convolutional
            var destinationIndex = 0;
            var connections = 0;
            foreach (var sourceNeuron in _neurons)
            {
                var destinationNeuron = layer._neurons[destinationIndex];

                var w = _weightBuilder.BuildWeight();
                var synapse = new Synapse(sourceNeuron, destinationNeuron, w);
                sourceNeuron.AddForwardSynapse(synapse);
                destinationNeuron.AddBackwardSynapse(synapse);

                connections++;
                if (connections == window)
                {
                    connections = 0;
                    destinationIndex++;
                }
            }
        }
    }
}
