using DotNetCraft.NeuralNetwork.Core;
using DotNetCraft.NeuralNetwork.Domain.Neurons;
using DotNetCraft.NeuralNetwork.Domain.WeightBuilders;

namespace DotNetCraft.NeuralNetwork.Domain.Layers
{
    public class FullyConnectedLayer: INeuralNetworkLayer
    {
        private readonly Neuron[] _neurons;

        public int Count => _neurons.Length;

        public FullyConnectedLayer(int inputNeurons, int neurons, IActivationFunction activationFunction, SimpleWeightBuilder simpleWeightBuilder)
        {
            _neurons = new Neuron[neurons];
            for (int index = 0; index < neurons; index++)
            {
                _neurons[index] = new Neuron(inputNeurons, activationFunction, simpleWeightBuilder);
            }
        }

        public void SetInputs(double[] inputs)
        {
            foreach (var neuron in _neurons)
            {
                neuron.Forward(inputs);
            }
        }

        public double[] GetOutputs()
        {
            double[] result = new double[_neurons.Length];
            for (var index = 0; index < _neurons.Length; index++)
            {
                var neuron = _neurons[index];
                result[index] = neuron.OutputValue;
            }

            return result;
        }
    }
}
