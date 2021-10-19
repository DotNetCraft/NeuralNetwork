using System;
using DotNetCraft.NeuralNetwork.Core;
using DotNetCraft.NeuralNetwork.Domain.Neurons;

namespace DotNetCraft.NeuralNetwork.Domain.Layers
{
    public class InputLayer : INeuralNetworkLayer
    {
        private readonly IActivationFunction _activationFunction;
        private readonly InputNeuron[] _neurons;

        public int Count => _neurons.Length;

        public InputLayer(int neurons, IActivationFunction activationFunction)
        {
            if (activationFunction == null) 
                throw new ArgumentNullException(nameof(activationFunction));

            _activationFunction = activationFunction;
            _neurons = new InputNeuron[neurons];
            for (int index = 0; index < neurons; index++)
            {
                _neurons[index] = new InputNeuron(activationFunction);
            }
        }

        #region Implementation of INeuralNetworkLayer

        public void SetInputs(double[] inputs)
        {
            for (int i = 0; i < inputs.Length; i++)
            {
                var input = inputs[i];
                _neurons[i].Forward(input);
            }
        }

        public double[] GetOutputs()
        {
            double[] result = new double[_neurons.Length];
            for (int i = 0; i < result.Length; i++)
            {
                var val = _neurons[i].OutputValue;
                result[i] = _activationFunction.Calculate(val);
            }

            return result;
        }

        #endregion
    }
}
