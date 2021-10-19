using System;
using DotNetCraft.NeuralNetwork.Core;

namespace DotNetCraft.NeuralNetwork.Domain.Neurons
{
    public enum NeuronType { Input, Hidden, Output }
    
    public class Neuron: BaseNeuron<double[]>
    {
        private readonly IActivationFunction _activationFunction;
        private readonly double[] _weights;

        public Neuron(int inputsCount, IActivationFunction activationFunction, IWeightBuilder weightBuilder)
        {
            if (activationFunction == null) 
                throw new ArgumentNullException(nameof(activationFunction));
            if (weightBuilder == null) 
                throw new ArgumentNullException(nameof(weightBuilder));
            if (inputsCount <= 0) 
                throw new ArgumentOutOfRangeException(nameof(inputsCount));

            _activationFunction = activationFunction;
            _weights = weightBuilder.BuildWeights(inputsCount);
        }

        protected override double OnForward(double[] inputs)
        {
            if (inputs.Length != _weights.Length)
                throw new ArgumentOutOfRangeException(nameof(inputs));

            var sum = 0.0;
            for (var index = 0; index < inputs.Length; index++)
            {
                var input = inputs[index];
                var weight = _weights[index];
                sum += input * weight;
            }

            sum = _activationFunction.Calculate(sum);
            return sum;
        }
    }
}
