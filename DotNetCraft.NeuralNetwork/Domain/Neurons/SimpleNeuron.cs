using System;
using System.Collections.Generic;
using DotNetCraft.NeuralNetwork.Core;
using DotNetCraft.NeuralNetwork.Domain.Synapses;

namespace DotNetCraft.NeuralNetwork.Domain.Neurons
{
    public enum NeuronType { Input, Hidden, Output }

    public class SimpleNeuron
    {
        private double _outputValue;

        private readonly IActivationFunction _activationFunction;
        /// <summary>
        /// A => B
        /// </summary>
        private readonly List<Synapse> _forwardSynapses;
        /// <summary>
        /// B => A
        /// </summary>
        private readonly List<Synapse> _backwardSynapses;

        public SimpleNeuron(IActivationFunction activationFunction)
        {
            if (activationFunction == null)
                throw new ArgumentNullException(nameof(activationFunction));

            _activationFunction = activationFunction;
            _forwardSynapses = new List<Synapse>();
            _backwardSynapses = new List<Synapse>();
        }

        public double SetInput(double input)
        {
            if (_backwardSynapses.Count != 0)
                throw new InvalidOperationException("Only for input neuron");

            return input;
        }

        public double SetInputs(double[] inputs)
        {
            var sum = 0.0;
            if (_backwardSynapses.Count == 0)
                throw new InvalidOperationException("Only for hidden neuron");

            for (var index = 0; index < inputs.Length; index++)
            {
                var input = inputs[index];
                var synapse = _backwardSynapses[index];
                var weight = synapse.Weight;
                sum += input * weight;
            }

            _outputValue = _activationFunction.Calculate(sum);
            return _outputValue;
        }

        public void BackPropagation(double idealOutput)
        {
            if (_forwardSynapses.Count != 0)
                throw new InvalidOperationException("Only for output neuron");

            var delta = idealOutput - _outputValue;
            delta *= _activationFunction.CalculatePrime(_outputValue);
            foreach (var backwardSynapse in _backwardSynapses)
            {
                backwardSynapse.SetDestinationDelta(this, delta);
            }
        }

        public void BackPropagation()
        {
            if (_forwardSynapses.Count == 0)
                throw new InvalidOperationException("Only for hidden neuron");

            var delta = 0.0;
            for (var index = 0; index < _forwardSynapses.Count; index++)
            {
                var synapse = _forwardSynapses[index];
                var weight = synapse.Weight;
                delta += weight * synapse.DestinationDelta;
            }

            delta *= _activationFunction.CalculatePrime(_outputValue);

            for (var index = 0; index < _backwardSynapses.Count; index++)
            {
                var backwardSynapse = _backwardSynapses[index];
#if RELEASE
                backwardSynapse.DestinationDelta = delta;
#else
                backwardSynapse.SetDestinationDelta(this, delta);
#endif
            }

            for (var index = 0; index < _forwardSynapses.Count; index++)
            {
                var synapse = _forwardSynapses[index];
                var destinationDelta = synapse.DestinationDelta;
                var gradW = destinationDelta * _outputValue;
                synapse.UpdateWeight(gradW);
            }
        }

        public void AddForwardSynapse(Synapse synapse)
        {
            _forwardSynapses.Add(synapse);
        }

        public void AddBackwardSynapse(Synapse synapse)
        {
            _backwardSynapses.Add(synapse);
        }
    }
}
