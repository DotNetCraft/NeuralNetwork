using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using DotNetCraft.NeuralNetwork.Core;
using DotNetCraft.NeuralNetwork.Domain.ActivationFunctions;
using DotNetCraft.NeuralNetwork.Domain.Layers;
using DotNetCraft.NeuralNetwork.Domain.WeightBuilders;

namespace DotNetCraft.NeuralNetwork
{
    public class NeuralNetwork
    {
        private readonly ActivationFunctionFactory _activationFunctionFactory;
        private readonly SimpleWeightBuilder _simpleWeightBuilder;
        private readonly List<INeuralNetworkLayer> _layers;
        private INeuralNetworkLayer _inputLayer;
        private INeuralNetworkLayer _outputLayer;

        public NeuralNetwork(ActivationFunctionFactory activationFunctionFactory,
            SimpleWeightBuilder simpleWeightBuilder)
        {
            if (activationFunctionFactory == null)
                throw new ArgumentNullException(nameof(activationFunctionFactory));
            if (simpleWeightBuilder == null)
                throw new ArgumentNullException(nameof(simpleWeightBuilder));

            _activationFunctionFactory = activationFunctionFactory;
            _simpleWeightBuilder = simpleWeightBuilder;
            _layers = new List<INeuralNetworkLayer>();
        }

        public void AddInputLayer(int neurons, ActivationFunctionType activationFunctionType)
        {
            var activationFunction = _activationFunctionFactory.Build(activationFunctionType);
            _inputLayer = new InputLayer(neurons, activationFunction);

            _layers.Add(_inputLayer);
        }

        public void AddLayer(int neurons, ActivationFunctionType activationFunctionType)
        {
            var prevLayer = _layers[_layers.Count - 1];
            var inputNeurons = prevLayer.Count;
            var activationFunction = _activationFunctionFactory.Build(activationFunctionType);
            var layer = new FullyConnectedLayer(inputNeurons, neurons, activationFunction, _simpleWeightBuilder);
            _layers.Add(layer);
        }

        public double[] Calculate(double[] features)
        {
            var inputs = features;
            foreach (var layer in _layers)
            {
                layer.SetInputs(inputs);
                inputs = layer.GetOutputs();
            }

            return inputs;
        }

        public void Train(NeuralNetworkDataSet trainSet, int epochs, int printResultEpoch = 1000)
        {
            
        }
    }
}
