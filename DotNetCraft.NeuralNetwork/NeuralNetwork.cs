using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using DotNetCraft.NeuralNetwork.Core;
using DotNetCraft.NeuralNetwork.Domain;
using DotNetCraft.NeuralNetwork.Domain.ActivationFunctions;
using DotNetCraft.NeuralNetwork.Domain.Layers;
using DotNetCraft.NeuralNetwork.Domain.WeightBuilders;

namespace DotNetCraft.NeuralNetwork
{
    public class NeuralNetwork
    {
        private readonly ActivationFunctionFactory _activationFunctionFactory;
        private readonly SimpleWeightBuilder _simpleWeightBuilder;
        private readonly List<SimpleNeuronLayer> _layers;

        public NeuralNetwork(ActivationFunctionFactory activationFunctionFactory,
            SimpleWeightBuilder simpleWeightBuilder)
        {
            if (activationFunctionFactory == null)
                throw new ArgumentNullException(nameof(activationFunctionFactory));
            if (simpleWeightBuilder == null)
                throw new ArgumentNullException(nameof(simpleWeightBuilder));

            _activationFunctionFactory = activationFunctionFactory;
            _simpleWeightBuilder = simpleWeightBuilder;
            _layers = new List<SimpleNeuronLayer>();
        }

        public void AddLayer(int neurons, ActivationFunctionType activationFunctionType)
        {
            var activationFunction = _activationFunctionFactory.Build(activationFunctionType);
            var layer = new SimpleNeuronLayer(neurons, activationFunction, _simpleWeightBuilder);

            if (_layers.Count > 0)
            {
                var prevLayer = _layers[_layers.Count - 1];
                prevLayer.Connect(layer);
            }

            _layers.Add(layer);
        }

        public void AddConvolutionalLayer(int window, ActivationFunctionType activationFunctionType)
        {
            if (_layers.Count == 0)
                throw new InvalidOperationException("It can be at first place");

            var prevLayer = _layers[_layers.Count - 1];
            var neuronsCount = prevLayer.NeuronsCount;
            var neurons = neuronsCount / window;

            if (neurons * window < neuronsCount)
                neurons++;

            var activationFunction = _activationFunctionFactory.Build(activationFunctionType);
            var layer = new SimpleNeuronLayer(neurons, activationFunction, _simpleWeightBuilder);
            prevLayer.Connect(layer, window);
        }


        public double[] Calculate(double[] features)
        {
            var inputs = features;
            var layer = _layers[0];
            inputs = layer.SetStartInputs(inputs);

            for (var index = 1; index < _layers.Count; index++)
            {
                layer = _layers[index];
                inputs = layer.SetInputs(inputs);
            }

            return inputs;
        }

        private void BackProp(double[] outputs)
        {
            var layer = _layers[_layers.Count - 1];
            layer.BackPropagation(outputs);

            for (var index = _layers.Count - 2; index >= 0; index--)
            {
                layer = _layers[index];
                layer.BackPropagation();
            }
        }

        public TrainDetails Train(NeuralNetworkDataSet trainSet, int epochs, int printResultEpoch = 1000)
        {
            var trainDetails = new TrainDetails();
            for (int epoch = 0; epoch < epochs; epoch++)
            {
                trainDetails.Reset();
                var sumError = 0.0;
                for (int sampleIndex = 0; sampleIndex < trainSet.SampleCount; sampleIndex++)
                {
                    var features = trainSet.GetFeatures(sampleIndex);
                    var labels = trainSet.GetLabels(sampleIndex);

                    var outputs = Calculate(features);

                    for (var index = 0; index < outputs.Length; index++)
                    {
                        var a = outputs[index];
                        var ideal = labels[index];
                        sumError += (ideal - a) * (ideal - a);
                    }

                    BackProp(labels);
                }

                trainDetails.MSE = sumError / trainSet.SampleCount;
                trainDetails.RootMSE = Math.Sqrt(trainDetails.MSE);
                if (epoch % printResultEpoch == 0)
                {
                    Console.WriteLine($"[{epoch}] {trainDetails}.");
                    Debug.WriteLine($"[{epoch}] {trainDetails}.");
                }
            }

            return trainDetails;
        }
    }
}
