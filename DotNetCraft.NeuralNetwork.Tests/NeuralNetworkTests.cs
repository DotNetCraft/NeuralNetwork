using System;
using System.Collections.Generic;
using DotNetCraft.NeuralNetwork.Core;
using DotNetCraft.NeuralNetwork.Domain.ActivationFunctions;
using DotNetCraft.NeuralNetwork.Domain.WeightBuilders;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace DotNetCraft.NeuralNetwork.Tests
{
    [TestClass]
    public class NeuralNetworkTests
    {
        private void GetSets(int count, int featureSize, int labelSize, out NeuralNetworkDataSet trainSet, out NeuralNetworkDataSet dataSet)
        {
            var random = new Random();
            trainSet = new NeuralNetworkDataSet(8000, featureSize, labelSize);
            dataSet = new NeuralNetworkDataSet(2000, featureSize, labelSize);
            for (int i = 0; i < count; i++)
            {
                var x1 = random.NextDouble();
                var x2 = random.NextDouble();
                var y = random.NextDouble();//x1 + x2;

                if (i < 8000)
                    trainSet.AddSample(new double[] { x1, x2 }, new double[] { y });
                else
                    dataSet.AddSample(new double[] { x1, x2 }, new double[] { y });
            }
        }

        [TestMethod]
        public void NeuralNetworkFullTest()
        {
            int featureSize = 2;
            int labelSize = 1;

           
            var functions = new Dictionary<ActivationFunctionType, IActivationFunction>
            {
                {ActivationFunctionType.None, new SigmoidActivationFunction()},
                {ActivationFunctionType.Sigmoid, new SigmoidActivationFunction()}
            };
            var activationFunctionFactory = new ActivationFunctionFactory(functions);
            var weightBuilder = new SimpleWeightBuilder();

            var network = new NeuralNetwork(activationFunctionFactory, weightBuilder);
            network.AddInputLayer(2, ActivationFunctionType.None);
            network.AddLayer(4, ActivationFunctionType.Sigmoid);
            network.AddLayer(4, ActivationFunctionType.Sigmoid);
            network.AddLayer(1, ActivationFunctionType.None);

            GetSets(10000, featureSize, labelSize, out var trainSet, out var dataSet);

            var features = dataSet.GetFeatures(0);
            var labels = dataSet.GetLabels(0);
            var result = network.Calculate(features);

            network.Train(trainSet, 1000, 100);

            //var goodPrediction = 0;
            //for (int i = 0; i < dataSet.SampleCount; i++)
            //{
            //    var features = dataSet.GetFeatures(i);
            //    var labels = dataSet.GetLabels(i);

            //    network.Forward(features);
            //    var result = network.GetOutputs();

            //    var delta = Math.Abs(result[0] - labels[0]);
            //    if (delta < 0.0001)
            //        goodPrediction++;
            //}

            //var prediction = 100 * (goodPrediction / (double)dataSet.SampleCount);
            //Console.WriteLine($"Error: {network.RootMSE:F5}; Prediction: {prediction:F3}");
        }
    }
}
