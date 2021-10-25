//using System;
//using System.Collections.Generic;
//using System.Linq;
//using System.Text;
//using System.Threading.Tasks;
//using DotNetCraft.NeuralNetwork.Core;
//using DotNetCraft.NeuralNetwork.Domain.Neurons;
//using DotNetCraft.NeuralNetwork.Domain.WeightBuilders;

//namespace DotNetCraft.NeuralNetwork.Domain.Layers
//{
//    public class OutputFullyConnectedLayer: INeuralNetworkLayer
//    {
//        protected readonly OutputNeuron[] _neurons;

//        public int Count => _neurons.Length;

//        public OutputFullyConnectedLayer(int inputNeurons, int neurons, IActivationFunction activationFunction, SimpleWeightBuilder simpleWeightBuilder)
//        {
//            _neurons = new OutputNeuron[neurons];
//            for (int index = 0; index < neurons; index++)
//            {
//                _neurons[index] = new OutputNeuron(inputNeurons, activationFunction, simpleWeightBuilder);
//            }
//        }

//        public void SetInputs(double[] inputs)
//        {
//            foreach (var neuron in _neurons)
//            {
//                neuron.Forward(inputs);
//            }
//        }

//        public double[] GetOutputs()
//        {
//            double[] result = new double[_neurons.Length];
//            for (var index = 0; index < _neurons.Length; index++)
//            {
//                var neuron = _neurons[index];
//                result[index] = neuron.OutputValue;
//            }

//            return result;
//        }

//        public virtual double[] BackProp(double[] prevDeltas)
//        {
//            double[] result = new double[_neurons.Length];
//            for (var index = 0; index < _neurons.Length; index++)
//            {
//                var neuron = _neurons[index];
//                var delta = prevDeltas[index];
//                delta = neuron.BackPropagation(delta);
//                result[index] = delta;
//            }

//            return result;
//        }
//    }
//}
