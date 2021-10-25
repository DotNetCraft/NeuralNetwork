//using System;
//using System.Collections.Generic;
//using System.Linq;
//using System.Text;
//using System.Threading.Tasks;
//using DotNetCraft.NeuralNetwork.Core;
//using DotNetCraft.NeuralNetwork.Domain.Synapses;

//namespace DotNetCraft.NeuralNetwork.Domain.Layers
//{
//    public class ConvolutionalLayer: SimpleNeuronLayer
//    {
//        public ConvolutionalLayer(int neurons, IActivationFunction activationFunction, IWeightBuilder weightBuilder) : base(neurons, activationFunction, weightBuilder)
//        {
//        }

//        public override void Connect(SimpleNeuronLayer layer)
//        {
//            //Current => layer
//            foreach (var sourceNeuron in _neurons)
//            {
//                foreach (var destinationNeuron in layer._neurons)
//                {
//                    var w = _weightBuilder.BuildWeight();
//                    var synapse = new Synapse(sourceNeuron, destinationNeuron, w);
//                    sourceNeuron.AddForwardSynapse(synapse);
//                    destinationNeuron.AddBackwardSynapse(synapse);
//                }
//            }
//        }
//    }
//}
