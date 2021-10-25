using System;
using DotNetCraft.NeuralNetwork.Domain.Neurons;

namespace DotNetCraft.NeuralNetwork.Domain.Synapses
{
    public class Synapse
    {
        //скорость обучения выше 0.00001 приводит к плохой сходимости НС
        public static double E = 0.7;
        public static double Alpha = 0.1;


        private double _prevWeight = 0.0;

        public SimpleNeuron Source { get; }
        public SimpleNeuron Destination { get; }
#if RELEASE
        public double Weight;
        public double DestinationDelta;
#else
        public double Weight { get; private set; }
        public double DestinationDelta { get; private set; }
#endif

        public Synapse(SimpleNeuron source, SimpleNeuron destination, double weight)
        {
            if (source == null) 
                throw new ArgumentNullException(nameof(source));
            if (destination == null) 
                throw new ArgumentNullException(nameof(destination));
            Source = source;
            Destination = destination;
            Weight = weight;
        }

        public void SetDestinationDelta(SimpleNeuron sender, double delta)
        {
            if (sender != Destination)
                throw new ArgumentOutOfRangeException(nameof(sender));

            DestinationDelta = delta;
        }

        public void UpdateWeight(double gradW)
        {
            _prevWeight = E * gradW + Alpha * _prevWeight; 
            Weight += _prevWeight;
        }
    }
}
