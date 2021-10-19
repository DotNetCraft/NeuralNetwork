using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using DotNetCraft.NeuralNetwork.Core;

namespace DotNetCraft.NeuralNetwork.Domain.Neurons
{
    public abstract class BaseNeuron<TInput>
    {
        public double OutputValue { get; private set; }

        public void Forward(TInput inputs)
        {
            OutputValue = OnForward(inputs);
        }

        protected abstract double OnForward(TInput inputs);
    }
}
