//using System;
//using System.Collections.Generic;
//using System.Linq;
//using System.Text;
//using System.Threading.Tasks;
//using DotNetCraft.NeuralNetwork.Core;

//namespace DotNetCraft.NeuralNetwork.Domain.Neurons
//{
//    public class OutputNeuron: SimpleNeuron
//    {
//        public OutputNeuron(int inputsCount, IActivationFunction activationFunction, IWeightBuilder weightBuilder) : base(inputsCount, activationFunction, weightBuilder)
//        {
//        }

//        #region Overrides of BaseNeuron<double[]>

//        protected override double OnBackPropagation(double idealOutput)
//        {
//            var delta = idealOutput - OutputValue;
//            delta *= _activationFunction.CalculatePrime(OutputValue);
//            delta = base.OnBackPropagation(delta);
//            return delta;
//        }

//        #endregion
//    }
//}
