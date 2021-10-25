//using System;
//using DotNetCraft.NeuralNetwork.Core;

//namespace DotNetCraft.NeuralNetwork.Domain.Neurons
//{
//    public class InputNeuron: BaseNeuron<double>
//    {
//        private readonly IActivationFunction _activationFunction;

//        public InputNeuron(IActivationFunction activationFunction)
//        {
//            if (activationFunction == null) 
//                throw new ArgumentNullException(nameof(activationFunction));

//            _activationFunction = activationFunction;
//        }

//        #region Overrides of BaseNeuron

//        protected override double OnForward(double inputs)
//        {
//            return inputs;
//        }

//        protected override double OnBackPropagation(double prevDelta)
//        {
//            throw new NotImplementedException();
//        }

//        #endregion
//    }
//}
