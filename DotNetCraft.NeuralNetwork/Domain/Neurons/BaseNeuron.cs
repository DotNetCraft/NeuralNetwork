//namespace DotNetCraft.NeuralNetwork.Domain.Neurons
//{
//    public abstract class BaseNeuron<TInput>
//    {
//        protected double OutputValue;

//        public double Forward(TInput inputs)
//        {
//            OutputValue = OnForward(inputs);
//            return OutputValue;
//        }

//        protected abstract double OnForward(TInput inputs);

//        public double BackPropagation(double prevDelta)
//        {
//            var res = OnBackPropagation(prevDelta);
//            return res;
//        }

//        protected abstract double OnBackPropagation(double prevDelta);
//    }
//}
