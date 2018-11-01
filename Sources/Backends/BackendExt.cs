
//
//This is modified from KerasSharp repo for use of Unity., by Xiaoxiao Ma, Aalto University, 
//
// Keras-Sharp: C# port of the Keras library
// https://github.com/cesarsouza/keras-sharp
//
// Based under the Keras library for Python. See LICENSE text for more details.
//
//    The MIT License(MIT)
//    
//    Permission is hereby granted, free of charge, to any person obtaining a copy
//    of this software and associated documentation files (the "Software"), to deal
//    in the Software without restriction, including without limitation the rights
//    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//    copies of the Software, and to permit persons to whom the Software is
//    furnished to do so, subject to the following conditions:
//    
//    The above copyright notice and this permission notice shall be included in all
//    copies or substantial portions of the Software.
//    
//    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//    SOFTWARE.
//
namespace KerasSharp.Backends
{
    using KerasSharp.Engine.Topology;
    using System;
    using UnityEngine;
    using System.Linq;

    public static class BackendExt
    {

        public static Tensor normal_probability(this IBackend b, Tensor input, Tensor mean, Tensor variance)
        {
            //probability
            var diff = input - mean;
            var temp1 = -1.0f*diff * diff;
            temp1 = temp1 / (2 * variance);
            temp1 = b.exp( temp1);

            var temp2 = 1.0f / b.sqrt((2 * Mathf.PI) * variance);
            return temp1 * temp2;
        }

        public static Tensor log_normal_probability(this IBackend b, Tensor input, Tensor mean, Tensor variance, Tensor logVariance)
        {
            //log probability
            var temp1 = -0.5f * b.square(input - mean) / variance;

            var temp2 = -0.5f * Mathf.Log(2.0f*Mathf.PI)-0.5f*logVariance;
            return temp1 + temp2;
        }

        /// <summary>
        /// https://github.com/taki0112/Spectral_Normalization-Tensorflow
        /// </summary>
        /// <param name="b"></param>
        /// <param name="w">the weight to be normalized. It need to be at least 2D</param>
        /// <param name="iteration"></param>
        /// <returns>normalized weight</returns>
        public static Tensor spectral_norm(this IBackend b, Tensor w, int iteration = 1)
        {
            using (b.name_scope("SpectralNormalization" + b.get_uid("SpectralNormalization")))
            {
                var wShape = w.shape;
                var uShape = new int[] { 1, wShape[wShape.Length - 1].Value };
                w = b.reshape(w, new int[2] { -1, wShape[wShape.Length - 1].Value });

                var initailzer = new Initializers.RandomNormal();
                var u = b.variable(tensor: initailzer.Call(uShape), name: "U");

                Tensor uHat = u;
                Tensor vHat = null;

                for (int i = 0; i < iteration; ++i)
                {
                    // power iteration
                    //Usually iteration = 1 will be enough
                    var v_ = b.dot(uHat, b.transpose(w, new int[] { 1, 0 }));
                    vHat = b.l2_normalize(v_, 1);

                    var u_ = b.dot(vHat, w);
                    uHat = b.l2_normalize(u_, 1);
                }

                uHat = b.stop_gradient(uHat);
                vHat = b.stop_gradient(vHat);

                var sigma = b.dot(b.dot(vHat, w), b.transpose(uHat, new int[] { 1, 0 }));

                var update = b.update(u, uHat);

                Tensor wNorm = null;
                using (b.dependency(update))
                {
                    wNorm = w / sigma;
                    wNorm = b.reshape(wNorm, wShape.Select(x => x.Value).ToArray());
                }

                return wNorm;
            }
        }
    }
}
