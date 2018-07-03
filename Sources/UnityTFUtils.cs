﻿//This is modified from KerasSharp repo for use of Unity., by Xiaoxiao Ma, Aalto University, 
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

using Accord.Math;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Runtime.Serialization;
using TensorFlow;
using UnityEngine;
using System.Linq;

using static KerasSharp.Backends.Current;
using KerasSharp.Engine.Topology;
using KerasSharp.Layers;
using KerasSharp;
using KerasSharp.Metrics;

public static class UnityTFUtils
{
    // Mappings from Python calls to .NET
    static ObjectIDGenerator generator = new ObjectIDGenerator();

    public static int[] Range(int a, int b)
    {
        return Accord.Math.Vector.Range(a, b);
    }

    public static int[] Range(int? a, int? b)
    {
        return Range(a.Value, b.Value);
    }

    public static long GetId(object x)
    {
        if (x == null)
            return 0;

        bool firstTime;
        return generator.GetId(x, out firstTime);
    }

    public static string ToString(object obj)
    {
        if (obj == null)
            return "null";

        if (obj is IEnumerable)
        {
            var l = new List<string>();
            foreach (object o in (IEnumerable)obj)
                l.Add(ToString(o));

            return "[" + string.Join(", ", l.ToArray()) + "]";
        }
        else if (obj is IDictionary)
        {
            var dict = obj as IDictionary;
            var l = new List<string>();
            foreach (object k in dict.Keys)
                l.Add($"{ToString(k)}: {ToString(dict[k])}");

            return "{" + string.Join(", ", l.ToArray()) + "}";
        }

        return obj.ToString();
    }

    public static HashSet<T> ToSet<T>(IEnumerable<T> x)
    {
        if (x == null)
            return null;
        return new HashSet<T>(x);
    }

    /// <summary>
    ///   Input() is used to instantiate a Keras tensor.
    /// </summary>
    /// 
    /// <remarks>
    ///   A Keras tensor is a tensor object from the underlying backend (Theano or TensorFlow), which we 
    ///   augment with certain attributes that allow us to build a Keras model just by knowing the inputs
    ///   and outputs of the model.
    /// </remarks>
    /// 
    /// <param name="shape">A shape tuple (integer), including the batch size. For instance, 
    ///   <c>batch_shape= (10, 32)</c> indicates that the expected input will be batches of 10 32-dimensional
    ///   vectors. <c>batch_shape= (None, 32)</c> indicates batches of an arbitrary number of 32-dimensional 
    ///   vectors.</param>
    /// <param name="batch_shape">The batch shape.</param>
    /// <param name="name">An optional name string for the layer. Should be unique in a model (do not reuse
    ///   the same name twice). It will be autogenerated if it isn't provided.</param>
    /// <param name="dtype">The data type expected by the input, as a string
    ///   (`float32`, `float64`, `int32`...)</param>
    /// <param name="sparse">A boolean specifying whether the placeholder to be created is sparse.</param>
    /// <param name="tensor">Optional existing tensor to wrap into the `Input` layer.
    ///   If set, the layer will not create a placeholder tensor.</param>
    ///   
    public static List<Tensor> Input(int?[] shape = null, int?[] batch_shape = null, string name = null,
        DataType? dtype = null, bool sparse = false, Tensor tensor = null)
    {
        // https://github.com/fchollet/keras/blob/f65a56fb65062c8d14d215c9f4b1015b97cc5bf3/keras/engine/topology.py#L1416

        if (batch_shape == null && tensor == null && shape == null)
        {
            throw new ArgumentException("Please provide to Input either a 'shape' or a 'batch_shape' argument. Note that " +
                "'shape' does not include the batch dimension.");
        }

        if (shape != null && batch_shape == null)
            batch_shape = new int?[] { null }.Concatenate(shape);

        var input_layer = new InputLayer(batch_input_shape: batch_shape,
                                 name: name, dtype: dtype,
                                 sparse: sparse,
                                 input_tensor: tensor);

        // Return tensor including _keras_shape and _keras_history.
        // Note that in this case train_output and test_output are the same pointer.
        return input_layer.inbound_nodes[0].output_tensors;
    }


    /// <summary>
    /// create a tensor based on input array and shape.
    /// If the array is not 1D, the array dimension need to match the shape
    /// </summary>
    /// <param name="array"></param>
    /// <param name="shape"></param>
    /// <param name="start"></param>
    /// <param name="length"></param>
    /// <returns></returns>
    public static TFTensor TFTensorFromArray(Array array, TFShape shape)
    {

        if (array.Rank != 1)
        {
            //check the dimensions
            Debug.Assert(shape.NumDimensions == array.Rank, "the input array is not 1d and the input shape and input array is not compatible");
            for(int i = 0; i < array.Rank; ++i)
            {
                if(array.GetLength(i) != shape[i] && shape[i] >= 0)
                {
                    Debug.LogError("the input shape and input array has no compatible");
                }
            }

            return new TFTensor(array);
        }
        else
        {
            //get the shape based on the tensor and input data length
            long[] actualShape = shape.ToArray();
            if (actualShape == null)
                actualShape = new long[] { array.Length };
            int oneBacthLength = Mathf.Abs((int)actualShape.Aggregate((s, n) => n * s));
            bool hasBatch = false;
            int indexOfBatch = actualShape.IndexOf(-1);
            if (indexOfBatch >= 0)
            {
                actualShape[indexOfBatch] = array.Length / oneBacthLength;
                hasBatch = true;
            }
            Debug.Assert(oneBacthLength <= array.Length, "Feed array does not have enough data");
            Debug.Assert(hasBatch || oneBacthLength == array.Length, "The array length should match the shape's elements count if the shape does not have dynamic axis");



            if (array.GetType().GetElementType() == typeof(float))
            {
                return TFTensor.FromBuffer(shape, (float[])array, 0, oneBacthLength * (array.Length / oneBacthLength));
            }
            else if (array.GetType().GetElementType() == typeof(double))
            {
                return TFTensor.FromBuffer(shape, (double[])array, 0, oneBacthLength * (array.Length / oneBacthLength));
            }
            else if (array.GetType().GetElementType() == typeof(int))
            {
                return TFTensor.FromBuffer(shape, (int[])array, 0, oneBacthLength * (array.Length / oneBacthLength));
            }
            else
            {
                Debug.LogError("Data type of _constant input " + array.GetType().GetElementType() + " is not support with shape input");
                return null;
            }
        }
    }

    public static TFTensor TFTensorFromT<T>(T value)
    {
        if (typeof(T) == typeof(float))
        {
           return new TFTensor((float)Convert.ChangeType(value, typeof(float)));
        }
        else if (typeof(T) == typeof(double))
        {
            return new TFTensor((double)Convert.ChangeType(value, typeof(double)));
        }
        else if (typeof(T) == typeof(int))
        {
            return new TFTensor((int)Convert.ChangeType(value, typeof(int)));
        }
        else
        {
            Debug.LogError("Does not Support Constant of type" + typeof(T).Name);
            return null;
        }
    }

    public static TValue TryGetOr<TKey, TValue>(this IDictionary<TKey, TValue> dict, TKey key, TValue def)
    {
        TValue r;
        if (dict.TryGetValue(key, out r))
            return r;
        return def;
    }

    // Methods to condense single elements and lists into dictionary
    // so they can be passed more easily along methods that follow
    // the Python interfaces. We include some markings to be able
    // to detect what those values originally were before being
    // transformed to dictionaries.

    public static Dictionary<string, T> dict_from_single<T>(this T value)
    {
        return new Dictionary<string, T>() { { "__K__single__", value } };
    }

    public static Dictionary<string, T> dict_from_list<T>(this List<T> list)
    {
        var dict = new Dictionary<string, T>();
        for (int i = 0; i < list.Count; i++)
            dict["__K__list__" + i] = list[i];
        return dict;
    }

    public static bool is_dict<T>(this Dictionary<string, T> dict)
    {
        if (dict == null)
            return false;
        return !dict.Keys.Any(x => x.StartsWith("__K__"));
    }

    public static bool is_list<T>(this Dictionary<string, T> dict)
    {
        if (dict == null)
            return false;
        return dict.Keys.All(x => x.StartsWith("__K__list__"));
    }

    public static bool is_single<T>(this Dictionary<string, T> dict)
    {
        if (dict == null)
            return true;
        return dict.Keys.Count == 1 && dict.ContainsKey("__K__single__");
    }

    public static List<T> to_list<T>(this Dictionary<string, T> dict)
    {
        List<T> list = new List<T>();
        for (int i = 0; i < dict.Keys.Count; i++)
            list.Add(dict["__K__list__" + i]);
        return list;
    }

    public static T to_single<T>(this Dictionary<string, T> dict)
    {
        if (dict == null)
            return default(T);
        return dict["__K__single__"];
    }



    public static List<T> Get<T>(IList<string> name)
    {
        if (name == null)
            return null;
        return name.Select(s => Get<T>(s)).ToList();
    }

    public static T Get<T>(string name)
    {
        Type baseType = typeof(T);
        Type foundType = get(name, baseType);

        if (foundType == null)
            foundType = get(name.Replace("_", ""), baseType); // try again after normalizing the name

        if (foundType == null)
            throw new ArgumentOutOfRangeException("name", $"Could not find {baseType.Name} '{name}'.");

        return (T)Activator.CreateInstance(foundType);
    }

    private static Type get(string name, Type type)
    {
        return AppDomain.CurrentDomain.GetAssemblies()
                                    .SelectMany(s => s.GetTypes())
                                    .Where(p => type.IsAssignableFrom(p) && !p.IsInterface)
                                    .Where(p => p.Name.ToUpperInvariant() == name.ToUpperInvariant())
                                    .FirstOrDefault();
    }

    public static List<T> Get<T>(IEnumerable<object> name)
        where T : class
    {
        return name.Select(s => Get<T>(s)).ToList();
    }

    public static T Get<T>(object obj)
        where T : class
    {
        if (obj is T)
            return (T)obj;

        if (obj is String)
            return Get<T>(obj as string);

        if (typeof(T) == typeof(IMetric))
        {
            var f2 = obj as Func<Tensor, Tensor, Tensor>;
            if (f2 != null)
                return new CustomMetric(f2) as T;

            var f3 = obj as Func<Tensor, Tensor, Tensor, Tensor>;
            if (f3 != null)
                return new CustomMetric(f3) as T;
        }

        throw new NotImplementedException();
    }



}


public class ConvUtils
{
    /// <summary>
    ///   Transforms a single int or iterable of ints into an int tuple.
    /// </summary>
    /// <param name="value">The value to validate and convert. Could an int, or any iterable of ints.</param>
    /// <param name="n">The size of the tuple to be returned.</param>
    /// <param name="name">The name of the argument being validated, e.g. "strides" or "kernel_size".This is only used to format error messages.</param>
    /// <returns>System.Object.</returns>
    internal int[] NormalizeTuple(int value, int n, string name)
    {
        return Vector.Create<int>(size: n, value: value);
    }


    internal object NormalizeDataFormat(DataFormatType? value)
    {
        // https://github.com/fchollet/keras/blob/f65a56fb65062c8d14d215c9f4b1015b97cc5bf3/keras/utils/conv_utils.py#L46

        if (value == null)
            value = K.image_data_format();

        return value;
    }

    /// <summary>
    ///   Determines output length of a convolution given input length.
    /// </summary>
    /// 
    public static int? ConvOutputLength(int? input_length, int filter_size, PaddingType padding, int stride, int dilation = 1)
    {
        if (input_length == null)
            return null;
        int dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1);
        int output_length = 0;
        if (padding == PaddingType.Same)
            output_length = input_length.Value;
        else if (padding == PaddingType.Valid)
            output_length = input_length.Value - dilated_filter_size + 1;
        else if (padding == PaddingType.Causal)
            output_length = input_length.Value;
        else if (padding == PaddingType.Full)
            output_length = input_length.Value + dilated_filter_size - 1;
        else
            throw new Exception();
        return (output_length + stride - 1) / stride;
    }
}
